"""EvidenceSelection ("Evidence Basket"): what to feed the LLM, decoupled
from what to do with it.

Selecting transitions is deliberately independent of prompting/generation:
a researcher builds up a basket from any number of episodes/ranges first,
then opens it in Prompt Studio. Stable references (episode id + step range,
or explicit transition ids) are stored, not copies of the data, so a basket
always reflects the current transition records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.experience import ExperienceStore
from storage.database import Database
from storage.models import EvidenceSelection, EvidenceSelectionItem, Transition

ACTIVE_BASKET_NAME = "__basket__"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def basket_label(selection: EvidenceSelection) -> str:
    """Display name for a basket -- the always-present default basket is
    stored under an internal name but shown as "Default"."""
    return "Default" if selection.name == ACTIVE_BASKET_NAME else selection.name


class EvidenceBasket:
    """CRUD over one session's :class:`EvidenceSelection` rows/items, plus
    resolving a selection's items into the concrete transitions they name."""

    def __init__(self, db: Database, session_id: str):
        self.db = db
        self.session_id = session_id

    def get_or_create_active(self) -> EvidenceSelection:
        """The always-on working basket the Play/Episodes views add to and
        Prompt Studio reads by default (as distinct from named, saved
        selections created explicitly)."""
        row = self.db.query_one(
            "SELECT * FROM evidence_selections WHERE session_id = ? AND name = ? "
            "ORDER BY id DESC LIMIT 1",
            (self.session_id, ACTIVE_BASKET_NAME),
        )
        if row:
            return EvidenceSelection.from_row(row)
        return self.create(ACTIVE_BASKET_NAME)

    def create(self, name: str, metadata: Optional[dict] = None) -> EvidenceSelection:
        selection = EvidenceSelection(id=None, session_id=self.session_id, name=name,
                                       created_at=_now(), metadata=metadata or {})
        selection.id = self.db.insert("evidence_selections", selection.to_row())
        return selection

    def get(self, selection_id: int) -> Optional[EvidenceSelection]:
        row = self.db.get("evidence_selections", "id", selection_id)
        return EvidenceSelection.from_row(row) if row else None

    def list_selections(self) -> list[EvidenceSelection]:
        rows = self.db.query(
            "SELECT * FROM evidence_selections WHERE session_id = ? AND name != ? ORDER BY id DESC",
            (self.session_id, ACTIVE_BASKET_NAME),
        )
        return [EvidenceSelection.from_row(r) for r in rows]

    def list_all(self) -> list[EvidenceSelection]:
        """Every basket in this session -- the always-present default one
        (auto-created if this is the first call) plus any named ones,
        oldest first. For basket-switcher UIs that want a single list to
        choose from."""
        self.get_or_create_active()
        rows = self.db.query(
            "SELECT * FROM evidence_selections WHERE session_id = ? ORDER BY id",
            (self.session_id,),
        )
        return [EvidenceSelection.from_row(r) for r in rows]

    def rename(self, selection: EvidenceSelection, new_name: str) -> EvidenceSelection:
        if selection.name == ACTIVE_BASKET_NAME:
            raise ValueError("The default basket cannot be renamed.")
        if not new_name or new_name == ACTIVE_BASKET_NAME:
            raise ValueError(f"{new_name!r} is not a valid basket name.")
        selection.name = new_name
        self.db.update("evidence_selections", "id", selection.to_row())
        return selection

    def delete(self, selection: EvidenceSelection) -> None:
        """Delete a named basket and its items. The default basket cannot
        be deleted (it always exists so Play/Episodes always have
        somewhere to add to)."""
        if selection.name == ACTIVE_BASKET_NAME:
            raise ValueError("The default basket cannot be deleted.")
        self.clear(selection)
        self.db.execute("DELETE FROM evidence_selections WHERE id = ?", (selection.id,))

    # -- adding evidence -------------------------------------------------

    def add_transition(self, selection: EvidenceSelection, episode_id: int, transition_id: int,
                        source_description: str = "") -> EvidenceSelectionItem:
        item = EvidenceSelectionItem(id=None, selection_id=selection.id, kind="transition",
                                      episode_id=episode_id, transition_id=transition_id,
                                      source_description=source_description, added_at=_now())
        item.id = self.db.insert("evidence_selection_items", item.to_row())
        return item

    def add_range(self, selection: EvidenceSelection, episode_id: int, start_step: int,
                  end_step: int, source_description: str = "") -> EvidenceSelectionItem:
        item = EvidenceSelectionItem(id=None, selection_id=selection.id, kind="range",
                                      episode_id=episode_id, start_step=start_step,
                                      end_step=end_step, source_description=source_description,
                                      added_at=_now())
        item.id = self.db.insert("evidence_selection_items", item.to_row())
        return item

    def add_episode(self, selection: EvidenceSelection, episode_id: int,
                     source_description: str = "") -> EvidenceSelectionItem:
        item = EvidenceSelectionItem(id=None, selection_id=selection.id, kind="episode",
                                      episode_id=episode_id, source_description=source_description,
                                      added_at=_now())
        item.id = self.db.insert("evidence_selection_items", item.to_row())
        return item

    # -- removing --------------------------------------------------------

    def remove_item(self, item_id: int) -> None:
        self.db.execute("DELETE FROM evidence_selection_items WHERE id = ?", (item_id,))

    def clear(self, selection: EvidenceSelection) -> None:
        self.db.execute("DELETE FROM evidence_selection_items WHERE selection_id = ?", (selection.id,))

    # -- reading -----------------------------------------------------------

    def list_items(self, selection: EvidenceSelection) -> list[EvidenceSelectionItem]:
        rows = self.db.query(
            "SELECT * FROM evidence_selection_items WHERE selection_id = ? ORDER BY id",
            (selection.id,),
        )
        return [EvidenceSelectionItem.from_row(r) for r in rows]

    def resolve_transitions(self, selection: EvidenceSelection,
                             experience: ExperienceStore) -> list[Transition]:
        """Expand every item into concrete transitions, de-duplicated by id
        (order preserved, first occurrence wins)."""
        seen: set[int] = set()
        resolved: list[Transition] = []
        for item in self.list_items(selection):
            if item.kind == "transition":
                candidates = [experience.get_transition(item.transition_id)]
            elif item.kind == "range":
                candidates = [
                    t for t in experience.get_transitions(item.episode_id)
                    if item.start_step <= t.step_index <= item.end_step
                ]
            else:  # "episode"
                candidates = experience.get_transitions(item.episode_id)
            for t in candidates:
                if t is not None and t.id not in seen:
                    seen.add(t.id)
                    resolved.append(t)
        return resolved

    def resolve_episode_ids(self, selection: EvidenceSelection) -> list[int]:
        return sorted({item.episode_id for item in self.list_items(selection)})

    def count(self, selection: EvidenceSelection, experience: ExperienceStore) -> int:
        return len(self.resolve_transitions(selection, experience))

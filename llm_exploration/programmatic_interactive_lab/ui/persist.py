"""Keep a page's config widgets at their last-set value across navigations.

NiceGUI's ``@ui.page`` reruns a page's whole ``render()`` on every visit,
so a plain local widget with a hardcoded ``value=`` literal always resets
to that literal. ``persist()`` two-way-binds a widget's value to a key in
a dict living in :mod:`ui.state` (process-wide, survives navigation) --
NiceGUI's own ``bind_value`` already seeds the widget from the dict when
the key is present and seeds the dict from the widget otherwise (the
backward direction wins on initial sync, see ``nicegui.binding.bind``), so
this is just that plus optional validation for selects whose options are
rebuilt per-render (a node/edge/model deleted since the value was stored)
so a stale stored value never gets displayed for an option that no longer
exists.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


def persist(widget, store: dict, key: str, valid_values: Optional[Iterable[Any]] = None) -> None:
    """Bind ``widget.value`` to ``store[key]``, seeding whichever side is
    missing from the other. ``valid_values``, if given, drops a stored
    value (or, for a multi-select's list value, its no-longer-valid
    entries) that isn't one of the widget's *current* options, so the
    widget falls back to its own literal default instead of showing a
    selection that no longer exists."""
    if valid_values is not None and key in store:
        valid_set = set(valid_values)
        stored = store[key]
        if isinstance(stored, list):
            store[key] = [v for v in stored if v in valid_set]
        elif stored not in valid_set:
            del store[key]
    widget.bind_value(store, key)

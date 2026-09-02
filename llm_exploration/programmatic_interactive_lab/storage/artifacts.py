"""Filesystem layout for large/raw artifacts (states, renders, node code
dumps, exports) referenced by path from SQLite rows.

Layout (mirrors the repo's existing ``outputs/<env>/...`` convention of a
predictable, inspectable directory tree rather than an opaque blob store)::

    programmatic_interactive_lab/data/
        database.sqlite
        sessions/<session-id>/
            states/<episode_id>/<step_index>_(state|next_state).json
            renders/<episode_id>/<step_index>.txt
            nodes/<node_id>.py
            exports/...
"""

from __future__ import annotations

from pathlib import Path


class ArtifactStore:
    """Resolves and creates paths under one session's data directory."""

    def __init__(self, data_root: Path | str, session_id: str):
        self.session_root = Path(data_root) / "sessions" / session_id
        self.states_dir = self.session_root / "states"
        self.renders_dir = self.session_root / "renders"
        self.nodes_dir = self.session_root / "nodes"
        self.exports_dir = self.session_root / "exports"
        for d in (self.states_dir, self.renders_dir, self.nodes_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def state_path(self, episode_id: int, step_index: int, which: str) -> Path:
        """``which`` is ``"state"`` or ``"next_state"``."""
        d = self.states_dir / str(episode_id)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{step_index}_{which}.json"

    def render_path(self, episode_id: int, step_index: int) -> Path:
        d = self.renders_dir / str(episode_id)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{step_index}.txt"

    def node_code_path(self, node_id: int) -> Path:
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        return self.nodes_dir / f"{node_id}.py"

    def write_text(self, path: Path, text: str) -> str:
        path.write_text(text)
        return str(path)

    def read_text(self, path: str | Path) -> str:
        return Path(path).read_text()

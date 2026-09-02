"""User-managed registry of selectable LLM models.

Distinct from ``core.llm.LLM_PRESETS`` (the legacy env-var-driven default
used by ``--llm`` at launch, kept for backward compatibility): this file
lets a researcher list any number of models -- each with its own
model name, endpoint URL, and API key written directly into the file -- and
pick among them per call in Prompt Studio, no relaunch or env vars needed.

To add a model: copy ``llm_models.example.json`` (checked into the repo) to
``llm_models.json`` (at the package root, next to this file's parent
directory -- git-ignored since it holds API keys) and add one entry per
model::

    [
      {"name": "gemini-flash", "model_name": "gemini-2.5-flash",
       "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
       "api_key": "..."}
    ]

``name`` is the label shown in the picker (must be unique); ``model_name``/
``url``/``api_key`` map directly to ``llm.client.LLMClient``'s ``model``/
``base_url``/``api_key``. Optional per-entry ``temperature``/``timeout``/
``max_retries``/``stream`` override this lab's defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

MODELS_FILE = Path(__file__).resolve().parent.parent / "llm_models.json"


def list_llm_models() -> list[dict]:
    """Every configured model entry, in file order. Empty (not an error) if
    ``llm_models.json`` doesn't exist yet -- nothing configured."""
    if not MODELS_FILE.exists():
        return []
    return json.loads(MODELS_FILE.read_text())


def get_llm_model(name: str) -> Optional[dict]:
    """The entry whose ``name`` matches, or ``None`` if there isn't one."""
    for entry in list_llm_models():
        if entry.get("name") == name:
            return entry
    return None

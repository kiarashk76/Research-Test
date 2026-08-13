"""Smoke test: is the LLM endpoint reachable and responding?

Run from the repo root:

    LLM_MODEL=... LLM_BASE_URL=... LLM_API_KEY=... python -m llm.test_api

Credentials are read from environment variables so no secret lives in this file.
Prints the model's reply to a trivial "hello world" prompt (and the raw error if
the call fails).
"""

from __future__ import annotations

import os

from .client import LLMClient


MODEL = os.environ.get("VULCAN_MODEL") # GEMINI_MODEL
BASE_URL = os.environ.get("VULCAN_BASE_URL") # GEMINI_BASE_URL
API_KEY = os.environ.get("VULCAN_API_KEY") # GEMINI_API_KEY


def main() -> None:
    print(f"model={MODEL!r} base_url={BASE_URL!r}")
    client = LLMClient(model=MODEL, base_url=BASE_URL, api_key=API_KEY,
                       timeout=30, max_retries=2, stream=True)
    try:
        reply = client.ask("Reply with exactly: hello world")
    except Exception as e:                        # noqa: BLE001 - surface any API error
        print(f"FAILED: {type(e).__name__}: {e}")
        return
    print(f"OK, reply: {reply!r}")


if __name__ == "__main__":
    main()

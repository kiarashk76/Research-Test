"""CLI configuration for launching the lab.

Follows the parent repo's ``--env``/``--env-overrides`` JSON-string
convention (see the root ``main.py``) rather than inventing a new one, but
resolves against this package's own ``core.environment.ENV_CONFIGS`` and
``core.llm.LLM_PRESETS`` registries -- this package does not import the
root ``config.py``.
"""

from __future__ import annotations

import argparse
import json

from core.environment import available_environment_names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="programmatic_interactive_lab",
        description="Interactive Programmatic Policy Lab: play, collect evidence, "
                     "generate and run LLM-written policies, all with full provenance.",
    )
    parser.add_argument("--env", default=None, choices=available_environment_names(),
                         help="Environment to launch (from this lab's own ENV_CONFIGS). Omit to "
                              "pick one interactively in-app instead (the Setup screen) -- ignored "
                              "if --session-id is given, since that session's environment is "
                              "already fixed.")
    parser.add_argument("--env-overrides", default="{}",
                         help="JSON dict merged onto the environment's default params.")
    parser.add_argument("--llm", default="GEMINI",
                         help="LLM preset name from this lab's own LLM_PRESETS.")
    parser.add_argument("--llm-overrides", default="{}",
                         help="JSON dict merged onto the LLM preset's default params.")
    parser.add_argument("--session-name", default=None,
                         help="Name for a newly created session (default: '<env> session').")
    parser.add_argument("--session-id", default=None,
                         help="Reopen an existing session by id instead of creating one.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--reload", action="store_true", help="Enable NiceGUI's auto-reload.")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def parse_json_overrides(text: str) -> dict:
    return json.loads(text) if text else {}

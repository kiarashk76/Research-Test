"""Shared environment-parameter widgets: one input per ``ENV_CONFIGS``
param, its type inferred from the registry default's own Python type --
used by both Setup (new session) and Queue (new queued experiment), so a
new environment/param needs no UI code in either place, just an entry in
``core.environment.ENV_CONFIGS``.
"""

from __future__ import annotations

from typing import Callable, Optional

from nicegui import ui

from core.environment import ChoiceParam


def param_widget(key: str, default_value):
    """A plain value, or a :class:`ChoiceParam` for a fixed set of choices
    (e.g. MiniHack's ``env_id``/OC_Atari's ``game_name``)."""
    if isinstance(default_value, ChoiceParam):
        return ui.select(default_value.choices, value=default_value.default, label=key).classes("w-64")
    if isinstance(default_value, bool):
        return ui.checkbox(key, value=default_value)
    if isinstance(default_value, int):
        return ui.number(key, value=default_value, format="%d").classes("w-40")
    if isinstance(default_value, float):
        return ui.number(key, value=default_value).classes("w-40")
    return ui.input(key, value=str(default_value)).classes("w-40")


def coerce(default_value, raw):
    """Casts a widget's current value back to the type the registry
    default had, so overrides stay JSON/constructor-compatible (e.g.
    ``size`` stays an ``int``, never drifts to a ``float`` because a
    number input returned one)."""
    if isinstance(default_value, ChoiceParam):
        return str(raw)
    if isinstance(default_value, bool):
        return bool(raw)
    if isinstance(default_value, int):
        return int(raw)
    if isinstance(default_value, float):
        return float(raw)
    return str(raw)


def render_params(params_container, param_widgets: dict, params: dict,
                   max_episode_steps_default_for: Optional[Callable[[str], int]] = None) -> None:
    """Fills ``params_container`` with one widget per ``params`` entry
    (mutating ``param_widgets`` in place) and, if the env's ``ENV_CONFIGS``
    entry supplies ``max_episode_steps_default_for`` (e.g. any of the
    MiniHack families -- see ``environments/minihack_room_env.py::
    default_max_episode_steps_for`` and its siblings), wires its
    ``env_id`` dropdown to keep ``max_episode_steps``'s displayed default
    in sync rather than silently leaving whichever variant's default
    showed first."""
    params_container.clear()
    param_widgets.clear()
    with params_container:
        ui.label("Environment parameters").classes("font-bold text-sm")
        with ui.row().classes("items-center gap-4 flex-wrap"):
            for key, default_value in params.items():
                param_widgets[key] = param_widget(key, default_value)

    if max_episode_steps_default_for and "env_id" in param_widgets and "max_episode_steps" in param_widgets:
        def sync_max_episode_steps() -> None:
            param_widgets["max_episode_steps"].value = (
                max_episode_steps_default_for(param_widgets["env_id"].value))

        param_widgets["env_id"].on_value_change(sync_max_episode_steps)

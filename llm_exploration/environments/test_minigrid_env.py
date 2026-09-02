"""Tests for MiniGridEnv's SymbolicImageWrapper -- decoded 'image'
(x, y, description) list, decoded 'direction', and the door-only state
exception."""

from __future__ import annotations

import pytest

minigrid = pytest.importorskip("minigrid")

from environments.minigrid_env import MiniGridEnv


def test_image_omits_empty_and_unseen_cells():
    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    objects = {description.split()[-1] for _x, _y, description in obs["image"]}
    assert "empty" not in objects
    assert "unseen" not in objects


def test_door_gets_state_other_objects_do_not():
    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    descriptions = [description for _x, _y, description in obs["image"]]

    door = next(d for d in descriptions if "door" in d)
    color, kind, state = door.split()
    assert kind == "door"
    assert state in ("open", "closed", "locked")

    # Only doors get a state word appended -- every other object type's
    # state byte is always 0 in MiniGrid's own encoding, so it's never
    # meaningful to show.
    assert "yellow key" in descriptions
    assert "red agent" in descriptions
    assert "green goal" in descriptions


def test_direction_is_one_of_the_four_compass_words():
    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    assert obs["direction"] in ("right", "down", "left", "up")


def test_mission_passed_through_unmodified():
    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    assert isinstance(obs["mission"], str)
    assert "key" in obs["mission"] and "door" in obs["mission"]


def test_partial_observability_still_decodes():
    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0", full_observability=False)
    obs, _ = env.reset(seed=0)
    assert isinstance(obs["image"], list)
    for x, y, description in obs["image"]:
        assert isinstance(x, int) and isinstance(y, int)
        assert isinstance(description, str)


def test_step_keeps_the_same_decoded_shape():
    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0")
    env.reset(seed=0)
    obs, _reward, _terminated, _truncated, _info = env.step(2)  # move forward
    assert isinstance(obs["image"], list)
    assert obs["direction"] in ("right", "down", "left", "up")
    assert isinstance(obs["mission"], str)


def test_rendered_text_present_under_full_observability():
    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0", full_observability=True)
    obs, _ = env.reset(seed=0)
    assert isinstance(obs["rendered_text"], str)
    assert obs["rendered_text"] == env.render()

    obs, _reward, _terminated, _truncated, _info = env.step(2)  # move forward
    assert obs["rendered_text"] == env.render()


def test_rendered_text_absent_under_partial_observability():
    env = MiniGridEnv("MiniGrid-DoorKey-5x5-v0", full_observability=False)
    obs, _ = env.reset(seed=0)
    assert "rendered_text" not in obs

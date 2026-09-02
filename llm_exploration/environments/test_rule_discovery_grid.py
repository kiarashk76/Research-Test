"""Tests for RuleDiscoveryGridEnv's reward_shaping flag and the
step-count-related terminal Y reward (added regardless of the flag)."""

from __future__ import annotations

import numpy as np

from environments.rule_discovery_grid import RuleDiscoveryGridEnv


def _place(env: RuleDiscoveryGridEnv, agent, a, b, x) -> None:
    """Force a fixed layout after reset() so tests can script exact moves."""
    env.agent_pos = np.array(agent, dtype=np.int64)
    env.a_pos = np.array(a, dtype=np.int64)
    env.b_pos = np.array(b, dtype=np.int64)
    env.x_pos = np.array(x, dtype=np.int64)
    env.y_pos = None
    env.a_on = False
    env._a_bonus_given = False
    env._b_bonus_given = False
    env.reward_value = (
        env._manhattan(env.agent_pos, env.a_pos)
        + env._manhattan(env.a_pos, env.b_pos)
        + env._manhattan(env.b_pos, env.x_pos)
    )


def _reset_env(reward_shaping: bool) -> RuleDiscoveryGridEnv:
    env = RuleDiscoveryGridEnv(max_steps=100, size=10, reward_shaping=reward_shaping)
    env.reset(seed=0)
    # Layout: agent at (0,0), a at (0,1) [adjacent], b at (0,3), x at (5,5).
    _place(env, agent=(0, 0), a=(0, 1), b=(0, 3), x=(5, 5))
    return env


def test_a_bonus_given_once_when_reward_shaping_on():
    env = _reset_env(reward_shaping=True)
    _, reward1, _, _, _ = env.step(4)  # koba adjacent to a: turns a_on True, first time
    assert env.a_on is True
    assert reward1 == 1 - 1  # +1 bonus, -1 step cost

    _, reward2, _, _, _ = env.step(4)  # koba again: turns a_on False
    assert env.a_on is False
    assert reward2 == -1  # no bonus this time

    _, reward3, _, _, _ = env.step(4)  # koba again: turns a_on True again
    assert env.a_on is True
    assert reward3 == -1  # still no bonus -- already given once


def test_no_a_bonus_when_reward_shaping_off():
    env = _reset_env(reward_shaping=False)
    _, reward, _, _, _ = env.step(4)  # koba adjacent to a
    assert env.a_on is True
    assert reward == -1  # no shaping bonus applied


def test_b_bonus_requires_a_on_first_and_fires_once():
    env = RuleDiscoveryGridEnv(max_steps=100, size=10, reward_shaping=True)
    env.reset(seed=0)
    # agent at (0,3), adjacent to b at (0,4); a is off (at (5,5), irrelevant).
    _place(env, agent=(0, 3), a=(9, 9), b=(0, 4), x=(2, 2))

    _, reward, _, _, _ = env.step(4)  # koba near b, but a_on is False
    assert env.x_pos is not None  # nothing converted
    assert reward == -1  # no bonus

    env.a_on = True  # simulate having already done koba on a
    _, reward2, terminated, _, _ = env.step(4)  # koba near b, a_on True -> converts x to y
    assert env.x_pos is None
    assert env.y_pos is not None
    assert reward2 == 1 - 1  # +1 bonus, -1 step cost
    assert not terminated  # agent isn't standing on y yet

    _, reward3, _, _, _ = env.step(4)  # koba near b again -- nothing left to convert
    assert reward3 == -1  # no repeat bonus


def test_terminal_reward_scales_with_reward_value_regardless_of_flag():
    for shaping in (False, True):
        env = RuleDiscoveryGridEnv(max_steps=100, size=10, reward_shaping=shaping)
        env.reset(seed=0)
        _place(env, agent=(0, 0), a=(9, 9), b=(9, 8), x=(3, 3))
        env.a_on = True
        env.y_pos = env.x_pos.copy()
        env.x_pos = None
        env.agent_pos = np.array([3, 4], dtype=np.int64)  # adjacent to y at (3,3)

        _, reward, terminated, _, _ = env.step(2)  # move left onto y
        assert terminated
        assert reward == env.reward_value - 1
        assert env.reward_value > 1  # sanity: not just the old flat +1


def test_reward_value_computed_at_reset_from_initial_positions():
    env = RuleDiscoveryGridEnv(max_steps=100, size=10, reward_shaping=False)
    env.reset(seed=0)
    _place(env, agent=(0, 0), a=(0, 2), b=(0, 5), x=(0, 9))
    assert env.reward_value == 2 + 3 + 4

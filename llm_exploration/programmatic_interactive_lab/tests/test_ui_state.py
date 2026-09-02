from __future__ import annotations

from app import build_context
from core.session import SessionManager
from ui import state

VALID_SOURCE = "def policy(observation):\n    return int(np.sum(observation)) % 4\n"
INVALID_SOURCE = "def not_policy(observation):\n    return 0\n"
RAISES_SOURCE = "def policy(observation):\n    return [][0]\n"


def _make_context(db, tmp_path):
    session = SessionManager(db).create("s", "SimpleGridEnv", {"size": 5, "max_steps": 20})
    return build_context(db, session, data_root=tmp_path / "data")


def test_default_controller_is_human(db, tmp_path):
    context = _make_context(db, tmp_path)
    state.set_context(context)
    assert state.get_play_controller() == ("human", None)
    assert state.get_play_session().actor_type == "human"


def test_switch_to_policy_and_reset_uses_policy_actor(db, tmp_path):
    context = _make_context(db, tmp_path)
    state.set_context(context)
    policy = context.nodes.create("p", VALID_SOURCE)

    assert state.set_play_controller("node", policy.id)
    assert state.get_play_controller() == ("node", policy.id)

    session = state.reset_play_session(seed=0)
    assert session.actor_type == "node"
    assert session.actor_id == str(policy.id)


def test_invalid_policy_reports_not_ready_and_falls_back_to_human(db, tmp_path):
    context = _make_context(db, tmp_path)
    state.set_context(context)
    policy = context.nodes.create("bad", INVALID_SOURCE)

    assert not state.set_play_controller("node", policy.id)
    assert state.play_runner_error() is not None

    session = state.reset_play_session(seed=0)
    assert session.actor_type == "human"  # falls back since the runner never became ready


def test_step_play_policy_advances_episode(db, tmp_path):
    context = _make_context(db, tmp_path)
    state.set_context(context)
    policy = context.nodes.create("p", VALID_SOURCE)
    state.set_play_controller("node", policy.id)
    state.reset_play_session(seed=0)

    transition, result, error = state.step_play_policy()
    assert error is None
    assert transition.actor_type == "node"
    assert transition.actor_id == str(policy.id)
    assert result.reward == transition.reward


def test_step_play_policy_raises_without_active_policy(db, tmp_path):
    context = _make_context(db, tmp_path)
    state.set_context(context)
    state.reset_play_session(seed=0)  # human mode

    try:
        state.step_play_policy()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_switching_policy_mid_episode_takes_effect_immediately(db, tmp_path):
    """Switching the controller to a different policy WITHOUT resetting must
    take effect on the very next step, and each transition must record
    exactly which policy actually produced it (no stale attribution to
    whichever policy the episode started with)."""
    context = _make_context(db, tmp_path)
    state.set_context(context)
    policy_a = context.nodes.create("a", VALID_SOURCE)
    policy_b = context.nodes.create("b", VALID_SOURCE)

    state.set_play_controller("node", policy_a.id)
    session = state.reset_play_session(seed=0)
    transition, _, _ = state.step_play_policy()
    assert transition.actor_id == str(policy_a.id)

    # Switch to policy B *without* resetting -- the same episode continues.
    assert state.set_play_controller("node", policy_b.id)
    transition, _, _ = state.step_play_policy()
    assert transition.actor_id == str(policy_b.id)
    assert transition.episode_id == session.episode.id  # still the same episode

    # The episode itself is now marked mixed-actor since two different
    # policies produced transitions within it.
    reloaded = context.experience.get_episode(session.episode.id)
    assert reloaded.actor_type == "mixed"


def test_switching_between_human_and_policy_mid_episode(db, tmp_path):
    context = _make_context(db, tmp_path)
    state.set_context(context)
    policy = context.nodes.create("p", VALID_SOURCE)

    session = state.reset_play_session(seed=0)  # starts as human
    session.step(context.adapter.sample_action(), actor_type="human", actor_id="human")

    assert state.set_play_controller("node", policy.id)
    transition, _, _ = state.step_play_policy()
    assert transition.actor_type == "node"
    assert transition.actor_id == str(policy.id)

    reloaded = context.experience.get_episode(session.episode.id)
    assert reloaded.actor_type == "mixed"


def test_step_play_policy_records_execution_error_metadata_and_tag(db, tmp_path):
    """When the live policy errors mid-step, the fallback transition itself
    -- not just a separate PolicyExecutionError row -- carries the error in
    its metadata and gets auto-tagged, so it can be selected as evidence for
    the {{execution_error}} placeholder later without retyping anything."""
    context = _make_context(db, tmp_path)
    state.set_context(context)
    policy = context.nodes.create("broken", RAISES_SOURCE)
    state.set_play_controller("node", policy.id)
    state.reset_play_session(seed=0)

    transition, _, error = state.step_play_policy()
    assert error is not None
    assert transition.metadata["execution_error"]["error_type"] == "IndexError"
    assert context.experience.get_tags(transition_id=transition.id) == ["execution-error"]


def test_switching_context_closes_runner_and_resets_controller(db, tmp_path):
    context_a = _make_context(db, tmp_path)
    state.set_context(context_a)
    policy = context_a.nodes.create("p", VALID_SOURCE)
    state.set_play_controller("node", policy.id)

    context_b = _make_context(db, tmp_path)
    state.set_context(context_b)
    assert state.get_play_controller() == ("human", None)

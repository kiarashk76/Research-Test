from __future__ import annotations

from core.runs import RunConfig

VALID_SOURCE = "def policy(observation, memory):\n    return int(np.sum(observation)) % 4\n"
# See test_policy_runner.py for why this uses an implicit interpreter error
# instead of `raise ValueError(...)` (exception classes aren't whitelisted).
RAISES_SOURCE = "def policy(observation, memory):\n    return [][0]\n"


def test_run_policy_creates_run_episodes_and_transitions(policy_store, run_manager, experience):
    node = policy_store.create("navigator", VALID_SOURCE)
    config = RunConfig(num_episodes=2, max_steps_per_episode=15, seeds=[0, 1])

    run = run_manager.run_node(node, config)

    assert run.status == "completed"
    assert run.num_episodes == 2
    episodes = experience.list_episodes(run_id=run.id)
    assert len(episodes) == 2
    for episode in episodes:
        assert episode.actor_type == "node"
        assert episode.actor_id == str(node.id)
        transitions = experience.get_transitions(episode.id)
        assert len(transitions) == episode.num_steps
        for t in transitions:
            assert t.run_id == run.id
            assert t.actor_type == "node"


def test_run_policy_records_execution_errors_and_falls_back(policy_store, run_manager, experience):
    node = policy_store.create("broken", RAISES_SOURCE)
    config = RunConfig(num_episodes=1, max_steps_per_episode=3, seeds=[0])

    run = run_manager.run_node(node, config)

    assert run.status == "completed"  # the run itself doesn't crash
    errors = run_manager.list_errors(node_id=node.id, run_id=run.id)
    assert len(errors) == 3  # every step errors, falls back to a random action
    assert all(e.error_type == "IndexError" for e in errors)

    episodes = experience.list_episodes(run_id=run.id)
    transitions = experience.get_transitions(episodes[0].id)
    assert len(transitions) == 3  # the episode still produced transitions via fallback actions

    # Every fallback transition carries its own error in metadata and is
    # auto-tagged, so it can be found and used to fill the execution_error
    # evidence placeholder later -- no free-text re-entry needed.
    for t in transitions:
        error = t.metadata.get("execution_error")
        assert error is not None
        assert error["error_type"] == "IndexError"
        assert experience.get_tags(transition_id=t.id) == ["execution-error"]


def test_run_policy_terminate_on_error_ends_episode_at_first_error(policy_store, run_manager, experience):
    node = policy_store.create("broken", RAISES_SOURCE)
    config = RunConfig(num_episodes=1, max_steps_per_episode=15, seeds=[0])

    run = run_manager.run_node(node, config, on_action_error="terminate")

    assert run.status == "completed"
    # Only one error recorded (the first step) -- unlike the "random"-mode
    # test above, the episode ends there instead of retrying for the full
    # max_steps_per_episode budget.
    errors = run_manager.list_errors(node_id=node.id, run_id=run.id)
    assert len(errors) == 1
    episodes = experience.list_episodes(run_id=run.id)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.num_steps == 0  # the erroring action was never actually stepped
    assert episode.total_reward == 0.0
    assert episode.truncated is True
    assert episode.terminated is False
    assert run.total_reward == 0.0
    assert run.num_steps == 0


# -- episode-scoped policy memory --------------------------------------------

MEMORY_SOURCE = (
    "def policy(observation, memory):\n"
    "    memory['count'] = memory.get('count', 0) + 1\n"
    "    return memory['count'] % 4\n"
)
INVALID_MEMORY_SOURCE = (
    "def policy(observation, memory):\n"
    "    memory['x'] = range(3)\n"  # not one of the accepted memory value types
    "    return 0\n"
)


def test_run_policy_memory_persists_within_and_resets_across_episodes(policy_store, run_manager, experience):
    node = policy_store.create("counter", MEMORY_SOURCE)
    config = RunConfig(num_episodes=2, max_steps_per_episode=3, seeds=[0, 1])

    run = run_manager.run_node(node, config)

    assert run.status == "completed"
    episodes = sorted(experience.list_episodes(run_id=run.id), key=lambda e: e.episode_index)
    assert len(episodes) == 2
    for episode in episodes:
        transitions = sorted(experience.get_transitions(episode.id), key=lambda t: t.step_index)
        # memory going into each step reflects the count *before* that
        # step's own increment -- 0, 1, 2 -- and resets for the next episode
        # rather than continuing from the previous episode's final count.
        assert [t.memory for t in transitions] == [{}, {"count": 1}, {"count": 2}]


def test_run_policy_invalid_memory_falls_back_like_invalid_action(policy_store, run_manager, experience):
    node = policy_store.create("bad_memory", INVALID_MEMORY_SOURCE)
    config = RunConfig(num_episodes=1, max_steps_per_episode=3, seeds=[0])

    run = run_manager.run_node(node, config)

    assert run.status == "completed"
    errors = run_manager.list_errors(node_id=node.id, run_id=run.id)
    assert len(errors) == 3
    assert all(e.error_type == "InvalidMemory" for e in errors)

    transitions = experience.get_transitions(experience.list_episodes(run_id=run.id)[0].id)
    # memory never actually advances past {} -- every step's invalid
    # mutation is reverted before the next step sees it.
    assert all(t.memory == {} for t in transitions)


def test_run_policy_invalid_memory_terminates_when_configured(policy_store, run_manager, experience):
    node = policy_store.create("bad_memory", INVALID_MEMORY_SOURCE)
    config = RunConfig(num_episodes=1, max_steps_per_episode=15, seeds=[0])

    run = run_manager.run_node(node, config, on_action_error="terminate")

    errors = run_manager.list_errors(node_id=node.id, run_id=run.id)
    assert len(errors) == 1
    assert errors[0].error_type == "InvalidMemory"
    episode = experience.list_episodes(run_id=run.id)[0]
    assert episode.num_steps == 0  # ended at the first invalid-memory step


def test_run_policy_with_invalid_policy_fails_fast(policy_store, run_manager):
    node = policy_store.create("bad", "def not_policy(observation):\n    return 0\n")
    config = RunConfig(num_episodes=1)

    run = run_manager.run_node(node, config)

    assert run.status == "failed"
    errors = run_manager.list_errors(node_id=node.id, run_id=run.id)
    assert len(errors) == 1
    assert errors[0].error_type == "InvalidCode"

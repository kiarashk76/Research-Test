from __future__ import annotations

from core.interaction import InteractionSession


def test_episode_and_transition_persistence(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    assert session.episode.id is not None
    assert session.episode.num_steps == 0

    action = adapter.get_human_controls()[0].action
    transition, result = session.step(action)

    assert transition.id is not None
    assert transition.step_index == 0
    assert transition.actor_type == "human"

    episode = experience.get_episode(session.episode.id)
    assert episode.num_steps == 1
    assert episode.total_reward == transition.reward

    # both termination signals survive independently -- never collapsed to `done`
    assert transition.terminated in (True, False)
    assert transition.truncated in (True, False)
    assert transition.done == (transition.terminated or transition.truncated)


def test_episode_runs_to_completion_and_finalizes(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < 100:
        transition, result = session.step(adapter.sample_action())
        done = result.done
        steps += 1

    episode = experience.get_episode(session.episode.id)
    assert episode.ended_at is not None
    assert episode.num_steps == steps
    assert episode.terminated or episode.truncated


def test_read_state_and_render_roundtrip(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=1)
    transition, _ = session.step(adapter.sample_action())

    state = experience.read_state(transition, "state")
    next_state = experience.read_state(transition, "next_state")
    assert state.shape == next_state.shape

    render_text = experience.read_render(transition)
    assert render_text is None or isinstance(render_text, str)


def test_list_transitions_filters_by_actor_and_episode(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=2)
    session.step(adapter.sample_action())
    session.step(adapter.sample_action())

    all_for_actor = experience.list_transitions(actor_type="human")
    assert len(all_for_actor) == 2

    none_for_other_actor = experience.list_transitions(actor_type="policy")
    assert len(none_for_other_actor) == 0

    for_episode = experience.list_transitions(episode_id=session.episode.id)
    assert len(for_episode) == 2


def test_tags_and_annotations(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=3)
    transition, _ = session.step(adapter.sample_action())

    experience.add_tag("interesting", transition_id=transition.id)
    experience.add_tag("collision", transition_id=transition.id)
    experience.add_annotation("Agent bumped into the wall here.", transition_id=transition.id)

    tags = experience.get_tags(transition_id=transition.id)
    notes = experience.get_annotations(transition_id=transition.id)
    assert set(tags) == {"interesting", "collision"}
    assert notes == ["Agent bumped into the wall here."]

    filtered = experience.list_transitions(tag="interesting")
    assert [t.id for t in filtered] == [transition.id]


def test_get_episode_tags_includes_transition_and_episode_level_tags(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=4)
    transition, _ = session.step(adapter.sample_action())

    # A tag on a single step...
    experience.add_tag("collision", transition_id=transition.id)
    # ...and a tag on the episode itself...
    experience.add_tag("human-demo", episode_id=session.episode.id)

    tags = experience.get_episode_tags(session.episode.id)
    assert set(tags) == {"collision", "human-demo"}

    # An episode with no tags at all.
    other = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    other.reset(seed=5)
    other.step(adapter.sample_action())
    assert experience.get_episode_tags(other.episode.id) == []


def test_delete_episode_removes_transitions_tags_and_annotations(adapter, experience, evidence):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=6)
    session.step(adapter.sample_action())
    transitions = experience.get_transitions(session.episode.id)

    experience.add_tag("collision", transition_id=transitions[0].id)
    experience.add_tag("human-demo", episode_id=session.episode.id)
    experience.add_annotation("note", transition_id=transitions[0].id)
    experience.add_annotation("episode note", episode_id=session.episode.id)

    selection = evidence.get_or_create_active()
    evidence.add_transition(selection, session.episode.id, transitions[0].id)

    episode_id = session.episode.id
    experience.delete_episode(episode_id)

    assert experience.get_episode(episode_id) is None
    assert experience.get_transitions(episode_id) == []
    assert experience.db.query(
        "SELECT * FROM transition_tags WHERE episode_id = ? OR transition_id = ?",
        (episode_id, transitions[0].id)) == []
    assert experience.db.query(
        "SELECT * FROM transition_annotations WHERE episode_id = ? OR transition_id = ?",
        (episode_id, transitions[0].id)) == []
    assert evidence.count(selection, experience) == 0


def test_delete_episode_does_not_affect_other_episodes(adapter, experience):
    session_a = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session_a.reset(seed=7)
    session_a.step(adapter.sample_action())

    session_b = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session_b.reset(seed=8)
    session_b.step(adapter.sample_action())

    experience.delete_episode(session_a.episode.id)

    assert experience.get_episode(session_a.episode.id) is None
    assert experience.get_episode(session_b.episode.id) is not None
    assert len(experience.get_transitions(session_b.episode.id)) == 1


def test_step_actor_override_records_per_transition_actor(adapter, experience):
    """A single InteractionSession can switch actors between steps (e.g.
    Play switching controllers mid-episode) and each transition records
    exactly who acted, overriding the session's default."""
    session = InteractionSession(adapter, experience, actor_type="policy", actor_id="1")
    session.reset(seed=0)

    t1, _ = session.step(adapter.sample_action())  # uses the session default
    assert t1.actor_type == "policy" and t1.actor_id == "1"

    t2, _ = session.step(adapter.sample_action(), actor_type="policy", actor_id="2")
    assert t2.actor_type == "policy" and t2.actor_id == "2"

    t3, _ = session.step(adapter.sample_action(), actor_type="human", actor_id="human")
    assert t3.actor_type == "human" and t3.actor_id == "human"

    # all three transitions belong to the same episode
    assert t1.episode_id == t2.episode_id == t3.episode_id == session.episode.id


def test_step_actor_override_marks_episode_mixed(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="policy", actor_id="1")
    session.reset(seed=0)
    session.step(adapter.sample_action())  # matches the started actor -- not mixed yet

    episode = experience.get_episode(session.episode.id)
    assert episode.actor_type == "policy"

    session.step(adapter.sample_action(), actor_type="policy", actor_id="2")  # different actor now

    episode = experience.get_episode(session.episode.id)
    assert episode.actor_type == "mixed"
    assert episode.actor_id is None


def test_step_actor_override_does_not_mark_mixed_when_actor_unchanged(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    session.step(adapter.sample_action(), actor_type="human", actor_id="human")
    session.step(adapter.sample_action())  # default, same as explicit above

    episode = experience.get_episode(session.episode.id)
    assert episode.actor_type == "human"  # never flipped to "mixed"


# -- episode-scoped policy memory --------------------------------------------

def test_session_memory_starts_empty_and_resets_on_reset(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="policy", actor_id="1")
    session.reset(seed=0)
    assert session.memory == {}

    session.memory = {"count": 3}
    session.reset(seed=1)  # a fresh episode
    assert session.memory == {}


def test_step_persists_the_memory_snapshot_passed_in(adapter, experience):
    """step()'s `memory` argument is stored as-is on the Transition -- it's
    the caller's job to pass the *pre-call* snapshot (see core.runs.
    RunManager.run_node), not whatever session.memory happens to be by the
    time step() runs."""
    session = InteractionSession(adapter, experience, actor_type="policy", actor_id="1")
    session.reset(seed=0)

    transition, _ = session.step(adapter.sample_action(), memory={"visited": True, "count": 1})
    assert transition.memory == {"visited": True, "count": 1}

    reloaded = experience.get_transitions(session.episode.id)[0]
    assert reloaded.memory == {"visited": True, "count": 1}


def test_step_without_memory_argument_persists_empty_dict(adapter, experience):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=0)
    transition, _ = session.step(adapter.sample_action())
    assert transition.memory == {}


def test_session_memory_survives_a_mid_episode_actor_switch(adapter, experience):
    """Memory belongs to the *episode* (the InteractionSession), not to
    whichever policy is currently driving it -- Play's controller switch
    reuses the same session without resetting the episode, so memory must
    carry over across it untouched."""
    session = InteractionSession(adapter, experience, actor_type="policy", actor_id="1")
    session.reset(seed=0)
    session.memory = {"from_node_a": 1}

    # Switch to a different controller mid-episode (no reset()) -- same
    # session, same memory.
    session.step(adapter.sample_action(), actor_type="policy", actor_id="2")
    assert session.memory == {"from_node_a": 1}

    session.step(adapter.sample_action(), actor_type="human", actor_id="human")
    assert session.memory == {"from_node_a": 1}

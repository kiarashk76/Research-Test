from __future__ import annotations

import pytest

from core.evidence import basket_label
from core.interaction import InteractionSession


def _play_episode(adapter, experience, num_steps=5, seed=0):
    session = InteractionSession(adapter, experience, actor_type="human", actor_id="human")
    session.reset(seed=seed)
    for _ in range(num_steps):
        _, result = session.step(adapter.sample_action())
        if result.done:
            break
    return session.episode


def test_add_transition_and_resolve(adapter, experience, evidence):
    episode = _play_episode(adapter, experience)
    transitions = experience.get_transitions(episode.id)

    selection = evidence.get_or_create_active()
    evidence.add_transition(selection, episode.id, transitions[0].id, "first step")
    evidence.add_transition(selection, episode.id, transitions[1].id, "second step")

    resolved = evidence.resolve_transitions(selection, experience)
    assert [t.id for t in resolved] == [transitions[0].id, transitions[1].id]
    assert evidence.count(selection, experience) == 2


def test_add_range_resolves_inclusive_bounds(adapter, experience, evidence):
    episode = _play_episode(adapter, experience, num_steps=6)
    transitions = experience.get_transitions(episode.id)
    assert len(transitions) >= 4

    selection = evidence.get_or_create_active()
    evidence.add_range(selection, episode.id, transitions[1].step_index, transitions[3].step_index)

    resolved = evidence.resolve_transitions(selection, experience)
    assert [t.id for t in resolved] == [t.id for t in transitions[1:4]]


def test_add_whole_episode(adapter, experience, evidence):
    episode = _play_episode(adapter, experience, num_steps=4)
    transitions = experience.get_transitions(episode.id)

    selection = evidence.get_or_create_active()
    evidence.add_episode(selection, episode.id)

    resolved = evidence.resolve_transitions(selection, experience)
    assert len(resolved) == len(transitions)


def test_duplicate_transitions_are_deduplicated(adapter, experience, evidence):
    episode = _play_episode(adapter, experience, num_steps=3)
    transitions = experience.get_transitions(episode.id)

    selection = evidence.get_or_create_active()
    evidence.add_transition(selection, episode.id, transitions[0].id)
    evidence.add_episode(selection, episode.id)  # includes transitions[0] again

    resolved = evidence.resolve_transitions(selection, experience)
    assert len(resolved) == len(transitions)  # not len(transitions) + 1


def test_remove_and_clear(adapter, experience, evidence):
    episode = _play_episode(adapter, experience, num_steps=3)
    transitions = experience.get_transitions(episode.id)

    selection = evidence.get_or_create_active()
    item = evidence.add_transition(selection, episode.id, transitions[0].id)
    evidence.add_transition(selection, episode.id, transitions[1].id)

    evidence.remove_item(item.id)
    assert evidence.count(selection, experience) == 1

    evidence.clear(selection)
    assert evidence.count(selection, experience) == 0


def test_list_all_includes_default_and_named_baskets(evidence):
    default = evidence.get_or_create_active()
    named_a = evidence.create("failures only")
    named_b = evidence.create("human demos")

    all_baskets = evidence.list_all()
    assert [b.id for b in all_baskets] == [default.id, named_a.id, named_b.id]
    assert basket_label(default) == "Default"
    assert basket_label(named_a) == "failures only"


def test_list_selections_excludes_default_basket(evidence):
    evidence.get_or_create_active()
    named = evidence.create("my basket")

    named_only = evidence.list_selections()
    assert [b.id for b in named_only] == [named.id]


def test_rename_named_basket(evidence):
    named = evidence.create("draft name")
    renamed = evidence.rename(named, "final name")
    assert renamed.name == "final name"
    assert evidence.get(named.id).name == "final name"


def test_rename_default_basket_raises(evidence):
    default = evidence.get_or_create_active()
    with pytest.raises(ValueError):
        evidence.rename(default, "not allowed")


def test_delete_named_basket_removes_its_items(adapter, experience, evidence):
    episode = _play_episode(adapter, experience, num_steps=2)
    transitions = experience.get_transitions(episode.id)

    named = evidence.create("temp basket")
    evidence.add_transition(named, episode.id, transitions[0].id)
    evidence.delete(named)

    assert evidence.get(named.id) is None
    assert named.id not in [b.id for b in evidence.list_all()]


def test_delete_default_basket_raises(evidence):
    default = evidence.get_or_create_active()
    with pytest.raises(ValueError):
        evidence.delete(default)

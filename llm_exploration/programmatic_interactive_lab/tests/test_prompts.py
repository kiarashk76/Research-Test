from __future__ import annotations

from core.formatters import TransitionFormatter
from core.interaction import InteractionSession
from core.nodes import NodeStore
from core.prompts import (
    ACTION_SPACE_KEY, BUILTIN_TEMPLATES, ENV_DESCRIPTION_KEY, OBSERVATION_SPACE_KEY, PromptRenderer,
    PromptTemplateStore, build_render_values, default_action_space_description,
    default_environment_description, default_observation_space_description, ensure_builtin_templates,
    node_field_names, node_placeholder_values, resolve_environment_context,
)
from storage.artifacts import ArtifactStore


def test_template_versioning_never_overwrites(db):
    store = PromptTemplateStore(db)
    v1 = store.create("demo", "system v1", "user v1")
    v2 = store.new_version(v1, system_template="system v2")

    assert v2.version == v1.version + 1
    assert v2.parent_version_id == v1.id
    assert v2.user_template == v1.user_template  # untouched field carried forward

    reloaded_v1 = store.get(v1.id)
    assert reloaded_v1.system_template == "system v1"  # original untouched

    history = store.history("demo")
    assert [t.version for t in history] == [1, 2]
    assert store.latest_by_name("demo").id == v2.id


def test_parses_as_code_defaults_false_and_persists(db):
    store = PromptTemplateStore(db)
    plain = store.create("plain", "sys", "usr")
    assert plain.parses_as_code is False

    code_producing = store.create("code-producer", "sys", "usr", parses_as_code=True)
    assert code_producing.parses_as_code is True
    assert store.get(code_producing.id).parses_as_code is True


def test_parses_as_code_carries_forward_on_new_version_unless_overridden(db):
    store = PromptTemplateStore(db)
    v1 = store.create("demo2", "sys", "usr", parses_as_code=True)

    v2 = store.new_version(v1, system_template="sys v2")
    assert v2.parses_as_code is True  # carried forward, not reset to the dataclass default

    v3 = store.new_version(v2, parses_as_code=False)
    assert v3.parses_as_code is False


def test_builtin_templates_flag_exactly_the_code_producing_ones():
    code_producing = {name for name, _, _, parses_as_code in BUILTIN_TEMPLATES if parses_as_code}
    assert code_producing == {
        "Direct Policy Update", "Update Policy From Critique", "Repair Policy From Code Diagnosis",
        "Direct Policy Update (Functional)", "Update Policy Functions From Critique",
        "Repair Policy Functions From Diagnosis",
    }


def test_delete_template_removes_all_versions(db):
    store = PromptTemplateStore(db)
    v1 = store.create("throwaway", "sys", "usr")
    store.new_version(v1, system_template="sys v2")

    store.delete("throwaway")

    assert store.latest_by_name("throwaway") is None
    assert store.history("throwaway") == []
    assert "throwaway" not in store.list_names()


def test_delete_one_template_does_not_affect_others(db):
    store = PromptTemplateStore(db)
    store.create("keep-me", "sys", "usr")
    store.create("delete-me", "sys", "usr")

    store.delete("delete-me")

    assert store.latest_by_name("keep-me") is not None
    assert store.latest_by_name("delete-me") is None


def test_renderer_substitutes_values_from_a_plain_dict():
    renderer = PromptRenderer()
    values = {"transitions": "T1\nT2", "notes": "watch the wall"}
    text = renderer.render("Evidence:\n{{transitions}}\nNotes: {{notes}}", values)
    assert text == "Evidence:\nT1\nT2\nNotes: watch the wall"


def test_renderer_leaves_unknown_placeholder_untouched():
    renderer = PromptRenderer()
    text = renderer.render("{{transitions}} {{not_a_real_placeholder}}", {"transitions": "X"})
    assert text == "X {{not_a_real_placeholder}}"


def test_used_placeholders_extraction():
    used = PromptRenderer.used_placeholders("{{transitions}} and {{parent.code}} twice {{transitions}}")
    assert used == ["parent.code", "transitions"]


# -- Node-attribute placeholder vocabulary ------------------------------------

def test_node_placeholder_values_for_none_renders_every_field_as_unset():
    values = node_placeholder_values(None)
    assert values["code"] == "(no code)"
    assert values["hypothesis"] == "(no hypothesis)"
    assert values["critique"] == "(no critique)"
    assert values["code_diagnosis"] == "(no code diagnosis)"
    assert values["avg_reward"] == "(not yet evaluated)"
    # transitions is not a Node field -- absent unless explicitly passed.
    assert "transitions" not in values


def test_node_placeholder_values_reflects_real_node_content(db, tmp_path):
    nodes = NodeStore(db, ArtifactStore(tmp_path, "s1"), "s1")
    node = nodes.create("n", code="def policy(observation, memory):\n    return 0\n", tag="t")
    values = node_placeholder_values(node, transitions_text="T1")
    assert values["code"] == "def policy(observation, memory):\n    return 0\n"
    assert values["validation_status"] == "valid"
    assert values["transitions"] == "T1"


def test_node_field_names_excludes_structural_fields():
    names = node_field_names()
    for excluded in ("id", "session_id", "parent_id", "metadata", "evidence_selection_id"):
        assert excluded not in names
    assert "code" in names
    assert "avg_reward" in names


def test_build_render_values_fills_transitions_and_parent_fields():
    values = build_render_values(
        node_fields=node_placeholder_values(None),
        transitions_text="transition 0 stuff",
        notes="focus on walls",
        environment_description="ENV", observation_space="OBS", action_space="ACT",
    )
    assert values["transitions"] == "transition 0 stuff"
    assert values["notes"] == "focus on walls"
    assert values["environment_description"] == "ENV"
    # No parent given -- {{parent.X}} still renders sensibly instead of being left untouched.
    assert values["parent.code"] == "(no code)"


def test_build_render_values_with_no_transitions_or_notes():
    values = build_render_values(node_fields=node_placeholder_values(None))
    assert values["transitions"] == "(none provided)"
    assert values["notes"] == "(none)"


def test_build_render_values_fills_parent_fields_when_given(db, tmp_path):
    nodes = NodeStore(db, ArtifactStore(tmp_path, "s1"), "s1")
    parent = nodes.create("parent", code="def policy(observation, memory):\n    return 1\n")
    values = build_render_values(node_fields=node_placeholder_values(None), parent=parent,
                                  parent_transitions_text="parent's evidence")
    assert values["parent.code"] == "def policy(observation, memory):\n    return 1\n"
    assert values["parent.transitions"] == "parent's evidence"


def test_resolve_environment_context_values_thread_through_render_values(adapter):
    description, observation_space, action_space = resolve_environment_context(adapter, {})
    values = build_render_values(
        node_fields=node_placeholder_values(None), environment_description=description,
        observation_space=observation_space, action_space=action_space,
    )
    assert values["environment_description"] == default_environment_description(adapter)
    assert values["observation_space"] == default_observation_space_description(adapter)
    # The raw Gym space repr is deliberately NOT shown to the LLM -- every
    # environment's own hint states shape/value-range in plain English
    # instead (see environments/simple_grid_env.py's instance-level hint).
    assert "Box(" not in values["observation_space"]
    assert "5x5" in values["observation_space"]  # this fixture's actual configured size
    assert "Grid cell codes" not in values["observation_space"]
    assert adapter.action_space_description() in values["action_space"]


def test_ensure_builtin_templates_seeds_every_builtin_globally(db):
    store = PromptTemplateStore(db)
    ensure_builtin_templates(store)

    names = store.list_names()
    for name, _, _, _ in BUILTIN_TEMPLATES:
        assert name in names
        template = store.latest_by_name(name)
        assert template.session_id is None  # global -- available in every session
        assert template.version == 1


def test_every_builtin_system_prompt_includes_environment_context():
    """Every template's system prompt must ground the LLM in the
    environment's interface (grid/actions), regardless of that template's
    narrow purpose -- not just the ones that also use it in the user prompt."""
    for name, system_template, _, _ in BUILTIN_TEMPLATES:
        assert "{{environment_description}}" in system_template, name
        assert "{{observation_space}}" in system_template, name
        assert "{{action_space}}" in system_template, name


def test_ensure_builtin_templates_is_idempotent(db):
    store = PromptTemplateStore(db)
    ensure_builtin_templates(store)
    ensure_builtin_templates(store)

    for name, _, _, _ in BUILTIN_TEMPLATES:
        assert len(store.history(name)) == 1  # not re-seeded/re-versioned


def test_ensure_builtin_templates_does_not_clobber_an_edited_builtin(db):
    store = PromptTemplateStore(db)
    ensure_builtin_templates(store)

    original = store.latest_by_name("Direct Policy Update")
    edited = store.new_version(original, system_template="my own rewritten system prompt")

    ensure_builtin_templates(store)  # re-running must not reset the edit

    latest = store.latest_by_name("Direct Policy Update")
    assert latest.id == edited.id
    assert latest.system_template == "my own rewritten system prompt"


def test_ensure_builtin_templates_removes_a_legacy_template_left_from_an_older_version(db):
    store = PromptTemplateStore(db)
    # Simulate a database that still has a pre-upgrade built-in template,
    # including a researcher-edited version of it.
    legacy = store.create("Improve Policy Using Processed Evidence", "old sys", "old usr")
    store.new_version(legacy, system_template="an edited legacy version")

    ensure_builtin_templates(store)

    assert store.latest_by_name("Improve Policy Using Processed Evidence") is None
    assert store.history("Improve Policy Using Processed Evidence") == []
    assert store.latest_by_name("Direct Policy Update") is not None


def test_resolve_environment_context_defaults_from_adapter_when_no_overrides(adapter):
    description, observation_space, action_space = resolve_environment_context(adapter, {})
    assert description == default_environment_description(adapter)
    assert observation_space == default_observation_space_description(adapter)
    assert action_space == default_action_space_description(adapter)


def test_resolve_environment_context_uses_session_metadata_overrides(adapter):
    metadata = {
        ENV_DESCRIPTION_KEY: "A mystery grid world.",
        OBSERVATION_SPACE_KEY: "",
        ACTION_SPACE_KEY: "Four unlabeled discrete actions.",
    }
    description, observation_space, action_space = resolve_environment_context(adapter, metadata)
    assert description == "A mystery grid world."
    assert observation_space == ""
    assert action_space == "Four unlabeled discrete actions."

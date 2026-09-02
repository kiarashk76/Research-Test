from __future__ import annotations

import json

import core.llm_models as llm_models
from core.llm import LLMService, build_llm_client_from_model_config


def test_list_llm_models_empty_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_models, "MODELS_FILE", tmp_path / "does-not-exist.json")
    assert llm_models.list_llm_models() == []
    assert llm_models.get_llm_model("anything") is None


def test_list_and_get_llm_models_from_file(tmp_path, monkeypatch):
    models_file = tmp_path / "llm_models.json"
    models_file.write_text(json.dumps([
        {"name": "gemini-flash", "model_name": "gemini-2.5-flash",
         "url": "https://example.com/v1", "api_key": "k1"},
        {"name": "local-vllm", "model_name": "Qwen2.5-7B",
         "url": "http://localhost:8000/v1", "api_key": "k2", "temperature": 0.2},
    ]))
    monkeypatch.setattr(llm_models, "MODELS_FILE", models_file)

    entries = llm_models.list_llm_models()
    assert [e["name"] for e in entries] == ["gemini-flash", "local-vllm"]

    found = llm_models.get_llm_model("local-vllm")
    assert found["model_name"] == "Qwen2.5-7B"
    assert found["temperature"] == 0.2

    assert llm_models.get_llm_model("nonexistent") is None


def test_build_llm_client_from_model_config_uses_entry_fields_directly():
    config = {"name": "custom", "model_name": "some-model", "url": "http://host/v1",
              "api_key": "secret", "temperature": 0.3}
    client = build_llm_client_from_model_config(config)
    assert client.model == "some-model"
    assert client.temperature == 0.3
    # Defaults fill in fields not present in the entry.
    assert client.stream is True


def test_llm_service_uses_model_config_when_given(db):
    config = {"name": "custom-picker-name", "model_name": "some-model",
              "url": "http://host/v1", "api_key": "secret"}
    service = LLMService(db, model_config=config)
    assert service.llm_name == "custom-picker-name"
    assert service.client.model == "some-model"


def test_llm_service_falls_back_to_legacy_preset_without_model_config(db, monkeypatch):
    monkeypatch.setenv("FAKE_MODEL", "fake-model-id")
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    monkeypatch.setenv("FAKE_BASE_URL", "http://localhost:1")
    import core.llm as llm_module
    monkeypatch.setitem(llm_module.LLM_PRESETS, "FAKE",
                         {"temperature": 0.7, "timeout": 60, "max_retries": 3, "stream": True})

    service = LLMService(db, llm_name="FAKE")
    assert service.llm_name == "FAKE"
    assert service.client.model == "fake-model-id"

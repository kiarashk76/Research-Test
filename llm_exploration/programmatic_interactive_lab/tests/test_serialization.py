from __future__ import annotations

import numpy as np

from storage.serialization import (
    deserialize_state,
    serialize_state,
    summarize_for_display,
    summarize_memory,
)


def test_roundtrip_numpy_array():
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    blob = serialize_state(arr)
    restored = deserialize_state(blob)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == arr.dtype
    assert np.array_equal(restored, arr)


def test_roundtrip_plain_python_values():
    for value in (3, 3.5, "text", True, None, [1, 2, 3], {"a": 1, "b": [1, 2]}):
        assert deserialize_state(serialize_state(value)) == value


def test_roundtrip_nested_dict_with_array():
    payload = {"grid": np.zeros((2, 2), dtype=np.int64), "step": 3}
    restored = deserialize_state(serialize_state(payload))
    assert restored["step"] == 3
    assert np.array_equal(restored["grid"], payload["grid"])


def test_summarize_for_display_small_values_shown_in_full():
    assert summarize_for_display(3) == "3"
    assert summarize_for_display(True) == "True"
    assert summarize_for_display(None) == "None"
    assert summarize_for_display("short") == "'short'"
    assert summarize_for_display([1, 2, 3]) == "list([1, 2, 3])"


def test_summarize_for_display_truncates_long_string_but_states_length():
    text = "x" * 1000
    summary = summarize_for_display(text, max_len=50)
    assert len(summary) <= 80
    assert "len=1000" in summary


def test_summarize_for_display_truncates_large_array_but_states_shape():
    arr = np.arange(1000)
    summary = summarize_for_display(arr, max_len=100)
    assert "shape=(1000,)" in summary
    assert "..." in summary


def test_summarize_for_display_truncates_large_list_but_states_length():
    big = list(range(1000))
    summary = summarize_for_display(big, max_len=100)
    assert "len=1000" in summary
    assert "..." in summary


def test_summarize_memory_shows_every_key_but_shortens_long_values():
    memory = {"visited": True, "count": 2, "history": list(range(1000))}
    summary = summarize_memory(memory, max_value_len=120)
    assert "'visited': True" in summary
    assert "'count': 2" in summary
    assert "'history':" in summary
    assert "len=1000" in summary


def test_summarize_memory_empty():
    assert summarize_memory({}) == "{}"

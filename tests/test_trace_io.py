import json

import pytest

from src.traces.io import (
    read_trace_records,
    repair_trailing_partial_record,
    validate_trace_coverage,
)


def test_trace_records_are_sorted_and_validated(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    records = [
        {"index": 1, "question": "q2", "response": "#### 2"},
        {"index": 0, "question": "q1", "response": "#### 1"},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    loaded = read_trace_records(path)
    responses, metrics = validate_trace_coverage(
        [
            {"question": "q1", "answer": "#### 1"},
            {"question": "q2", "answer": "#### 2"},
        ],
        loaded,
    )
    assert responses == {"q1": "#### 1", "q2": "#### 2"}
    assert metrics["accuracy"] == 1.0


def test_incomplete_trace_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing train index 0"):
        validate_trace_coverage([{"question": "q", "answer": "#### 1"}], [])


def test_trailing_partial_trace_is_repaired(tmp_path) -> None:
    path = tmp_path / "partial.jsonl"
    path.write_bytes(b'{"index": 0, "question": "q", "response": "#### 1"}\n{"index"')
    assert repair_trailing_partial_record(path)
    assert len(read_trace_records(path)) == 1

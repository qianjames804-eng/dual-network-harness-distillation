from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.eval.answers import extract_final_answer


def repair_trailing_partial_record(path: str | Path) -> bool:
    """Remove only an unterminated, invalid final JSONL fragment after a crash."""
    target = Path(path)
    if not target.exists():
        return False
    payload = target.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return False
    boundary = payload.rfind(b"\n") + 1
    tail = payload[boundary:]
    try:
        json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        with target.open("r+b") as handle:
            handle.truncate(boundary)
        return True
    with target.open("ab") as handle:
        handle.write(b"\n")
    return True


def read_trace_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {target} at line {line_number}"
                ) from error
            index = int(record.get("index", -1))
            if index < 0 or index in seen_indexes:
                raise ValueError(
                    f"Invalid or duplicate trace index {index} in {target}"
                )
            if not record.get("question") or not record.get("response"):
                raise ValueError(
                    f"Trace line {line_number} lacks question/response in {target}"
                )
            seen_indexes.add(index)
            records.append(record)
    return sorted(records, key=lambda item: int(item["index"]))


def validate_trace_coverage(
    rows: Iterable[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, Any]]:
    expected_rows = list(rows)
    by_index = {int(record["index"]): record for record in records}
    responses: dict[str, str] = {}
    correct = 0
    for index, row in enumerate(expected_rows):
        record = by_index.get(index)
        if record is None:
            raise ValueError(f"Teacher trace is incomplete: missing train index {index}")
        if record["question"] != row["question"]:
            raise ValueError(
                f"Teacher trace/data mismatch at train index {index}; "
                "check dataset revision and seed"
            )
        response = str(record["response"]).strip()
        if row["question"] in responses:
            raise ValueError(f"Duplicate training question at index {index}")
        responses[row["question"]] = response
        predicted = extract_final_answer(response)
        expected = extract_final_answer(row["answer"])
        correct += int(predicted == expected and expected is not None)
    total = len(expected_rows)
    if len(records) != total:
        raise ValueError(
            f"Teacher trace has {len(records)} records but dataset has {total} examples"
        )
    return responses, {
        "examples": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
    }

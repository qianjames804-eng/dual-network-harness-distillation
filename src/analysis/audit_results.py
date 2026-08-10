from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import RESULT_FIELDS


def audit(results_path: Path, run_id: str | None = None) -> dict:
    frame = pd.read_csv(results_path)
    missing_columns = [column for column in RESULT_FIELDS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing result columns: {missing_columns}")
    if run_id:
        frame = frame[frame["run_id"] == run_id]
    else:
        frame = frame[frame["status"] == "smoke_success"]
    if frame.empty:
        raise ValueError("No matching result rows")

    methods = set(frame["method"])
    required = {"B0-Base", "B1-Clean-KD"}
    if not required.issubset(methods):
        raise ValueError(f"Missing B0/B1 rows: {sorted(required - methods)}")
    if frame["seed"].isna().any():
        raise ValueError("At least one result row is missing a random seed")

    b0 = frame[frame["method"] == "B0-Base"].iloc[-1]
    b1 = frame[frame["method"] == "B1-Clean-KD"].iloc[-1]
    if int(b0["train_tokens"]) != 0 or int(b1["train_tokens"]) <= 0:
        raise ValueError("Train-token accounting is inconsistent for B0/B1")
    if b0["student_model"] != b1["student_model"]:
        raise ValueError("B0 and B1 use different base Student checkpoints")
    if "pipeline_validation_only" not in str(b1["notes"]):
        raise ValueError("Resource-adapted result is missing its claim-scope label")

    return {
        "status": "pass",
        "run_id": str(b1["run_id"]),
        "rows": int(len(frame)),
        "seed": int(b1["seed"]),
        "fixed_raw_pool": int(b1["raw_examples"]) == int(b1["retained_examples"]),
        "schema_columns": len(RESULT_FIELDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.results, args.run_id)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

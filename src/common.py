from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


RESULT_FIELDS = [
    "run_id", "method", "defense", "defense_strength", "teacher_model",
    "student_model", "dataset", "seed", "raw_examples",
    "retained_examples", "train_tokens", "lora_r", "lr", "steps",
    "teacher_accuracy", "student_base_accuracy", "student_accuracy",
    "distill_gain", "recovery", "nn1_spearman", "nn1_auroc",
    "nn2_auroc", "nn2_f1", "nn2_brier", "nn2_ece", "nn2_spearman",
    "gpu_hours", "wall_time", "status", "notes",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def utc_run_id(profile: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{profile}-s{seed}-{stamp}"


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_result(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != RESULT_FIELDS:
            raise ValueError("results.csv schema does not match the execution plan")
    else:
        with target.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_FIELDS).writeheader()

    normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
    with target.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_FIELDS).writerow(normalized)

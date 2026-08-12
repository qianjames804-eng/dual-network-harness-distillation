"""MATH / MATH-500 normalization with an explicit no-overlap guard."""
from __future__ import annotations
from typing import Any
from datasets import Dataset, load_dataset

def _question(row: dict) -> str:
    return str(row.get("problem") or row.get("question") or row.get("input") or "")
def _answer(row: dict) -> str:
    return str(row.get("solution") or row.get("answer") or row.get("output") or "")
def load_math_subsets(config: dict[str, Any], seed: int) -> tuple[Dataset, Dataset]:
    train = load_dataset(config["train"], split=config.get("train_split", "train"), cache_dir=config.get("cache_dir")).shuffle(seed=seed)
    test = load_dataset(config["test"], split=config.get("test_split", "test"), cache_dir=config.get("cache_dir")).shuffle(seed=seed)
    train = train.select(range(min(int(config["train_examples"]), len(train))))
    test = test.select(range(min(int(config["eval_examples"]), len(test))))
    overlap = {_question(row) for row in train}.intersection(_question(row) for row in test)
    if overlap: raise RuntimeError(f"MATH train/test leakage: {len(overlap)} duplicated questions")
    return train, test

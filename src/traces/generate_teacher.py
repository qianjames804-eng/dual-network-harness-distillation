from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock, Timeout
from tqdm import tqdm

from src.common import append_jsonl, load_yaml, set_seed
from src.data.gsm8k import load_subsets, prompt_text
from src.eval.answers import extract_final_answer
from src.modeling import load_causal_lm
from src.traces.io import (
    read_trace_records,
    repair_trailing_partial_record,
    validate_trace_coverage,
)
from src.validate_config import validate


def trace_path(config: dict[str, Any]) -> Path:
    configured = config["teacher"].get("trace_path")
    if not configured:
        raise ValueError("teacher.trace_path is required for generated traces")
    return Path(configured)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic, resumable clean Teacher traces"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--max-new-records",
        type=int,
        default=None,
        help="Optional chunk limit; rerun the same command to resume",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate(config)
    if config["teacher"].get("trace_source") != "generated":
        raise ValueError("This command only supports teacher.trace_source=generated")

    seed = int(config["seed"])
    set_seed(seed)
    train_rows, _ = load_subsets(config["dataset"], seed)
    target = trace_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(target) + ".lock")
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise RuntimeError(f"Another trace generator holds {lock.lock_file}") from error

    try:
        repaired = repair_trailing_partial_record(target)
        if repaired:
            print(f"repaired trailing partial trace record: {target}")
        existing = read_trace_records(target)
        existing_by_index = {int(record["index"]): record for record in existing}
        for index, record in existing_by_index.items():
            if (
                index >= len(train_rows)
                or record["question"] != train_rows[index]["question"]
            ):
                raise ValueError(
                    f"Existing trace/data mismatch at index {index}; do not mix "
                    "dataset revisions or seeds in one trace file"
                )
        if len(existing_by_index) == len(train_rows):
            _, metrics = validate_trace_coverage(train_rows, existing)
            print(json.dumps({"status": "complete", **metrics}, indent=2))
            return

        model, tokenizer = load_causal_lm(
            config["teacher"], config["dataset"]["cache_dir"]
        )
        model.eval()
        generation = config["teacher"].get("generation", {})
        generation_config = deepcopy(model.generation_config)
        generation_config.do_sample = bool(generation.get("do_sample", False))
        if not generation_config.do_sample:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        added = 0
        progress = tqdm(
            total=len(train_rows),
            initial=len(existing_by_index),
            desc="Teacher traces",
        )
        for index, row in enumerate(train_rows):
            if index in existing_by_index:
                continue
            prompt = prompt_text(tokenizer, row["question"])
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(generation.get("max_prompt_length", 1024)),
            ).to("cuda")
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    generation_config=generation_config,
                    max_new_tokens=int(generation.get("max_new_tokens", 512)),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response_ids = output[0, encoded["input_ids"].shape[1] :]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            if not response:
                raise RuntimeError(f"Teacher returned an empty response at index {index}")
            predicted = extract_final_answer(response)
            expected = extract_final_answer(row["answer"])
            append_jsonl(
                target,
                {
                    "index": index,
                    "question": row["question"],
                    "response": response,
                    "predicted": predicted,
                    "expected": expected,
                    "correct": predicted == expected and expected is not None,
                    "response_tokens": int(response_ids.numel()),
                    "teacher_model": config["teacher"]["id"],
                    "teacher_revision": config["teacher"].get("revision"),
                    "seed": seed,
                },
            )
            existing_by_index[index] = {"index": index}
            added += 1
            progress.update(1)
            if args.max_new_records is not None and added >= args.max_new_records:
                break
        progress.close()
        records = read_trace_records(target)
        if len(records) == len(train_rows):
            _, metrics = validate_trace_coverage(train_rows, records)
            status = "complete"
        else:
            metrics = {"examples": len(records), "expected_examples": len(train_rows)}
            status = "partial"
        print(json.dumps({"status": status, "new_records": added, **metrics}, indent=2))
    finally:
        lock.release()


if __name__ == "__main__":
    main()

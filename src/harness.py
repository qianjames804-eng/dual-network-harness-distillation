from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from filelock import FileLock, Timeout

from src.common import append_jsonl, append_result, load_yaml, set_seed, utc_run_id
from src.data.gsm8k import GSM8KSFTDataset, load_subsets
from src.eval.gsm8k_eval import evaluate
from src.lora_train.train import train_lora
from src.modeling import load_causal_lm
from src.traces.io import read_trace_records, validate_trace_coverage
from src.validate_config import validate


def load_student(config: dict, cache_dir: str):
    return load_causal_lm(config, cache_dir)


def result_row(
    *, run_id: str, method: str, config: dict, base_accuracy: float,
    accuracy: float, train_tokens: int, steps: int, wall_seconds: float,
    teacher_accuracy: float, retained_examples: int, status: str, notes: str,
) -> dict:
    return {
        "run_id": run_id,
        "method": method,
        "defense": "clean",
        "defense_strength": "none",
        "teacher_model": config["teacher"]["id"],
        "student_model": config["student"]["id"],
        "dataset": config["dataset"]["name"],
        "seed": config["seed"],
        "raw_examples": config["dataset"]["train_examples"],
        "retained_examples": retained_examples,
        "train_tokens": train_tokens if method != "B0-Base" else 0,
        "lora_r": config["training"]["lora_r"] if method != "B0-Base" else "",
        "lr": config["training"]["learning_rate"] if method != "B0-Base" else "",
        "steps": steps if method != "B0-Base" else 0,
        "teacher_accuracy": teacher_accuracy,
        "student_base_accuracy": base_accuracy,
        "student_accuracy": accuracy,
        "distill_gain": accuracy - base_accuracy,
        "recovery": "",
        "gpu_hours": wall_seconds / 3600,
        "wall_time": wall_seconds,
        "status": status,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate(config)
    supported_profiles = {"resource_adapted_smoke", "paper_faithful_mvp"}
    if config.get("profile") not in supported_profiles:
        raise RuntimeError(
            "This B0/B1 harness supports only resource_adapted_smoke and "
            "paper_faithful_mvp. Full/NN/defense stages remain gated."
        )

    lock_path = Path(config["outputs"]["root"]) / ".harness.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise RuntimeError("Another Harness experiment is already running") from error

    try:
        seed = int(config["seed"])
        set_seed(seed)
        run_id = utc_run_id(config["profile"], seed)
        root = Path(config["outputs"]["root"]) / run_id
        metrics_dir = root / "metrics"
        checkpoint_dir = root / "checkpoints" / "b1_clean_kd"
        log_path = root / "run.jsonl"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(log_path, {"event": "run_started", "config": config})

        train_rows, eval_rows = load_subsets(config["dataset"], seed)
        overlap = set(train_rows["question"]).intersection(eval_rows["question"])
        if overlap:
            raise RuntimeError(
                f"Train/test leakage detected for {len(overlap)} questions"
            )
        append_jsonl(
            log_path,
            {
                "event": "data_ready",
                "train": len(train_rows),
                "eval": len(eval_rows),
                "question_overlap": 0,
            },
        )

        responses_by_question = None
        if config["teacher"].get("trace_source") == "generated":
            records = read_trace_records(config["teacher"]["trace_path"])
            responses_by_question, teacher_metrics = validate_trace_coverage(
                train_rows, records
            )
        else:
            teacher_metrics = {
                "examples": len(train_rows),
                "correct": len(train_rows),
                "accuracy": 1.0,
            }
        append_jsonl(log_path, {"event": "teacher_traces_ready", **teacher_metrics})

        model, tokenizer = load_student(
            config["student"], config["dataset"]["cache_dir"]
        )

        base_started = time.perf_counter()
        base_metrics = evaluate(
            model,
            tokenizer,
            eval_rows,
            config["evaluation"],
            metrics_dir / "b0_predictions.jsonl",
        )
        base_seconds = time.perf_counter() - base_started
        base_accuracy = float(base_metrics["accuracy"])
        append_jsonl(log_path, {"event": "b0_complete", **base_metrics})

        sft_dataset = GSM8KSFTDataset(
            train_rows,
            tokenizer,
            int(config["training"]["max_length"]),
            responses_by_question=responses_by_question,
        )
        if len(sft_dataset) != len(train_rows):
            raise RuntimeError("At least one training example lost all answer tokens")
        train_started = time.perf_counter()
        model, train_metrics = train_lora(
            model,
            tokenizer,
            sft_dataset,
            config["training"],
            checkpoint_dir,
            seed,
        )
        train_seconds = time.perf_counter() - train_started
        append_jsonl(log_path, {"event": "b1_training_complete", **train_metrics})

        eval_started = time.perf_counter()
        distilled_metrics = evaluate(
            model,
            tokenizer,
            eval_rows,
            config["evaluation"],
            metrics_dir / "b1_predictions.jsonl",
        )
        distilled_eval_seconds = time.perf_counter() - eval_started
        distilled_accuracy = float(distilled_metrics["accuracy"])
        total_seconds = base_seconds + train_seconds + distilled_eval_seconds
        is_smoke = config["profile"] == "resource_adapted_smoke"
        if is_smoke:
            notes = (
                "pipeline_validation_only; offline GSM8K reference reasoning "
                "used as clean trace proxy; intended 7B teacher and 1.5B "
                "student were not run"
            )
            status = "smoke_success"
            claim_scope = "engineering smoke test only"
        else:
            notes = "paper_faithful_mvp; generated clean Teacher reasoning traces"
            status = "paper_mvp_success"
            claim_scope = "paper-faithful B0/B1 MVP, seed 42"
        steps = int(train_metrics["global_step"])
        teacher_accuracy = float(teacher_metrics["accuracy"])
        results_path = Path(config["outputs"]["results_csv"])
        common_result = {
            "run_id": run_id,
            "config": config,
            "base_accuracy": base_accuracy,
            "train_tokens": sft_dataset.train_tokens,
            "teacher_accuracy": teacher_accuracy,
            "retained_examples": len(sft_dataset),
            "status": status,
            "notes": notes,
        }
        append_result(
            results_path,
            result_row(
                method="B0-Base",
                accuracy=base_accuracy,
                steps=0,
                wall_seconds=base_seconds,
                **common_result,
            ),
        )
        append_result(
            results_path,
            result_row(
                method="B1-Clean-KD",
                accuracy=distilled_accuracy,
                steps=steps,
                wall_seconds=train_seconds + distilled_eval_seconds,
                **common_result,
            ),
        )
        summary = {
            "run_id": run_id,
            "profile": config["profile"],
            "teacher_trace_accuracy": teacher_accuracy,
            "base_accuracy": base_accuracy,
            "distilled_accuracy": distilled_accuracy,
            "distill_gain": distilled_accuracy - base_accuracy,
            "train_tokens": sft_dataset.train_tokens,
            "train_metrics": train_metrics,
            "wall_seconds": total_seconds,
            "claim_scope": claim_scope,
        }
        with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        append_jsonl(log_path, {"event": "run_complete", **summary})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        lock.release()


if __name__ == "__main__":
    main()

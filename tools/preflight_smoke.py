from __future__ import annotations

from transformers import AutoTokenizer

from src.common import load_yaml
from src.data.gsm8k import GSM8KSFTDataset, load_subsets


def main() -> None:
    config = load_yaml("configs/resource_adapted_smoke.yaml")
    seed = int(config["seed"])
    train_rows, eval_rows = load_subsets(config["dataset"], seed)
    overlap = set(train_rows["question"]).intersection(eval_rows["question"])
    if overlap:
        raise RuntimeError(f"train/test overlap: {len(overlap)}")

    student = config["student"]
    tokenizer = AutoTokenizer.from_pretrained(
        student["id"],
        revision=student.get("revision"),
        cache_dir=config["dataset"]["cache_dir"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    training = GSM8KSFTDataset(
        train_rows, tokenizer, int(config["training"]["max_length"])
    )
    answer_lengths = [
        sum(label != -100 for label in item["labels"]) for item in training.items
    ]
    if len(training) != len(train_rows) or min(answer_lengths) <= 0:
        raise RuntimeError("answer-label truncation check failed")

    print(f"train_examples={len(train_rows)}")
    print(f"eval_examples={len(eval_rows)}")
    print("question_overlap=0")
    print(f"supervised_answer_tokens={training.train_tokens}")
    print(f"min_answer_tokens={min(answer_lengths)}")
    print(f"max_answer_tokens={max(answer_lengths)}")
    print("preflight=PASS")


if __name__ == "__main__":
    main()

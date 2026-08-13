from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset


SYSTEM_PROMPT = (
    "Solve the math word problem step by step. End with the final numeric "
    "answer in the exact form: #### <answer>."
)


def load_subsets(config: dict[str, Any], seed: int) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(
        config["name"],
        config.get("config"),
        revision=config.get("revision"),
        cache_dir=config.get("cache_dir"),
    )
    train = dataset[config.get("train_split", "train")].shuffle(seed=seed)
    evaluation = dataset[config.get("eval_split", "test")].shuffle(seed=seed)
    train_count = min(int(config["train_examples"]), len(train))
    eval_count = min(int(config["eval_examples"]), len(evaluation))
    return train.select(range(train_count)), evaluation.select(range(eval_count))


def prompt_text(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class GSM8KSFTDataset(TorchDataset):
    def __init__(
        self,
        rows: Dataset,
        tokenizer,
        max_length: int,
        responses_by_question: dict[str, str] | None = None,
        sample_weights: dict[str, float] | None = None,
    ):
        self.items: list[dict[str, Any]] = []
        eos = tokenizer.eos_token or ""
        for row in rows:
            prompt_ids = tokenizer(
                prompt_text(tokenizer, row["question"]), add_special_tokens=False
            )["input_ids"]
            response = (
                responses_by_question[row["question"]]
                if responses_by_question is not None
                else row["answer"]
            )
            answer_ids = tokenizer(response + eos, add_special_tokens=False)[
                "input_ids"
            ]
            input_ids = (prompt_ids + answer_ids)[:max_length]
            prompt_length = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            if not any(label != -100 for label in labels):
                continue
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                    "sample_weight": float((sample_weights or {}).get(row["question"], 1.0)),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    @property
    def train_tokens(self) -> int:
        return sum(
            sum(label != -100 for label in item["labels"]) for item in self.items
        )


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels, weights = [], [], [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            weights.append(float(feature["sample_weight"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(weights, dtype=torch.float32),
        }

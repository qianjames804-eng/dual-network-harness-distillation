from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.data.gsm8k import prompt_text
from src.eval.answers import extract_final_answer


def evaluate(
    model,
    tokenizer,
    rows,
    config: dict[str, Any],
    predictions_path: str | Path,
) -> dict[str, Any]:
    target = Path(predictions_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    correct = 0
    records = []
    model.eval()
    original_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = bool(config.get("do_sample", False))
    if not generation_config.do_sample:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None

    for index, row in enumerate(tqdm(rows, desc="GSM8K eval")):
        prompt = prompt_text(tokenizer, row["question"])
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(config["max_prompt_length"]),
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                generation_config=generation_config,
                max_new_tokens=int(config["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response_ids = generated[0, encoded["input_ids"].shape[1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        predicted = extract_final_answer(response)
        expected = extract_final_answer(row["answer"])
        is_correct = predicted == expected and expected is not None
        correct += int(is_correct)
        records.append(
            {
                "index": index,
                "question": row["question"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "response": response,
            }
        )

    model.config.use_cache = original_cache
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"accuracy": correct / max(len(records), 1), "examples": len(records)}

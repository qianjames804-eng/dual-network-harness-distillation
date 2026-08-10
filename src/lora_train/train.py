from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from src.data.gsm8k import CausalCollator


class WeightedTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        weights = inputs.pop("sample_weight")
        labels = inputs["labels"]
        outputs = model(**inputs)
        shifted_logits = outputs.logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shifted_labels)
        mask = shifted_labels.ne(-100)
        per_example = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        weights = weights.to(per_example.device)
        loss = (per_example * weights).sum() / weights.sum().clamp_min(1)
        return (loss, outputs) if return_outputs else loss


def train_lora(
    base_model,
    tokenizer,
    train_dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    seed: int,
):
    if config.get("gradient_checkpointing", False):
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
    base_model.config.use_cache = False
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias="none",
    )
    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    dtype_name = str(config.get("dtype", "bfloat16"))
    use_bf16 = (
        torch.cuda.is_available()
        and dtype_name == "bfloat16"
        and torch.cuda.is_bf16_supported()
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config["per_device_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        max_steps=int(config.get("max_steps", -1)),
        num_train_epochs=float(config.get("epochs", 1)),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config.get("warmup_ratio", 0.0)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=seed,
    )
    collator = CausalCollator(tokenizer.pad_token_id)
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    metrics = trainer.train().metrics
    metrics["global_step"] = int(trainer.state.global_step)
    final_dir = Path(output_dir) / "final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    return model, metrics

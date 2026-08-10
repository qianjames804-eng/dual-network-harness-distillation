from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_causal_lm(config: dict[str, Any], cache_dir: str):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment profile")
    dtype_name = str(config.get("dtype", "bfloat16"))
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported model dtype: {dtype_name}")
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU/PyTorch build does not support bfloat16")

    common = {
        "revision": config.get("revision"),
        "cache_dir": cache_dir,
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    tokenizer = AutoTokenizer.from_pretrained(config["id"], **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_options: dict[str, Any] = {
        **common,
        "torch_dtype": DTYPES[dtype_name],
        "low_cpu_mem_usage": True,
    }
    if config.get("attn_implementation"):
        model_options["attn_implementation"] = config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(
        config["id"], **model_options
    ).to("cuda")
    return model, tokenizer

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

import torch


def main() -> None:
    cache_dir = Path(os.environ.get("HF_HOME", ".cache/huggingface"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    architecture = platform.machine().lower()
    cuda_available = torch.cuda.is_available()
    report = {
        "platform": platform.platform(),
        "architecture": architecture,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
        "hf_home": str(cache_dir.resolve()),
        "cache_free_gib": round(shutil.disk_usage(cache_dir).free / 2**30, 1),
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        report.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(capability),
                "reported_gpu_memory_gib": round(properties.total_memory / 2**30, 1),
            }
        )
    failures = []
    if architecture not in {"aarch64", "arm64"}:
        failures.append("container is not running on ARM64")
    if not cuda_available:
        failures.append("CUDA is not visible inside the container")
    if cuda_available and not torch.cuda.is_bf16_supported():
        failures.append("bfloat16 is unavailable")
    if cuda_available and tuple(report["compute_capability"]) < (12, 0):
        failures.append("GPU is not a Blackwell-class DGX Spark device")
    if report["cache_free_gib"] < 80:
        failures.append("less than 80 GiB free in persistent model cache")
    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

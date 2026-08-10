from __future__ import annotations

import argparse
from pathlib import Path

from src.common import load_yaml


def validate(config: dict) -> None:
    if not config.get("profile"):
        raise ValueError("profile is required")
    if "seed" not in config and "seeds" not in config:
        raise ValueError("seed or seeds is required")
    training = config.get("training", {})
    for key in ("learning_rate", "lora_r", "lora_alpha", "lora_dropout"):
        if key not in training:
            raise ValueError(f"training.{key} is required")
    if config.get("profile") in {"resource_adapted_smoke", "paper_faithful_mvp"}:
        for section in ("dataset", "teacher", "student", "evaluation", "outputs"):
            if not isinstance(config.get(section), dict):
                raise ValueError(f"{section} mapping is required")
        if config["teacher"].get("trace_source") == "generated" and not config[
            "teacher"
        ].get("trace_path"):
            raise ValueError("teacher.trace_path is required for generated traces")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate(config)
    print(f"validated: {args.config} ({config['profile']})")


if __name__ == "__main__":
    main()

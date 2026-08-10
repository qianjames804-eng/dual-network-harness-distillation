from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_PATHS = [
    ".cnb.yml",
    ".dockerignore",
    "Dockerfile.spark",
    "requirements-spark.lock",
    "configs/paper_faithful_mvp.yaml",
    "scripts/spark/preflight.sh",
    "scripts/spark/run.sh",
    "src/traces/generate_teacher.py",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_PATHS if not (root / path).is_file()]
    if missing:
        raise SystemExit(f"missing Spark/CNB files: {', '.join(missing)}")

    with (root / ".cnb.yml").open("r", encoding="utf-8") as handle:
        cnb = yaml.safe_load(handle)
    if not isinstance(cnb, dict) or "$" not in cnb:
        raise SystemExit(".cnb.yml must define the all-branches '$' trigger")

    with (root / "configs/paper_faithful_mvp.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    if config["teacher"]["trace_source"] != "generated":
        raise SystemExit("paper MVP must use generated Teacher traces")
    if "7B" not in config["teacher"]["id"] or "1.5B" not in config["student"]["id"]:
        raise SystemExit("paper MVP model identities were unexpectedly changed")

    requirements = [
        line.strip().lower()
        for line in (root / "requirements-spark.lock").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any(line.startswith("torch") for line in requirements):
        raise SystemExit("requirements-spark.lock must preserve NGC's PyTorch build")
    print("Spark/CNB package validation: pass")


if __name__ == "__main__":
    main()

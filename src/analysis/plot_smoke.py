from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    smoke = frame[frame["status"] == "smoke_success"].copy()
    if smoke.empty:
        raise ValueError("No successful smoke rows to plot")
    latest_run = smoke.iloc[-1]["run_id"]
    smoke = smoke[smoke["run_id"] == latest_run]
    smoke = smoke[smoke["method"].isin(["B0-Base", "B1-Clean-KD"])]
    order = ["B0-Base", "B1-Clean-KD"]
    smoke["method"] = pd.Categorical(smoke["method"], order, ordered=True)
    smoke = smoke.sort_values("method")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "smoke_b0_b1.csv"
    smoke[["run_id", "method", "student_accuracy", "seed", "notes"]].to_csv(
        data_path, index=False
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    bars = axis.bar(
        smoke["method"].astype(str),
        smoke["student_accuracy"],
        color=["#667085", "#2E90FA"],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("GSM8K exact-answer accuracy")
    axis.set_title("Resource-adapted B0/B1 pipeline smoke test")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, smoke["student_accuracy"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(float(value) + 0.03, 0.97),
            f"{float(value):.3f}",
            ha="center",
        )
    figure.tight_layout()
    figure.savefig(args.output_dir / "smoke_b0_b1.png", dpi=180)
    figure.savefig(args.output_dir / "smoke_b0_b1.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()


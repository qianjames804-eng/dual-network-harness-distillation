"""Aggregate the completed GSM8K seed-42 five-arm Rewrite study.

This tool intentionally produces only figures supported by this run: it does
not fabricate multi-seed error bars, MATH comparisons, or ablations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = ("B0-Base", "B1-Clean-KD", "B4-Rewrite-KD", "Ours-Clean", "Ours-Rewrite")
PREFIX = {
    "B0-Base": "b0_base-s42-",
    "B1-Clean-KD": "b1_clean_kd-s42-",
    "B4-Rewrite-KD": "b4_rewrite_kd-s42-",
    "Ours-Clean": "ours_clean-s42-",
    "Ours-Rewrite": "ours_rewrite-s42-",
}


def latest_run(root: Path, prefix: str) -> Path:
    candidates = sorted(path for path in root.glob(prefix + "*") if (path / "metrics.json").exists())
    if not candidates:
        raise FileNotFoundError(f"No completed run for {prefix}")
    return candidates[-1]


def bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, *, colors=None) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(labels, values, color=colors)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20)
    for item, value in zip(bars, values):
        axis.text(item.get_x() + item.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def number(value) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/rewrite_five_s42"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        run = latest_run(args.outputs, PREFIX[method])
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        train = metrics["train_metrics"]
        rows.append({
            "method": method, "run_dir": str(run), "accuracy": metrics["accuracy"],
            "exact_match": metrics["exact_match"], "delta_vs_b0": metrics["delta_vs_b0"],
            "train_tokens": metrics["train_tokens"], "global_step": train.get("global_step"),
            "weighted_loss_consumed_batches": train.get("weighted_loss_consumed_batches"),
            "nn1_spearman": metrics["nn1_spearman"], "nn1_auroc": metrics["nn1_auroc"],
            "nn2_auroc": metrics["nn2_auroc"], "nn2_f1": metrics["nn2_f1"],
            "nn2_brier": metrics["nn2_brier"], "nn2_ece": metrics["nn2_ece"],
            "nn2_spearman": metrics["nn2_spearman"],
        })
    by_method = {row["method"]: row for row in rows}
    rewrite_drop = by_method["B1-Clean-KD"]["accuracy"] - by_method["B4-Rewrite-KD"]["accuracy"]
    recovery = by_method["Ours-Rewrite"]["accuracy"] - by_method["B4-Rewrite-KD"]["accuracy"]
    recovery_rate = recovery / rewrite_drop if rewrite_drop else None
    fields = list(rows[0])
    with (args.output_dir / "rewrite_five_seed42_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = {"dataset": "GSM8K", "seed": 42, "runs": rows, "rewrite_drop": rewrite_drop, "recovery": recovery, "recovery_rate": recovery_rate, "scope": "single-seed, five-arm Trace Rewriting study; no multi-seed, MATH, or ablation inference"}
    (args.output_dir / "rewrite_five_seed42_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "rewrite_five_seed42_report.md").write_text(
        "# GSM8K seed=42 Trace Rewriting five-arm result\n\n"
        "Single-seed results only; no confidence intervals are implied.\n\n"
        "| Method | Accuracy / EM | Δ vs B0 | NN1 Spearman | NN2 AUROC | NN2 Brier |\n|---|---:|---:|---:|---:|---:|\n" +
        "".join(f"| {row['method']} | {row['accuracy']:.4f} | {row['delta_vs_b0']:.4f} | {number(row['nn1_spearman'])} | {number(row['nn2_auroc'])} | {number(row['nn2_brier'])} |\n" for row in rows) +
        f"\n- Rewrite Drop = Acc(B1) − Acc(B4) = {rewrite_drop:.4f}\n- Recovery = Acc(Ours-Rewrite) − Acc(B4) = {recovery:.4f}\n- Recovery Rate = {recovery_rate:.4f}\n",
        encoding="utf-8",
    )
    bar(args.output_dir / "fig1_accuracy", [r["method"] for r in rows], [100 * r["accuracy"] for r in rows], "GSM8K seed=42: accuracy", "Accuracy (%)")
    selected = [by_method[name] for name in ("B1-Clean-KD", "B4-Rewrite-KD", "Ours-Clean", "Ours-Rewrite")]
    bar(args.output_dir / "fig2_clean_rewrite", [r["method"] for r in selected], [100 * r["accuracy"] for r in selected], "Clean vs Trace Rewriting", "Accuracy (%)")
    ours = [by_method["Ours-Clean"], by_method["Ours-Rewrite"]]
    bar(args.output_dir / "fig3_nn1", ["Clean Spearman", "Clean AUROC", "Rewrite Spearman", "Rewrite AUROC"], [ours[0]["nn1_spearman"], ours[0]["nn1_auroc"], ours[1]["nn1_spearman"], ours[1]["nn1_auroc"]], "NN1 proxy-utility metrics", "Score")
    bar(args.output_dir / "fig4_nn2", ["Clean AUROC", "Clean F1", "Rewrite AUROC", "Rewrite F1"], [ours[0]["nn2_auroc"], ours[0]["nn2_f1"], ours[1]["nn2_auroc"], ours[1]["nn2_f1"]], "NN2 mastery-Judgment discrimination", "Score")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

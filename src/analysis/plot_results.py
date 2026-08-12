"""Generate paper figures and a multi-seed mean±std table from results.csv."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path.with_suffix('.png'),dpi=180); fig.savefig(path.with_suffix('.svg')); plt.close(fig)
def plot(frame, x, y, hue, title, path):
    fig,ax=plt.subplots(figsize=(11,5)); sns.barplot(data=frame,x=x,y=y,hue=hue,errorbar="sd",ax=ax); ax.set_title(title); ax.tick_params(axis="x",rotation=30); save(fig,path)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,default=Path("results.csv")); p.add_argument("--output-dir",type=Path,default=Path("outputs/figures/paper")); a=p.parse_args(); raw=pd.read_csv(a.results)
    for c in ("student_accuracy","distill_gain","nn1_spearman","nn2_spearman"): raw[c]=pd.to_numeric(raw[c],errors="coerce")
    group=raw.groupby(["dataset","method","defense"],dropna=False).agg(accuracy_mean=("student_accuracy","mean"),accuracy_std=("student_accuracy","std"),gain_mean=("distill_gain","mean"),runs=("run_id","nunique")).reset_index(); a.output_dir.mkdir(parents=True,exist_ok=True); group.to_csv(a.output_dir/"complete_results_table.csv",index=False)
    plot(raw,"method","student_accuracy","dataset","Overall performance",a.output_dir/"fig1_overall")
    plot(raw[raw.defense.isin(["clean","ads","doge","rewrite"])],"defense","student_accuracy","method","Clean vs anti-distillation",a.output_dir/"fig2_antidistill")
    plot(raw.dropna(subset=["nn1_spearman"]),"method","nn1_spearman","defense","NN1 data-filter utility correlation",a.output_dir/"fig3_nn1_filter")
    plot(raw.dropna(subset=["nn2_spearman"]),"method","nn2_spearman","dataset","NN2 Judgment vs actual mastery",a.output_dir/"fig4_nn2_judgment")
    plot(raw[raw.method.isin(["No-NN","NN1-only","NN2-only","NN1+NN2"])],"method","student_accuracy","dataset","Dual-network ablation",a.output_dir/"fig5_ablation")
    plot(raw,"dataset","student_accuracy","method","GSM8K / MATH cross-task",a.output_dir/"fig6_cross_task")
if __name__ == "__main__": main()

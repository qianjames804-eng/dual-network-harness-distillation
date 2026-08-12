"""NN2: generated-output/adapter-feature → actual mastery Judgment pipeline.

The NN2 label is mean correctness over an original evaluation question and a
held-out paraphrase.  Paraphrases never enter SFT; NN2 is an evaluator in the
primary study, not a hidden source of Student supervision.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from .model import MasteryMLP

@dataclass
class NN2Result: scores: np.ndarray; auroc: float | None; f1: float; brier: float; ece: float; spearman: float
def mastery_targets(original: np.ndarray, paraphrase: np.ndarray) -> np.ndarray: return (np.asarray(original, dtype=np.float32)+np.asarray(paraphrase, dtype=np.float32))/2
def ece(scores: np.ndarray, labels: np.ndarray, bins: int=10) -> float:
    ans=0.; edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(scores>=lo)&((scores<hi) if hi<1 else (scores<=hi)); ans += float(m.mean()*abs(scores[m].mean()-labels[m].mean())) if m.any() else 0
    return ans
def fit_judge(features: np.ndarray, targets: np.ndarray, *, epochs: int, lr: float, seed: int, hidden_dim: int=128) -> NN2Result:
    torch.manual_seed(seed); model=MasteryMLP(features.shape[1],hidden_dim); opt=torch.optim.AdamW(model.parameters(),lr=lr); x,y=torch.tensor(features,dtype=torch.float32),torch.tensor(targets,dtype=torch.float32)
    for _ in range(epochs): opt.zero_grad(); loss=torch.nn.functional.mse_loss(model(x),y); loss.backward(); opt.step()
    with torch.no_grad(): scores=model(x).numpy()
    binary=(targets>=.5).astype(int); auc=float(roc_auc_score(binary,scores)) if len(set(binary))==2 else None
    return NN2Result(scores,auc,float(f1_score(binary,scores>=.5,zero_division=0)),float(brier_score_loss(binary,scores)),ece(scores,targets),float(spearmanr(scores,targets).statistic))

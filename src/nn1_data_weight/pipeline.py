"""NN1: trace-feature → proxy-utility weight/filter pipeline.

Labels are built on a dedicated calibration split: each label is the positive
change in clean validation loss after one proxy-Student update on its trace.
The final test split is never used to construct an NN1 label.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from .model import DataWeightMLP

@dataclass
class NN1Result: weights: np.ndarray; spearman: float; auroc: float | None

def trace_features(records: list[dict]) -> np.ndarray:
    questions = [str(r["question"]) for r in records]; count = {q: questions.count(q) for q in set(questions)}
    rows = [[len(str(r["question"])), len(str(r.get("response", ""))), float(r.get("response_tokens", 0)), count[str(r["question"])], float(bool(r.get("correct", False))), float(r.get("token_entropy", 0)), float(r.get("base_nll", 0))] for r in records]
    x = np.asarray(rows, dtype=np.float32); return (x - x.mean(0)) / (x.std(0) + 1e-6)

def utility_targets(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(before) - np.asarray(after), 0); return value / max(float(value.max()), 1e-8)

def fit_predict(features: np.ndarray, targets: np.ndarray, *, epochs: int, lr: float, seed: int, hidden_dim: int = 128) -> NN1Result:
    torch.manual_seed(seed); model = DataWeightMLP(features.shape[1], hidden_dim); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x, y = torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    for _ in range(epochs): opt.zero_grad(); loss=torch.nn.functional.mse_loss(model(x), y); loss.backward(); opt.step()
    with torch.no_grad(): weights=model(x).numpy()
    labels=(targets > np.median(targets)).astype(int); auc=float(roc_auc_score(labels, weights)) if len(set(labels)) == 2 else None
    return NN1Result(weights, float(spearmanr(weights, targets).statistic), auc)

def retained_indices(weights: np.ndarray, retention: float) -> np.ndarray:
    if not 0 < retention <= 1: raise ValueError("retention must be in (0,1]")
    return np.argsort(weights)[-max(1, round(len(weights) * retention)):]

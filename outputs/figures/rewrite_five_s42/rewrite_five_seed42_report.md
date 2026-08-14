# GSM8K seed=42 Trace Rewriting five-arm result

Single-seed results only; no confidence intervals are implied.

| Method | Accuracy / EM | Δ vs B0 | NN1 Spearman | NN2 AUROC | NN2 Brier |
|---|---:|---:|---:|---:|---:|
| B0-Base | 0.7031 | 0.0000 |  |  |  |
| B1-Clean-KD | 0.5703 | -0.1328 |  |  |  |
| B4-Rewrite-KD | 0.0703 | -0.6328 |  |  |  |
| Ours-Clean | 0.5625 | -0.1406 | 0.0671 | 0.9221 | 0.1252 |
| Ours-Rewrite | 0.0625 | -0.6406 | 0.4542 | 0.5312 | 0.2424 |

- Rewrite Drop = Acc(B1) − Acc(B4) = 0.5000
- Recovery = Acc(Ours-Rewrite) − Acc(B4) = -0.0078
- Recovery Rate = -0.0156

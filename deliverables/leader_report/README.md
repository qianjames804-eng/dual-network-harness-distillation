# Dual-Network Harness Distillation 阶段性汇报

建议阅读顺序：

1. [实验总结](01_experiment_summary.md)：了解目标、NN1/NN2 分工和实验链路。
2. [核心结果表](02_results_table.csv) 与 [阶段性结论](04_stage_conclusion.md)：先看性能变化与当前边界。
3. [NN 指标表](03_nn_metrics.csv) 和 `figures/`：查看两类网络的信号质量。
4. [下一步计划](05_next_steps.md) 与 [可复现信息](06_reproducibility.md)：了解后续验证路径和结果来源。

本目录只汇总已完成的 GSM8K、seed=42、Trace Rewriting 五组正式实验。所有数字和图来自真实 run；没有 mock 数据、补写结果或新训练。该轮仅一个 seed，不能用于多 seed、跨任务或最终论文结论。

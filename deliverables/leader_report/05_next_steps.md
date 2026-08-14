# 下一步计划

1. 运行 3 seeds 稳定性实验，报告 mean ± std，并先检查 B4/Ours-Rewrite 的不稳定来源。
2. 运行 No-NN、NN1-only、NN2-only、NN1+NN2 消融，区分两类网络各自的边际作用。
3. 分析 NN1 权重机制：utility 标签分布、权重分布、top/bottom 权重样本的训练贡献和质量差异。
4. 分析 NN2 在 Rewrite 场景下降的原因：paraphrase 一致性、错误类型、confidence/NLL 校准和 LoRA 特征分布漂移。
5. 在 GSM8K 链路稳定后扩展到 MATH，保持相同的隔离 gate 和固定实验协议。
6. 后续接入 DOGe official-code reproduction 和 ADS 官方公开 artifact；在 artifact/provenance 完整前不把近似实现标记为官方结果。

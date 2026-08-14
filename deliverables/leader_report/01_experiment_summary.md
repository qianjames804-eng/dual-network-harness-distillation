# 实验总结

本阶段目标是验证 Dual-Network Harness Distillation 的完整工程链路能否在真实 GSM8K 数据和官方 Trace Rewriting 输入上运行：从 Teacher trace 输入，到 Student 的 LoRA/KD 训练，再到双神经网络的质量控制与能力判断。此次只运行 seed=42、512 条固定训练子集和 128 条固定评测题，比较 B0、Clean-KD、Rewrite-KD、Ours-Clean、Ours-Rewrite 五组。

系统中有两个职责明确的神经网络。NN1 面向数据：它从 Question/Answer/trace 特征出发，对校准集中的每条 trace 做一次真实 proxy update，测量 clean validation loss 是否改善，得到 utility 标签；再输出样本权重。该权重真实进入 Student 的 weighted LoRA/KD loss，因此 NN1 负责数据筛选和训练信号的强弱分配。NN2 面向结果：Student 分别回答原题和独立 paraphrase 题，NN2 使用输出置信度、NLL、一致性、LoRA A/B/BA 范数、梯度范数和 adapter on/off 差异，判断模型是否真正掌握能力。NN2 不向 Student 提供训练标签或训练样本。

Harness 将三段串成可审计流程：先通过 SHA-256 manifest 隔离 SFT、NN1 calibration、NN2 train/validation 和最终 test；NN1 在独立 calibration 中学习 utility 并给 SFT 样本赋权；Student 在固定 Student、LoRA、token budget、optimizer、epoch 和 decoding 配置下进行 LoRA/KD；NN2 最后在独立 calibration questions 及其合格 paraphrase 上评估 mastery。每组都保存 config、checkpoint metadata、预测、原始 metrics 和相应的 NN1/NN2 文件。

本轮的关键价值是“真实链路已跑通”，而不是证明方法已经成功。Trace Rewriting 对 Clean-KD 造成了显著破坏；当前 Ours-Rewrite 在一个 seed 下没有恢复性能。后续应把这当作定位问题的基线，继续做多 seed、消融和跨任务验证。

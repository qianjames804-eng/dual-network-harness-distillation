# 阶段性结论

- 完整链路已经真实跑通：官方 Trace Rewriting 输入、NN1 utility/权重、weighted LoRA/KD、NN2 Judgment、最终 GSM8K evaluation 和结果落盘均已执行。
- Trace Rewriting 对 Clean-KD 造成明显性能下降：B1 从 57.03% 降至 B4 的 7.03%，Rewrite Drop 为 **50.00pp**。
- 当前单 seed 下 Ours-Rewrite 未恢复性能：Ours-Rewrite 为 6.25%，相对 B4 的 Recovery 为 **-0.78pp**，Recovery Rate 为 **-1.56%**。
- NN1 在 Rewrite 场景出现一定识别信号：Ours-Rewrite 的 NN1 Spearman 为 0.4542、AUROC 为 0.6367；但这还不足以证明权重机制稳定有效。
- NN2 在 Clean 场景效果较好：Ours-Clean 的 AUROC 为 0.9221、F1 为 0.8636、Spearman 为 0.8370；在 Rewrite 场景明显下降至 AUROC 0.5312、F1 0.3750、Spearman 0.1180。
- 本结果只是一阶段验证。它来自 GSM8K 的一个固定 seed，不能直接推导多 seed 稳定性、跨任务结论或最终方法有效性。

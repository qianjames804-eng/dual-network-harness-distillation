# 双神经网络 Harness 蒸馏对比实验报告

生成日期：2026-08-10（Asia/Shanghai）

## 1. 当前结论

原方案第一道工程门禁 B0 Base + B1 Clean-KD 已在原 Windows 主机跑通，数据、
LoRA、GSM8K exact-answer 评测、统一结果表、checkpoint、JSONL 日志和审计链路
可工作。但该机器只有 GTX 1650 4 GB，因此该结果使用显式标注的
`resource_adapted_smoke`，不能支持论文假设。

面向用户的 DGX Spark，paper-MVP 代码已经改为真实
`DeepSeek-R1-Distill-Qwen-7B` trace → `Qwen2.5-1.5B-Instruct` LoRA/SFT，
并加入 ARM64 NGC 容器、Spark 预检、trace 断点续跑和 CNB CPU gate。该路径
尚未在用户的物理 Spark 上执行，因此目前没有 paper-faithful 指标。

## 2. 已完成的本机 smoke

- run_id：`resource_adapted_smoke-s42-20260810T040742Z`
- 数据：openai/gsm8k 固定 revision；16 train + 8 test；seed 42
- Student：Qwen2.5-0.5B-Instruct 固定 revision
- trace：GSM8K reference answer 离线 proxy，不是 7B Teacher 输出
- LoRA：r=16、alpha=32、dropout=0.05
- supervised answer tokens：1,644

| 方法 | Accuracy | 相对 B0 增益 |
|---|---:|---:|
| B0 Base | 0.000 | 0.000 |
| B1 Clean-KD | 0.375 | +0.375 |

这些数值只证明工程链路能运行。样本只有 8 题、单 seed、Student 和 trace 均为
资源适配版本，不能估计真实 7B Teacher 蒸馏增益。

## 3. DGX Spark paper-MVP 实现

执行顺序：

1. 容器内验证 ARM64、CUDA、BF16、Blackwell 计算能力和持久盘空间。
2. 固定 GSM8K revision 和 seed 42，逐条生成 7B Teacher clean traces。
3. 每条 trace 立即写入 JSONL；续跑时校验 index、question、revision 和 seed。
4. 统计 Teacher trace exact-answer accuracy，而不是硬编码 1.0。
5. 顺序卸载 Teacher 进程，再加载 1.5B Student，运行 B0。
6. 以真实 Teacher response 构造监督 token，运行 LoRA/SFT 和 B1。
7. 将预测、adapter、summary、耗时、训练 token 和统一结果写盘。

## 4. 公平性与审计

- GSM8K train/test 使用官方不同 split，并检查 question 文本交集。
- B0/B1 使用相同 Student revision、prompt、test subset 和答案解析器。
- 模型/data revision、seed、LoRA 参数和训练 token 数进入配置或结果日志。
- paper-MVP 不再使用 reference answer proxy，也不静默替换 Teacher/Student。
- Teacher trace 不完整或与 dataset 顺序不符时，Harness 会拒绝训练。
- `run_full.sh` 在未通过 NN1/NN2/防御门禁前明确返回非零状态。

## 5. 尚未执行，禁止据此作结论

- DGX Spark 上真实 7B → 1.5B 的 B0/B1 数值
- NN1 proxy utility、weighted loss、top-k retention
- NN2 mastery 校准和 AUROC/F1/Brier/ECE/Spearman
- No-NN / NN1-only / NN2-only / Dual 消融
- ADS、DOGe、Trace Rewriting
- 3 seeds、MATH/MATH-500、fixed-train-token 对比和最终七张论文图

下一道门禁是在 DGX Spark 上跑完 paper-MVP 并审计 `results.csv`。只有该结果
可重复后，才按原方案进入 NN1；Full 实验不能提前解锁。

# 可复现信息

## 本轮范围

- 数据集：GSM8K，最终 evaluation 固定 128 题。
- seed：42。
- 训练输入：固定 512 条 GSM8K 子集中的官方 Trace Rewriting trace；五组采用相同 Student、LoRA rank、optimizer、epoch、token budget 和 decoding 设置。
- Student：`Qwen/Qwen2.5-1.5B-Instruct`。
- NN2 paraphraser：本地只读 `/home/zixiao/Qwen3-model/Qwen3-8B`；20 条 GSM8K strict gate 的最终通过率为 100%。
- 分组：B0 Base、B1 Clean-KD、B4 Rewrite-KD、Ours-Clean、Ours-Rewrite。

## 数据隔离与审计

运行时 SHA-256 manifest 检查 SFT、NN1 calibration、NN2 calibration train、NN2 calibration validation 和最终 test 的交集；发现交集即终止。NN1 utility 仅用 NN1 calibration；NN2 mastery 仅来自 NN2 calibration questions 和独立 paraphrase；最终 test 不参与标签构造。

## 结果来源

- B0：`outputs/b0_base-s42-20260813T060036Z/metrics.json`
- B1：`outputs/b1_clean_kd-s42-20260813T064129Z/metrics.json`
- B4：`outputs/b4_rewrite_kd-s42-20260813T071922Z/metrics.json`
- Ours-Clean：`outputs/ours_clean-s42-20260813T073348Z/metrics.json`
- Ours-Rewrite：`outputs/ours_rewrite-s42-20260813T084521Z/metrics.json`

每个 run 目录包含固定 config、JSONL predictions、metrics；训练组还含 `checkpoints/final_adapter/`。Ours 组另含 `nn1_weights.jsonl` 和 `nn2_predictions.json`。本汇报图基于上述 metrics 生成，没有修改原始实验结果。

## GitHub

- 分支：`main`
- 最新结果 commit：`f31bd3f Summarize GSM8K seed42 rewrite five-arm results`
- 仓库：`https://github.com/qianjames804-eng/dual-network-harness-distillation.git`

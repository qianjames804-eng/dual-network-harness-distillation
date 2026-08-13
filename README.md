# 双神经网络 Harness 蒸馏对比实验

本仓库按《双神经网络Harness蒸馏对比实验_Codex执行方案》执行，并把本地
Windows smoke、DGX Spark paper-MVP、尚未解锁的 Full 实验严格分开。

## 当前可运行范围

- `resource_adapted_smoke`：已经在原 4 GB Windows GPU 上跑通的工程验证；
  使用 GSM8K 标准解答作为离线 trace proxy，不能作为论文结果。
- `paper_faithful_mvp`：DGX Spark 目标路径。使用
  `DeepSeek-R1-Distill-Qwen-7B` 生成真实 clean reasoning traces，再以相同
  trace 训练 `Qwen2.5-1.5B-Instruct` 的 LoRA，输出 B0/B1。
- `full`：只保留实验矩阵声明。NN1、NN2、ADS、DOGe、Rewrite 尚未完成门禁，
  `run_full.sh` 会明确返回非零状态，避免把配置验证误报成完整实验成功。

## DGX Spark 最短运行路径

```bash
export HF_TOKEN='<仅在模型需要鉴权时设置>'
bash scripts/spark/run.sh preflight
SPARK_SKIP_BUILD=1 bash scripts/spark/run.sh traces
SPARK_SKIP_BUILD=1 bash scripts/spark/run.sh mvp
```

Teacher trace 写入 `outputs/traces/`，每生成一条就落盘；中断后重复运行会续跑。
模型缓存写入被 Git 忽略的 `.spark-data/huggingface/`，不会进入 CNB 仓库或镜像。
详细迁移步骤见 [CNB_DGX_SPARK.md](CNB_DGX_SPARK.md)。

## CNB 的职责边界

`.cnb.yml` 在 push 和 pull request 时运行 CPU 门禁：编译、单元测试、三个配置
校验和 Spark 包装检查。默认配置不宣称 CNB 公共构建机能直接调度你的物理
DGX Spark；GPU 实验在 Spark 上 clone/pull CNB 仓库后执行。

## 输出

- `results.csv`：统一结果 schema。
- `outputs/<run_id>/metrics/`：B0/B1 预测与 summary。
- `outputs/<run_id>/checkpoints/`：LoRA adapter。
- `outputs/traces/`：可恢复的 Teacher traces。

所有 profile、模型 revision、数据 revision、seed、训练 token 数和替换说明都会
写入配置、JSONL 日志或结果表。

## 完整实验工程

本仓库区分**可重复的真实研究运行**和单元测试用的 synthetic 小样本：测试数据
绝不写入 `results.csv`，正式 trace 必须来自指定 Teacher 或相应论文的官方防御
实现。B0 是未训练 Student；B1 是 Clean Teacher trace 的 LoRA/SFT；B2/B3/B4
分别是 ADS、DOGe、Rewrite trace 的同预算 LoRA/SFT；Ours-* 是对应 trace 上的
`NN1 → weighted/filter LoRA → NN2 Judgment`。消融是 No-NN、NN1-only、NN2-only
与 NN1+NN2。

NN1 只负责样本权重/过滤：其 proxy-utility 标签为校准集上一次 proxy update
带来的干净验证 loss 改善。NN2 只负责最终能力 Judgment：其 mastery 标签是原题
与从未用于 SFT 的改写题上的平均正确率。详细的无泄漏约束见
[docs/NN_LABELS.md](docs/NN_LABELS.md)。

### 命令

```bash
# 真正的最小 B0/B1 链路（资源适配 smoke）
bash run_smoke.sh

# 真实 7B Teacher → 1.5B Student 的小规模硬件验证，不使用参考答案 trace
bash run_paper_faithful_smoke.sh

# 列出完整主矩阵（54 jobs）及消融（24 jobs），不会启动 GPU
bash run_full.sh --dry-run
bash run_ablation.sh --dry-run

# 用论文官方实现生成防御 traces；缺失 official command 时会明确失败
bash run_antidistill.sh doge --official-command '<DOGe command>'
bash run_antidistill.sh ads --official-command '<ADS command>'
bash run_antidistill.sh rewrite --official-command '<Rewrite command>'

# 从统一 CSV 生成 PNG/SVG 论文图与 mean±std 完整结果表
.venv/bin/python plot_results.py --results results.csv
```

全量执行由显式模板触发，便于记录每种 ADS/DOGe/Rewrite 官方实现的 commit 和
命令，避免静默替代防御 trace：

```bash
.venv/bin/python -m src.matrix_runner --suite full --execute \
  --split-manifest outputs/manifests/gsm8k_s42_splits.json \
  --execute-template 'your_job_command --dataset {dataset} --seed {seed} --method {method}'
```

正式矩阵为 2 数据集（GSM8K、MATH）× 3 seeds × 9 主方法 = 54 jobs，另有 24
个消融 jobs；NN1 retention 曲线和三档防御强度为附加分析 jobs。建议 32 GB
以上 GPU/统一内存（7B BF16 Teacher 推理约 18–24 GB；1.5B LoRA 约 10–16 GB）。

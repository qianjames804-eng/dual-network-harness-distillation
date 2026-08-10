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

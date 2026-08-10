# CNB → DGX Spark 迁移与运行手册

## 1. 架构边界

默认采用以下稳定路径：

1. CNB 保存 Git 仓库，并在 push/PR 上执行 CPU 质量门禁。
2. DGX Spark 从 CNB clone/pull 仓库。
3. Spark 本机基于 NVIDIA NGC PyTorch ARM64 镜像构建实验容器。
4. GPU 实验、模型缓存、Teacher traces、checkpoint 和结果都保留在 Spark。

这样不依赖 CNB 公共 runner 能访问你局域网中的 Spark，也不会把 Hugging Face
token、模型权重或大体积 trace 上传到 CNB。

## 2. 推送到 CNB

在 CNB 新建空仓库后，在当前项目目录执行（替换仓库地址）：

```bash
git init
git add .
git commit -m "prepare DGX Spark paper MVP"
git branch -M main
git remote add origin https://cnb.cool/<组织>/<仓库>.git
git push -u origin main
```

CNB 会读取根目录 `.cnb.yml`，执行以下 CPU gate：

- Python compileall
- pytest
- smoke / paper-MVP / full YAML 配置验证
- Dockerfile、ARM64 requirements、Spark 启动文件完整性检查

## 3. Spark 首次准备

先更新 DGX OS/驱动，并确认 Docker 可以直接看到 GPU：

```bash
uname -m
nvidia-smi
docker ps
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

预期架构为 `aarch64`，CUDA 可见且 GPU 为 GB10/DGX Spark。

然后 clone CNB 仓库：

```bash
git clone https://cnb.cool/<组织>/<仓库>.git
cd <仓库>
bash scripts/spark/run.sh preflight
```

预检要求：ARM64、容器内 CUDA、BF16，以及持久化缓存至少 80 GiB 可用空间。

## 4. 生成可恢复 Teacher traces

```bash
export HF_TOKEN='<仅在 Hugging Face 要求鉴权时设置>'
SPARK_SKIP_BUILD=1 bash scripts/spark/run.sh traces
```

如果希望分批生成，例如每次最多 500 条：

```bash
SPARK_SKIP_BUILD=1 TRACE_MAX_NEW_RECORDS=500 bash scripts/spark/run.sh traces
SPARK_SKIP_BUILD=1 TRACE_MAX_NEW_RECORDS=500 bash scripts/spark/run.sh traces
```

重复命令直到输出 `"status": "complete"`。程序会拒绝混用不同 dataset
revision、seed 或打乱顺序的旧 trace。

## 5. 运行 B0/B1 paper-MVP

```bash
SPARK_SKIP_BUILD=1 bash scripts/spark/run.sh mvp
```

`mvp` 会再次检查 trace 完整性，然后依次运行：

1. B0：未训练 Student 的 GSM8K test accuracy。
2. B1 训练：真实 Teacher reasoning traces → Student LoRA/SFT。
3. B1 评测：同一 GSM8K test split 和同一评测脚本。
4. 结果落盘：predictions、adapter、summary、JSONL 日志、`results.csv`。

## 6. 更新代码与续跑

```bash
git pull --ff-only
bash scripts/spark/run.sh preflight
SPARK_SKIP_BUILD=1 bash scripts/spark/run.sh traces
```

只有 requirements 或 Dockerfile 变化时才必须重新构建镜像；否则可用
`SPARK_SKIP_BUILD=1`。默认模型缓存位于仓库下 `.spark-data/huggingface/`。
如需放到另一块持久盘：

```bash
SPARK_DATA_ROOT=/data/dual-nn-harness bash scripts/spark/run.sh preflight
```

后续所有命令必须使用同一个 `SPARK_DATA_ROOT`。

## 7. 当前不应执行的部分

`run_full.sh` 目前故意失败，因为原方案要求先通过真实 paper-MVP B0/B1，之后
依次实现和验证 NN1、NN2，再接 ADS、DOGe，最后才做 Rewrite、三随机种子和
完整消融。当前仓库没有把这些未完成阶段包装成“已跑通”。

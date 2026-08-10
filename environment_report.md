# Environment Report

Generated: 2026-08-10 (Asia/Shanghai)

## Hardware and operating system

- OS: Microsoft Windows 10 Pro 10.0.19045
- CPU: Intel Core i5-9300H, 4 cores / 8 logical processors
- RAM: 15.84 GB total; 5.93 GB free at inspection
- GPU: NVIDIA GeForce GTX 1650, 4096 MiB VRAM (3832 MiB free)
- NVIDIA driver: 592.82; reported CUDA compatibility: 13.1
- Workspace disk E: 487.83 GB free

## Software and connectivity

- Python: 3.12.4 at `D:\\Anaconda\\python.exe`
- Git: 2.49.0.windows.1
- Initial ML environment: PyTorch, Transformers, Datasets and PEFT absent
- Connectivity checks: GitHub 200, Hugging Face 200, hf-mirror 200
- DOCX visual QA limitation: LibreOffice/soffice is not installed. The source
  document was checked structurally (205 paragraphs and all 7 tables), but not
  rendered page-by-page.

## Resource decision

The paper-faithful 7B teacher plus 1.5B Student LoRA pipeline is not reliable
on 4 GB VRAM and 16 GB system RAM. The requested configuration is preserved in
`configs/paper_faithful_mvp.yaml`; it was not silently changed.

This host will execute `configs/resource_adapted_smoke.yaml` first:

- clean traces: deterministic GSM8K reference reasoning (offline proxy)
- Student: Qwen2.5-0.5B-Instruct
- train/test: 16 train examples and 8 disjoint official test examples
- seed: 42
- objective: validate B0 Base and B1 Clean-KD, exact-answer evaluation,
  logging, checkpointing and the unified results schema

Metrics from this profile are engineering smoke-test evidence only, not support
for the research hypotheses.

## Installed experiment runtime

- Virtual environment: `.venv`
- PyTorch: 2.7.1+cu126; CUDA available; GTX 1650 FP16 matmul passed
- Transformers: 4.53.2
- Datasets: 4.0.0
- PEFT: 0.16.0
- Accelerate: 1.8.1
- `pip check`: no broken requirements
- Tests: 6 passed (latest source-level suite; Spark hardware run pending)
- Hugging Face Xet transfer: `hf_xet==1.6.0`

## DGX Spark target (declared by user; not inspected from this host)

- Target runtime: ARM64 DGX OS + NVIDIA Container Runtime
- Container baseline: `nvcr.io/nvidia/pytorch:25.11-py3`
- Persistent cache: `.spark-data/huggingface/` or explicit `SPARK_DATA_ROOT`
- Preflight gate: ARM64, CUDA visible in container, BF16, at least 80 GiB free
- Intended workload: sequential 7B Teacher trace generation, then 1.5B Student
  B0/B1 LoRA/SFT; the Teacher and Student are not resident together
- Actual driver, CUDA, disk and GPU values will be recorded by
  `python -m tools.preflight_spark` when run on the Spark

# Status

## Completed

- Parsed the complete execution plan, including 205 paragraphs and 7 tables.
- Detected GPU, RAM, disk, Python and network access.
- Selected an explicit resource-adapted smoke profile while preserving the
  paper-faithful MVP and Full configurations.
- Scaffolded a resumable, YAML-driven B0/B1 harness and unified results schema.
- Installed and verified an isolated CUDA ML environment; `pip check` passes.
- Passed unit tests, configuration validation and data preflight.
- Completed valid run `resource_adapted_smoke-s42-20260810T040742Z`:
  B0=0.000, B1=0.375, gain=+0.375 on 8 test examples.
- Saved predictions, LoRA adapter, JSONL logs, unified results, audit JSON,
  PNG/SVG plot and plot-source CSV.
- Marked two GPU-overlapped diagnostic runs `invalid_concurrent_run` and added
  a Harness process lock.

## Current gate

B0/B1 engineering smoke gate: PASS. The metric is explicitly not a paper-scale
result because it uses a 0.5B Student, reference-answer trace proxy, 16 train
examples, 8 test examples and one seed.

DGX Spark paper-MVP implementation gate: READY FOR HARDWARE VALIDATION. The
repository now contains an ARM64 NGC container, Spark preflight, resumable real
7B Teacher trace generation, 1.5B Student B0/B1 execution, and a CNB CPU gate.
It has not yet been executed on the user's physical DGX Spark, so no
paper-faithful metric is claimed.

## Not yet claimed

- NN1 proxy-utility experiment
- NN2 mastery calibration
- ADS, DOGe or Trace Rewriting reproduction
- three-seed paper-scale conclusions or the seven final figures

These stages require B0/B1 to pass first and substantially more compute for a
paper-faithful run.

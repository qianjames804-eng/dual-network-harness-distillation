$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv. Create it and install requirements.lock first."
}
& $Python -m src.harness --config (Join-Path $ProjectRoot "configs\resource_adapted_smoke.yaml")
& $Python -m src.analysis.audit_results --results (Join-Path $ProjectRoot "results.csv") --output (Join-Path $ProjectRoot "outputs\metrics\latest_smoke_audit.json")
& $Python -m src.analysis.plot_smoke --results (Join-Path $ProjectRoot "results.csv") --output-dir (Join-Path $ProjectRoot "outputs\figures")

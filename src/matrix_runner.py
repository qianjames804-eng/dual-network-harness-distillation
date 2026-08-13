"""Auditable full-matrix dispatcher.

It writes every requested job manifest before execution.  A formal executor is
intentionally explicit: users provide the per-job command template so external
ADS/DOGe/rewrite implementations and their commit hashes cannot be hidden.
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from src.experiment_spec import ablation_jobs, full_jobs
from src.data.splits import assert_manifest_disjoint

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--suite",choices=("full","ablation"),required=True); p.add_argument("--manifest",type=Path,default=Path("outputs/manifests/jobs.json")); p.add_argument("--split-manifest",type=Path); p.add_argument("--execute-template"); p.add_argument("--execute",action="store_true"); a=p.parse_args()
    rows=(full_jobs if a.suite=="full" else ablation_jobs)(["gsm8k","math"],[42,43,44])
    a.manifest.parent.mkdir(parents=True,exist_ok=True); a.manifest.write_text(json.dumps([j.__dict__ for j in rows],indent=2),encoding="utf-8")
    if not a.execute: print(json.dumps({"status":"dry_run","jobs":len(rows),"manifest":str(a.manifest)})); return
    if not a.execute_template: raise ValueError("--execute requires --execute-template with {dataset}, {seed}, {method}")
    if a.split_manifest is None: raise ValueError("--execute requires --split-manifest; split-intersection gate must pass before any job")
    assert_manifest_disjoint(a.split_manifest)
    for job in rows: subprocess.run(a.execute_template.format(**job.__dict__),shell=True,check=True)
if __name__ == "__main__": main()

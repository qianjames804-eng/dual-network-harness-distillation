"""Adapters for anti-distillation traces; never substitutes synthetic formal data."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
from src.common import append_jsonl
from src.traces.io import read_trace_records
OFFICIAL = {"doge":"https://github.com/UNITES-Lab/DOGe", "rewrite":"https://github.com/xhOwenMa/trace-rewriting"}
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--method",choices=("ads","doge","rewrite"),required=True); p.add_argument("--input",type=Path); p.add_argument("--output",type=Path); p.add_argument("--official-command"); p.add_argument("--smoke-rewrite",action="store_true"); a=p.parse_args()
    if not (a.method=="rewrite" and a.smoke_rewrite):
        if not a.official_command: raise RuntimeError(f"{a.method} requires an official command/checkpoint; no synthetic formal trace is allowed")
        subprocess.run(a.official_command,shell=True,check=True); return
    for row in read_trace_records(a.input):
        row=dict(row); row["response"]="\n".join(reversed([x for x in row["response"].splitlines() if x.strip()])); row["defense"]="rewrite_smoke_only"; append_jsonl(a.output,row)
if __name__ == "__main__": main()

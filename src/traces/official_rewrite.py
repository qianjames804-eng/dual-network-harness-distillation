"""Validate and materialize official Trace Rewriting GSM8K traces.

The source dataset remains external and immutable.  This adapter validates the
official schema, maps `problem` exactly to the pinned GSM8K train split, rejects
test overlap, and records source-file hashes in every produced provenance file.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import date
from pathlib import Path
from datasets import load_dataset, load_from_disk
from src.common import append_jsonl

REQUIRED={"problem","solution","original_trace","rewrite_trace"}
def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def materialize(source: Path, output: Path, *, seed: int, examples: int, cache_dir: str, revision: str) -> dict:
    dataset=load_from_disk(str(source)); missing=REQUIRED-set(dataset.column_names)
    if missing: raise RuntimeError(f"official Trace Rewriting schema missing: {sorted(missing)}")
    gsm=load_dataset("openai/gsm8k","main",revision=revision,cache_dir=cache_dir)
    train={row["question"]:row for row in gsm["train"]}; test=set(gsm["test"]["question"])
    source_questions=set(dataset["problem"])
    if source_questions & test: raise RuntimeError("official trace data overlaps GSM8K final test")
    if not source_questions <= set(train): raise RuntimeError("official trace problem cannot be mapped to GSM8K train")
    selected=dataset.shuffle(seed=seed).select(range(min(examples,len(dataset))))
    for index,row in enumerate(selected):
        append_jsonl(output,{"index":index,"question":row["problem"],"response":row["original_trace"],"rewrite_response":row["rewrite_trace"],"expected":train[row["problem"]]["answer"],"correct":str(row.get("is_original_correct","False"))=="True","source":"official_trace_rewriting","official_row":index})
    hashes={path.name:file_hash(path) for path in source.iterdir() if path.is_file()}
    provenance={"status":"ready","paper":"Protecting Language Models Against Unauthorized Distillation through Trace Rewriting (ACL 2026)","repository":"https://github.com/xhOwenMa/trace-rewriting","commit":"a2afa048f6967badf9d4894c912469547eafd452","dataset":"official GSM8K optimized traces mapped exactly to pinned GSM8K train","source_directory":str(source),"source_sha256":hashes,"schema":sorted(REQUIRED),"rows_source":len(dataset),"rows_selected":len(selected),"seed":seed,"license":"not specified in repository root; use restricted to reproducibility pending licensor clarification","retrieval_date":str(date.today())}
    return provenance
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--provenance",type=Path,required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--examples",type=int,default=512); p.add_argument("--cache-dir",default=".cache/huggingface"); p.add_argument("--revision",required=True); a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True); prov=materialize(a.source,a.output,seed=a.seed,examples=a.examples,cache_dir=a.cache_dir,revision=a.revision); a.provenance.write_text(json.dumps(prov,indent=2)+"\n",encoding="utf-8"); print(json.dumps(prov,indent=2))
if __name__ == "__main__": main()

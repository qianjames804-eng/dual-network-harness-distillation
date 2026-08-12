"""Deterministic, disjoint SFT/NN-calibration partitions with provenance."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from datasets import Dataset

@dataclass(frozen=True)
class StudySplits:
    sft: Dataset; nn1_calibration: Dataset; nn2_calibration: Dataset

def split_training_pool(rows: Dataset, *, seed: int, nn1_count: int, nn2_count: int) -> StudySplits:
    if nn1_count + nn2_count >= len(rows): raise ValueError("calibration pools must leave at least one SFT row")
    shuffled=rows.shuffle(seed=seed)
    nn1=shuffled.select(range(nn1_count)); nn2=shuffled.select(range(nn1_count,nn1_count+nn2_count)); sft=shuffled.select(range(nn1_count+nn2_count,len(rows)))
    sets=[set(part["question"]) for part in (sft,nn1,nn2)]
    if any(a & b for i,a in enumerate(sets) for b in sets[i+1:]): raise RuntimeError("question overlap across NN/SFT splits")
    return StudySplits(sft,nn1,nn2)

def write_split_manifest(path: Path, splits: StudySplits, *, seed: int, dataset_revision: str) -> None:
    def digest(rows: Dataset) -> str: return hashlib.sha256("\n".join(rows["question"]).encode()).hexdigest()
    payload={"seed":seed,"dataset_revision":dataset_revision,"sft":{"count":len(splits.sft),"question_sha256":digest(splits.sft)},"nn1_calibration":{"count":len(splits.nn1_calibration),"question_sha256":digest(splits.nn1_calibration)},"nn2_calibration":{"count":len(splits.nn2_calibration),"question_sha256":digest(splits.nn2_calibration)}}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2),encoding="utf-8")

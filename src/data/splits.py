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

def _item_hashes(rows: Dataset) -> list[str]:
    return sorted(hashlib.sha256(question.strip().encode("utf-8")).hexdigest() for question in rows["question"])

def write_split_manifest(path: Path, splits: StudySplits, *, seed: int, dataset_revision: str, final_test: Dataset | None = None, final_paraphrases: Dataset | None = None, extra_partitions: dict[str, Dataset] | None = None) -> None:
    def entry(rows: Dataset) -> dict:
        hashes=_item_hashes(rows)
        return {"count":len(rows),"question_sha256":hashlib.sha256("\n".join(hashes).encode()).hexdigest(),"item_sha256":hashes}
    # When NN2 is explicitly subdivided, only the two leaf partitions enter
    # the overlap gate.  The parent is metadata, not a separate sample split.
    nn2_entry=(
        {"count":len(splits.nn2_calibration),"subdivided_into":sorted(extra_partitions)}
        if extra_partitions and {"nn2_train","nn2_validation"}.issubset(extra_partitions)
        else entry(splits.nn2_calibration)
    )
    payload={"seed":seed,"dataset_revision":dataset_revision,"sft":entry(splits.sft),"nn1_calibration":entry(splits.nn1_calibration),"nn2_calibration":nn2_entry}
    if final_test is not None: payload["final_test"]=entry(final_test)
    if final_paraphrases is not None: payload["final_test_paraphrases"]=entry(final_paraphrases)
    for name, rows in (extra_partitions or {}).items():
        if name in payload: raise ValueError(f"duplicate manifest partition: {name}")
        payload[name]=entry(rows)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2),encoding="utf-8")

def assert_manifest_disjoint(path: Path) -> None:
    """Runtime hard gate: all present partitions must have zero item overlap."""
    data=json.loads(path.read_text(encoding="utf-8")); names=[name for name,value in data.items() if isinstance(value,dict) and "question_sha256" in value and "item_sha256" in value]
    for i,left in enumerate(names):
        expected=data[left]["question_sha256"]
        actual=hashlib.sha256("\n".join(data[left]["item_sha256"]).encode()).hexdigest()
        if actual != expected: raise RuntimeError(f"split manifest integrity mismatch: {left}")
        for right in names[i+1:]:
            overlap=set(data[left]["item_sha256"]).intersection(data[right]["item_sha256"])
            if overlap: raise RuntimeError(f"data leakage: {left} intersects {right} ({len(overlap)} questions)")

import numpy as np
from src.experiment_spec import ablation_jobs, full_jobs
from src.nn1_data_weight.pipeline import fit_predict, retained_indices, utility_targets
from src.nn2_judgment.pipeline import fit_judge, mastery_targets
from datasets import Dataset
from src.data.splits import assert_manifest_disjoint, split_training_pool, write_split_manifest
from src.traces.official_rewrite import REQUIRED
import pytest

def test_matrix_counts():
    assert len(full_jobs(["gsm8k","math"],[42,43,44])) == 54
    assert len(ablation_jobs(["gsm8k","math"],[42,43,44])) == 24
def test_nn1_weights_and_filter():
    x=np.array([[0.,1.],[1.,0.],[2.,1.],[3.,2.]]); y=utility_targets(np.array([3,3,3,3]),np.array([2,2.5,1,2]))
    r=fit_predict(x,y,epochs=2,lr=.01,seed=1); assert r.weights.shape==(4,) and len(retained_indices(r.weights,.5))==2
def test_nn2_mastery():
    x=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]]); y=mastery_targets(np.array([0,0,1,1]),np.array([0,1,1,1]))
    assert fit_judge(x,y,epochs=2,lr=.01,seed=1).scores.shape==(4,)
def test_disjoint_study_splits(tmp_path):
    rows=Dataset.from_dict({"question":[f"q{i}" for i in range(8)],"answer":["a"]*8})
    splits=split_training_pool(rows,seed=42,nn1_count=2,nn2_count=2)
    assert set(splits.sft["question"]).isdisjoint(splits.nn1_calibration["question"])
    path=tmp_path/"splits.json"; write_split_manifest(path,splits,seed=42,dataset_revision="fixed"); assert path.exists(); assert_manifest_disjoint(path)
    broken=path.read_text().replace('"item_sha256": [', '"item_sha256": ["bad", ', 1); path.write_text(broken)
    with pytest.raises(RuntimeError): assert_manifest_disjoint(path)
def test_official_rewrite_schema_contract():
    assert REQUIRED == {"problem", "solution", "original_trace", "rewrite_trace"}

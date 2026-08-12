import numpy as np
from src.experiment_spec import ablation_jobs, full_jobs
from src.nn1_data_weight.pipeline import fit_predict, retained_indices, utility_targets
from src.nn2_judgment.pipeline import fit_judge, mastery_targets

def test_matrix_counts():
    assert len(full_jobs(["gsm8k","math"],[42,43,44])) == 54
    assert len(ablation_jobs(["gsm8k","math"],[42,43,44])) == 24
def test_nn1_weights_and_filter():
    x=np.array([[0.,1.],[1.,0.],[2.,1.],[3.,2.]]); y=utility_targets(np.array([3,3,3,3]),np.array([2,2.5,1,2]))
    r=fit_predict(x,y,epochs=2,lr=.01,seed=1); assert r.weights.shape==(4,) and len(retained_indices(r.weights,.5))==2
def test_nn2_mastery():
    x=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]]); y=mastery_targets(np.array([0,0,1,1]),np.array([0,1,1,1]))
    assert fit_judge(x,y,epochs=2,lr=.01,seed=1).scores.shape==(4,)

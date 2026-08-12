# NN1 and NN2 label construction

NN1 is the **data weight / filter** network. For trace `i`, its feature vector
is `x_i=[len(q_i), len(t_i), tokens(t_i), duplicate(q_i), correct_T(i),
entropy_T(i), NLL_base(i), embeddings(q_i,t_i)]`; embeddings are optional but
must be frozen. Its proxy-utility label is `u_i=max(L_val(theta)-L_val(theta'_i),0)`,
where `theta'_i` is a **single proxy Student update on trace i** and `L_val` is
cross-entropy on a clean NN1-calibration validation partition. Training uses
the normalized target `y_i=u_i/(max_j u_j+eps)`. This makes a high label mean
"this trace improves clean validation loss", not merely "this trace is long".

The training pool is deterministically shuffled by seed into mutually disjoint
SFT, NN1-calibration and NN2-calibration pools; each split writes a question
SHA-256 manifest. Neither calibration pool may contain a final-test question.
The proxy's validation partition is additionally disjoint from the trace being
updated. The final held-out benchmark is never used for feature fitting,
normalization statistics, utility labels, thresholds or early stopping.
NN1 outputs `w_data ∈ [0,1]`, applied either as the existing weighted loss or
as a top-k retention selection under a fixed training-token budget.

NN2 is the **mastery Judgment** network. For evaluation item `j`, its feature
vector is `z_j=[confidence_j, NLL_j, length_j, consistency_j, ||A||, ||B||,
||BA||, ||grad||, adapter_on_off_delta]`. Its mastery target is
`m_j=(C(q_j)+C(para(q_j)))/2`, where `C` is exact answer correctness and
`para(q_j)` is an equivalent held-out paraphrase.  Thus `m_j ∈ {0, .5, 1}` and
does not reward merely matching one response style. Paraphrases and their
labels are evaluation-only; they are never used as SFT input, NN1 proxy data,
or NN2 feature-normalization training data. Report
AUROC/F1, Brier/ECE and Spearman correlation with this actual mastery target.

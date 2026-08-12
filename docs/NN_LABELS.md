# NN1 and NN2 label construction

NN1 is the **data weight / filter** network.  Its inputs are per-trace
Question/Answer lengths, response-token count, duplicate rate, Teacher answer
correctness, optional Teacher token entropy and Base Student NLL; production
embeddings may be appended.  Labels are proxy distillation utility on a
calibration split: `max(clean_validation_loss_before - loss_after_one_proxy_update, 0)`.
The proxy validation split and the final test split are disjoint from SFT data.
NN1 outputs `w_data ∈ [0,1]`, applied either as the existing weighted loss or
as a top-k retention selection under a fixed training-token budget.

NN2 is the **mastery Judgment** network.  Its inputs are Student-output
confidence/NLL, response length, original/paraphrase consistency and LoRA
adapter statistics (A/B/BA norm, gradient norm and adapter-on/off deltas).
Its label is mean correctness on the original held-out item and a held-out
paraphrase: `(original_correct + paraphrase_correct) / 2`.  Paraphrases and
their labels are evaluation-only; they are never used as SFT input.  Report
AUROC/F1, Brier/ECE and Spearman correlation with this actual mastery target.

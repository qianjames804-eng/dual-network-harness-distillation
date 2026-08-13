"""Real five-arm Trace Rewriting experiment scheduler.

No synthetic utility, paraphrase, or mastery target is accepted.  The tiny
smoke uses fewer *official* traces and real model calls; the formal config uses
the same code path on 512 official records.
"""
from __future__ import annotations
import argparse, json, math, time, traceback
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.common import append_jsonl, append_result, load_yaml, set_seed, utc_run_id
from src.data.gsm8k import GSM8KSFTDataset, prompt_text
from src.data.splits import StudySplits, split_training_pool, write_split_manifest, assert_manifest_disjoint
from src.eval.gsm8k_eval import evaluate
from src.eval.answers import extract_final_answer
from src.lora_train.train import train_lora
from src.modeling import load_causal_lm
from src.nn1_data_weight.pipeline import fit_predict, trace_features, utility_targets
from src.nn2_judgment.pipeline import fit_judge, mastery_targets
from src.nn2_judgment.paraphrase import generate_paraphrase
from src.traces.io import read_trace_records

METHODS=("B0-Base","B1-Clean-KD","B4-Rewrite-KD","Ours-Clean","Ours-Rewrite")
def _norm(values):
    x=np.asarray(values,dtype=np.float32); return (x-x.mean(0))/(x.std(0)+1e-6)
def _normalise_train_valid(train, valid):
    train=np.asarray(train,dtype=np.float32); valid=np.asarray(valid,dtype=np.float32)
    mean=train.mean(0); std=train.std(0)+1e-6
    return (train-mean)/std, (valid-mean)/std
def _load_rows(cfg):
    records=read_trace_records(cfg["traces"]["path"])
    count=int(cfg["dataset"]["train_examples"])
    if len(records)<count: raise RuntimeError("official trace file has fewer rows than configured")
    rows=Dataset.from_list([{"question":r["question"],"answer":r["expected"],"original_trace":r["response"],"rewrite_trace":r["rewrite_response"]} for r in records[:count]])
    gsm=load_dataset("openai/gsm8k","main",revision=cfg["dataset"]["revision"],cache_dir=cfg["dataset"]["cache_dir"])
    final=gsm["test"].shuffle(seed=int(cfg["seed"])).select(range(int(cfg["dataset"]["eval_examples"])))
    if set(rows["question"]) & set(final["question"]): raise RuntimeError("official traces overlap final evaluation")
    return rows,final
def _lora(model,cfg):
    t=cfg["training"]; model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache=False
    return get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=int(t["lora_r"]),lora_alpha=int(t["lora_alpha"]),lora_dropout=float(t["lora_dropout"]),target_modules=list(t["target_modules"]),bias="none"))
def _loss(model,tok,rows,responses,max_length):
    losses=[]; device=next(model.parameters()).device
    for row in rows:
        p=tok(prompt_text(tok,row["question"]),add_special_tokens=False)["input_ids"]; a=tok(responses[row["question"]]+tok.eos_token,add_special_tokens=False)["input_ids"]
        ids=(p+a)[:max_length]; labels=[-100]*min(len(p),len(ids))+ids[min(len(p),len(ids)):]
        out=model(input_ids=torch.tensor([ids],device=device),attention_mask=torch.ones((1,len(ids)),device=device,dtype=torch.long),labels=torch.tensor([labels],device=device)); losses.append(float(out.loss.detach()))
    return float(np.mean(losses))
def _proxy_weights(cfg, split: StudySplits, trace_key: str, run: Path):
    """One actual LoRA proxy update per NN1 calibration trace."""
    proxy_cfg={"id":cfg["proxy"]["id"],"revision":cfg["proxy"]["revision"],"dtype":cfg["proxy"]["dtype"]}; base,tok=load_causal_lm(proxy_cfg,cfg["dataset"]["cache_dir"]); model=_lora(base,cfg)
    cand=list(split.nn1_calibration); pivot=max(1,len(cand)//2); updates=cand[:pivot]; validation=cand[pivot:]
    if not validation: raise RuntimeError("NN1 calibration needs disjoint proxy validation questions")
    answer={r["question"]:r[trace_key] for r in cand}; before=_loss(model,tok,validation,answer,int(cfg["training"]["max_length"]))
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=float(cfg["proxy"]["learning_rate"]))
    after=[]
    snapshot={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    for row in updates:
        opt.zero_grad(); loss=_loss(model,tok,[row],answer,int(cfg["training"]["max_length"])); torch.tensor(loss,device=next(model.parameters()).device) # loss was measured under inference-style scalar
        # Recompute differentiable candidate loss.
        p=tok(prompt_text(tok,row["question"]),add_special_tokens=False)["input_ids"]; a=tok(answer[row["question"]]+tok.eos_token,add_special_tokens=False)["input_ids"]; ids=(p+a)[:int(cfg["training"]["max_length"])]; labels=[-100]*min(len(p),len(ids))+ids[min(len(p),len(ids)):]; device=next(model.parameters()).device
        out=model(input_ids=torch.tensor([ids],device=device),attention_mask=torch.ones((1,len(ids)),device=device,dtype=torch.long),labels=torch.tensor([labels],device=device)); out.loss.backward(); opt.step(); after.append(_loss(model,tok,validation,answer,int(cfg["training"]["max_length"])))
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in snapshot: p.copy_(snapshot[n])
        opt.state.clear()
    utility=utility_targets(np.full(len(after),before),np.asarray(after)); train_records=[dict(r,response=r[trace_key]) for r in updates]; sft_records=[dict(r,response=r[trace_key]) for r in split.sft]
    all_records=train_records+sft_records; raw=trace_features(all_records); fitted=fit_predict(raw[:len(train_records)],utility,epochs=int(cfg["nn1"]["epochs"]),lr=float(cfg["nn1"]["learning_rate"]),seed=int(cfg["seed"]),hidden_dim=int(cfg["nn1"]["hidden_dim"]),predict_features=raw[len(train_records):])
    weights=fitted.weights.tolist()
    mapping={r["question"]:w for r,w in zip(split.sft,weights)}; append_jsonl(run/"nn1_weights.jsonl",{"calibration_before_loss":before,"utility":utility.tolist(),"calibration_weights":fitted.weights.tolist(),"sft_weights":mapping,"spearman":fitted.spearman,"auroc":fitted.auroc})
    del model,base; torch.cuda.empty_cache(); return mapping,fitted
def _response_features(model,tok,question,response,max_length):
    """Actual output confidence/NLL, LoRA gradient norm, and adapter-on/off delta."""
    device=next(model.parameters()).device; prompt=tok(prompt_text(tok,question),add_special_tokens=False)["input_ids"]; answer=tok(response+tok.eos_token,add_special_tokens=False)["input_ids"]; ids=(prompt+answer)[:max_length]; cut=min(len(prompt),len(ids)); labels=[-100]*cut+ids[cut:]
    x=torch.tensor([ids],device=device); y=torch.tensor([labels],device=device); mask=y.ne(-100)
    model.eval()
    with torch.inference_mode():
        logits=model(input_ids=x,attention_mask=torch.ones_like(x)).logits[:,:-1]; shifted=y[:,1:]; active=shifted.ne(-100); lp=torch.log_softmax(logits,dim=-1); nll=float((-lp.gather(-1,shifted.clamp_min(0).unsqueeze(-1)).squeeze(-1)[active]).mean()) if active.any() else 0.; confidence=float(logits.softmax(-1).max(-1).values[active].mean()) if active.any() else 0.
        with model.disable_adapter():
            off=float(model(input_ids=x,attention_mask=torch.ones_like(x),labels=y).loss)
    model.train(); model.zero_grad(set_to_none=True); loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y).loss; loss.backward(); grad=float(torch.sqrt(sum((p.grad.detach()**2).sum() for n,p in model.named_parameters() if "lora_" in n and p.grad is not None))); model.zero_grad(set_to_none=True); model.eval()
    a=[float(p.detach().norm()) for n,p in model.named_parameters() if "lora_A" in n]; b=[float(p.detach().norm()) for n,p in model.named_parameters() if "lora_B" in n]
    return [confidence,nll,float(np.mean(a) if a else 0),float(np.mean(b) if b else 0),float(np.mean(a)*np.mean(b) if a and b else 0),grad,off-nll]
def _nn2(cfg,model,tok,nn2_train,nn2_valid,run):
    base,tbase=load_causal_lm(cfg["paraphraser"],cfg["dataset"]["cache_dir"])
    predictions=[]; features=[]; original=[]; paraphrase=[]; partitions=[]
    for partition, rows in (("train",list(nn2_train)),("validation",list(nn2_valid))):
      for row in rows:
        para, attempts=generate_paraphrase(base,tbase,row["question"],max_new_tokens=int(cfg["nn2"]["paraphrase_max_new_tokens"])); pair=Dataset.from_list([row,{"question":para,"answer":row["answer"]}]); result=evaluate(model,tok,pair,cfg["evaluation"],run/"nn2_tmp.jsonl")
        records=[json.loads(x) for x in (run/"nn2_tmp.jsonl").read_text().splitlines()]; o,p=records[0],records[1]; original.append(o["correct"]); paraphrase.append(p["correct"]); consistency=float(o["predicted"]==p["predicted"]); raw_features=_response_features(model,tok,row["question"],o["response"],int(cfg["training"]["max_length"]))+_response_features(model,tok,para,p["response"],int(cfg["training"]["max_length"]))+[consistency]; features.append(raw_features); partitions.append(partition); predictions.append({"partition":partition,"question":row["question"],"paraphrase":para,"paraphrase_attempts":attempts,"original":o,"paraphrase_result":p,"raw_features":raw_features})
    target=mastery_targets(np.asarray(original),np.asarray(paraphrase)); cut=len(nn2_train); train_x,valid_x=_normalise_train_valid(features[:cut],features[cut:]); result=fit_judge(train_x,target[:cut],epochs=int(cfg["nn2"]["epochs"]),lr=float(cfg["nn2"]["learning_rate"]),seed=int(cfg["seed"]),hidden_dim=int(cfg["nn2"]["hidden_dim"]),validation_features=valid_x,validation_targets=target[cut:]); (run/"nn2_predictions.json").write_text(json.dumps({"feature_names":["original_confidence","original_nll","original_lora_a_norm","original_lora_b_norm","original_lora_ba_norm","original_gradient_norm","original_adapter_delta","paraphrase_confidence","paraphrase_nll","paraphrase_lora_a_norm","paraphrase_lora_b_norm","paraphrase_lora_ba_norm","paraphrase_gradient_norm","paraphrase_adapter_delta","answer_consistency"],"fit_partition":"train","metric_partition":"validation","items":predictions,"validation_scores":result.scores.tolist(),"targets":target.tolist()},ensure_ascii=False,indent=2)); del base; torch.cuda.empty_cache(); return result
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--smoke",action="store_true"); p.add_argument("--methods",nargs="+",default=list(METHODS)); a=p.parse_args(); cfg=load_yaml(a.config); set_seed(int(cfg["seed"])); rows,final=_load_rows(cfg)
    if a.smoke:
        rows=rows.select(range(int(cfg["smoke"]["train_examples"])))
        final=final.select(range(int(cfg["smoke"]["eval_examples"])))
    splits=split_training_pool(rows,seed=int(cfg["seed"]),nn1_count=int(cfg["nn1"]["calibration_examples"]),nn2_count=int(cfg["nn2"]["calibration_examples"])); nn2_pool=splits.nn2_calibration.shuffle(seed=int(cfg["seed"])+1); nn2_cut=len(nn2_pool)//2
    if nn2_cut < 1 or len(nn2_pool)-nn2_cut < 1: raise RuntimeError("NN2 needs disjoint train and validation calibration questions")
    nn2_train,nn2_valid=nn2_pool.select(range(nn2_cut)),nn2_pool.select(range(nn2_cut,len(nn2_pool))); manifest=Path(cfg["outputs"]["root"])/"manifests"/f"rewrite_s{cfg['seed']}_{'smoke' if a.smoke else 'formal'}.json"; write_split_manifest(manifest,splits,seed=int(cfg["seed"]),dataset_revision=cfg["dataset"]["revision"],final_test=final,extra_partitions={"nn2_train":nn2_train,"nn2_validation":nn2_valid}); assert_manifest_disjoint(manifest)
    for method in a.methods:
        run=Path(cfg["outputs"]["root"])/utc_run_id(method.replace("-","_").lower(),int(cfg["seed"])); run.mkdir(parents=True); (run/"config.json").write_text(json.dumps(cfg,indent=2)); append_jsonl(run/"run.jsonl",{"event":"isolation_passed","manifest":str(manifest),"method":method})
        base,tok=load_causal_lm(cfg["student"],cfg["dataset"]["cache_dir"]); b0=evaluate(base,tok,final,cfg["evaluation"],run/"b0_predictions.jsonl"); nn1=nn2=None
        if method!="B0-Base":
            key="rewrite_trace" if "Rewrite" in method else "original_trace"; weights={}
            if method.startswith("Ours"): weights,nn1=_proxy_weights(cfg,splits,key,run)
            train=GSM8KSFTDataset(splits.sft,tok,int(cfg["training"]["max_length"]),responses_by_question={r["question"]:r[key] for r in splits.sft},sample_weights=weights); base,metrics=train_lora(base,tok,train,cfg["training"],run/"checkpoints",int(cfg["seed"])); result=evaluate(base,tok,final,cfg["evaluation"],run/"predictions.jsonl");
            if method.startswith("Ours"):
                try: nn2=_nn2(cfg,base,tok,nn2_train,nn2_valid,run)
                except Exception:
                    (run/"nn2_failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
                    raise
        else: result=b0; metrics={"global_step":0}; train=type("T",(),{"train_tokens":0})()
        summary={"method":method,"accuracy":result["accuracy"],"exact_match":result["accuracy"],"base_accuracy":b0["accuracy"],"delta_vs_b0":result["accuracy"]-b0["accuracy"],"train_tokens":train.train_tokens,"train_metrics":metrics,"nn1_spearman":None if nn1 is None else nn1.spearman,"nn1_auroc":None if nn1 is None else nn1.auroc,"nn2_auroc":None if nn2 is None else nn2.auroc,"nn2_f1":None if nn2 is None else nn2.f1,"nn2_brier":None if nn2 is None else nn2.brier,"nn2_ece":None if nn2 is None else nn2.ece,"nn2_spearman":None if nn2 is None else nn2.spearman}; (run/"metrics.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); del base; torch.cuda.empty_cache()
if __name__ == "__main__": main()

"""Qwen paraphrase gate with deterministic numeric placeholders.

This is an evaluation-only smoke tool.  It never produces SFT data.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Match money, fractions, decimals, percentages, and plain integer expressions
# as one protected token; ordering is intentional (most structured first).
NUMBER=re.compile(r"\$\s*[-+]?\d+(?:\.\d+)?(?:%|/\d+(?:\.\d+)?)?|[-+]?\d+(?:\.\d+)?\s*/\s*[-+]?\d+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?%|[-+]?\d+(?:\.\d+)?")
PH=re.compile(r"<NUM_\d+>")
FORBIDDEN=("therefore","solution","answer is","calculate:","=")
STRATEGIES=(
    "Use different wording and a different sentence structure while retaining every fact.",
    "Reorder the explanatory sentences naturally, then restate the question in fresh wording; retain every fact.",
    "Use concise but meaning-preserving synonyms and grammar; retain every fact and condition.",
)
def protect(text: str) -> tuple[str,list[str]]:
    values=[]
    def replace(match): values.append(match.group(0)); return f"<NUM_{len(values)-1}>"
    return NUMBER.sub(replace,text),values
def restore(text: str, values: list[str]) -> tuple[str,str|None]:
    found=PH.findall(text); expected=[f"<NUM_{i}>" for i in range(len(values))]
    if found!=expected: return text,"placeholder_error"
    for i,value in enumerate(values): text=text.replace(f"<NUM_{i}>",value)
    return text,None
def validate(original: str, output: str, values: list[str]) -> tuple[str|None,str]:
    if not output.startswith("QUESTION:"): return "missing_question_prefix",output
    text=output.removeprefix("QUESTION:").strip(); text,error=restore(text,values)
    if error: return error,text
    if not text: return "empty_question",text
    normalize=lambda value: re.sub(r"\s+|[^\w<>]", "", value).lower()
    if normalize(text)==normalize(original): return "unchanged_question",text
    if not text.rstrip().endswith("?"): return "incomplete_question",text
    # This second check defends against malformed restoration / numeric injection.
    if NUMBER.findall(text)!=values: return "numbers_changed",text
    if any(word in text.lower() for word in FORBIDDEN): return "contains_solution_or_equation",text
    return None,text
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--count",type=int,default=20); p.add_argument("--seed",type=int,default=42); p.add_argument("--candidates",type=int,default=3); p.add_argument("--cache-dir",default=".cache/huggingface"); a=p.parse_args()
    ds=load_dataset("openai/gsm8k","main",cache_dir=a.cache_dir)["train"].shuffle(seed=a.seed).select(range(a.count)); tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to("cuda").eval(); records=[]
    for row in ds:
        protected,values=protect(row["question"]); attempts=[]; accepted=None
        for candidate in range(a.candidates):
            strategy=STRATEGIES[candidate % len(STRATEGIES)]
            prompt=(
                "You are a question paraphraser, not a solver. Produce a genuinely rewritten, "
                "complete version of ORIGINAL. The original wording itself is invalid output. "
                f"{strategy} Output exactly one line beginning `QUESTION:` followed only by the rewritten question. "
                "Do not give an answer, a calculation, an explanation, or any reasoning. "
                "The tokens <NUM_0>, <NUM_1>, etc. are immutable atomic placeholders: copy every one "
                "exactly once, in the identical left-to-right order; never add, remove, rename, duplicate, "
                "or move any placeholder. End the question with ?.\n"
                "ORIGINAL: "+protected
            )
            rendered=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            encoded=tok(rendered,return_tensors="pt").to("cuda")
            # Candidate strategy is deterministic; decoding is greedy and reproducible.
            with torch.inference_mode(): out=model.generate(**encoded,max_new_tokens=192,do_sample=False,repetition_penalty=1.05,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            raw=tok.decode(out[0,encoded.input_ids.shape[1]:],skip_special_tokens=True).strip(); reason,text=validate(row["question"],raw,values); attempts.append({"candidate":candidate+1,"raw":raw,"restored":text,"passed":reason is None,"failure_reason":reason})
            if reason is None: accepted=text; break
        records.append({"question":row["question"],"protected_question":protected,"numeric_values":values,"attempts":attempts,"passed":accepted is not None,"paraphrase":accepted})
    first=sum(bool(r["attempts"] and r["attempts"][0]["passed"]) for r in records); passed=sum(r["passed"] for r in records); reasons={}
    for row in records:
        if not row["passed"]:
            for attempt in row["attempts"]: reasons[attempt["failure_reason"]]=reasons.get(attempt["failure_reason"],0)+1
    for reason in ("numbers_changed","placeholder_error","unchanged_question"):
        reasons.setdefault(reason,0)
    payload={"model":str(a.model),"count":a.count,"candidates":a.candidates,"first_pass":first,"final_passed":passed,"failure_counts":reasons,"records":records}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)); print(json.dumps({k:payload[k] for k in ("count","candidates","first_pass","final_passed","failure_counts","output")}))
if __name__=="__main__": main()

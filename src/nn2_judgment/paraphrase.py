"""Strict, deterministic numeric-placeholder paraphrasing for NN2 only."""
from __future__ import annotations
import re
import torch

NUMBER = re.compile(r"\$\s*[-+]?\d+(?:\.\d+)?(?:%|/\d+(?:\.\d+)?)?|[-+]?\d+(?:\.\d+)?\s*/\s*[-+]?\d+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?%|[-+]?\d+(?:\.\d+)?")
PH = re.compile(r"<NUM_\d+>")
FORBIDDEN=("therefore", "solution", "answer is", "calculate:", "=")
STRATEGIES=("Use different wording and sentence structure while retaining every fact.", "Naturally reorder explanatory sentences and restate the question in fresh wording; retain every fact.", "Use concise meaning-preserving synonyms and grammar; retain every fact and condition.")

def protect_numbers(question: str) -> tuple[str, list[str]]:
    values: list[str] = []
    def repl(match: re.Match) -> str:
        values.append(match.group(0)); return f"<NUM_{len(values)-1}>"
    return NUMBER.sub(repl, question), values

def validate_paraphrase(original: str, raw: str, values: list[str]) -> tuple[str | None, str]:
    if not raw.startswith("QUESTION:"): return "missing_question_prefix", raw
    text=raw.removeprefix("QUESTION:").strip(); expected=[f"<NUM_{i}>" for i in range(len(values))]
    if PH.findall(text) != expected: return "placeholder_error", text
    for i,value in enumerate(values): text=text.replace(f"<NUM_{i}>", value)
    normal=lambda x: re.sub(r"\s+|[^\w<>]", "", x).lower()
    if not text: return "empty_question", text
    if normal(text)==normal(original): return "unchanged_question", text
    if not text.endswith("?"): return "incomplete_question", text
    if NUMBER.findall(text) != values: return "numbers_changed", text
    if any(word in text.lower() for word in FORBIDDEN): return "contains_solution_or_equation", text
    return None, text

def generate_paraphrase(model, tokenizer, question: str, *, max_new_tokens: int, candidates: int=3) -> tuple[str, list[dict]]:
    protected, values=protect_numbers(question); device=next(model.parameters()).device; attempts=[]
    for idx in range(candidates):
        prompt=("You are a question paraphraser, not a solver. Produce a genuinely rewritten complete version of ORIGINAL; the original wording itself is invalid output. " + STRATEGIES[idx % len(STRATEGIES)] + " Output exactly one line beginning `QUESTION:` followed only by the rewritten question. Do not give an answer, calculation, explanation, or reasoning. The tokens <NUM_0>, <NUM_1>, etc. are immutable: copy every one exactly once in identical left-to-right order; never add, remove, rename, duplicate, or move placeholders. End with ?.\nORIGINAL: " + protected)
        rendered=tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded=tokenizer(rendered,return_tensors="pt").to(device)
        with torch.inference_mode():
            output=model.generate(**encoded,max_new_tokens=max_new_tokens,do_sample=False,repetition_penalty=1.05,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        raw=tokenizer.decode(output[0,encoded.input_ids.shape[1]:],skip_special_tokens=True).strip(); reason,text=validate_paraphrase(question,raw,values)
        attempts.append({"candidate":idx+1,"raw":raw,"restored":text,"passed":reason is None,"failure_reason":reason})
        if reason is None: return text, attempts
    raise RuntimeError("NN2 paraphrase protocol failure: " + "; ".join(str(x["failure_reason"]) for x in attempts))

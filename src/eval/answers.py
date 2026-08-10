from __future__ import annotations

import re


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    compact = value.replace(",", "").strip()
    try:
        number = float(compact)
    except ValueError:
        return compact
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def extract_final_answer(text: str) -> str | None:
    marker = re.findall(r"####\s*(" + NUMBER_RE.pattern + r")", text)
    if marker:
        return normalize_number(marker[-1])
    boxed = re.findall(r"\\boxed\{\s*(" + NUMBER_RE.pattern + r")\s*\}", text)
    if boxed:
        return normalize_number(boxed[-1])
    numbers = NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else None

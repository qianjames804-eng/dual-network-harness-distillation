from src.eval.answers import extract_final_answer


def test_extracts_hash_marker() -> None:
    assert extract_final_answer("work\n#### 1,234") == "1234"


def test_extracts_boxed_answer() -> None:
    assert extract_final_answer(r"Therefore \\boxed{42}.") == "42"


def test_falls_back_to_last_number() -> None:
    assert extract_final_answer("First 3, finally 7.5") == "7.5"

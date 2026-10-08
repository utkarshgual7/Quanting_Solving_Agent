from guardrails import is_math_query


def test_keyword_query_accepted():
    assert is_math_query("Solve x^2 = 4")


def test_compact_arithmetic_accepted():
    assert is_math_query("2+2")


def test_latex_accepted():
    assert is_math_query(r"what is \frac{1}{2} of 10")


def test_non_math_rejected():
    assert not is_math_query("Tell me a story about dragons")

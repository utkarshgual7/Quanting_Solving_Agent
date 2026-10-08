from answer_check import is_correct


def test_expression_match():
    assert is_correct("3x^2 + 4x - 5", "f'(x) = 3x^2 + 4x - 5")


def test_spacing_insensitive():
    assert is_correct("x = 1/2", "x=1/2, x=-3")


def test_wrong_answer_rejected():
    assert not is_correct("7", "the answer is 8")


def test_numeric_answer_followed_by_full_stop():
    assert is_correct("5", "Therefore x = 5.")

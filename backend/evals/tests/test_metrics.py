from math import comb

import pytest

from evals.metrics import cohen_kappa, mean_pass_at_k, pass_at_k


def test_pass_at_1_is_fraction_correct():
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)


def test_hand_checked_pass_at_5():
    # 1 - C(7,5)/C(10,5) = 1 - 21/252 = 11/12
    assert pass_at_k(10, 3, 5) == pytest.approx(11 / 12)


def test_edges():
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 3) == 1.0
    assert pass_at_k(5, 1, 5) == 1.0  # k = n: the one correct sample is always drawn
    assert pass_at_k(10, 8, 3) == 1.0  # only 2 failures, can't fill 3 slots with them


def test_stable_form_matches_binomial_form():
    for n in range(1, 25):
        for c in range(n + 1):
            for k in range(1, n + 1):
                exact = 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
                assert pass_at_k(n, c, k) == pytest.approx(exact)


def test_large_n_does_not_overflow():
    assert 0.0 < pass_at_k(2000, 3, 100) < 1.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, 2, 6)


def test_mean_over_tasks():
    # task A: 1/2 correct, task B: 0/2 correct -> pass@1 = (0.5 + 0) / 2
    assert mean_pass_at_k([(2, 1), (2, 0)], 1) == pytest.approx(0.25)


def test_kappa_textbook_example():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15
    # p_o = 35/50 = 0.7, p_e = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = (0.7-0.5)/(1-0.5) = 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_small_hand_example():
    # p_o = 3/4; rater a: 1,1,2,2  rater b: 1,2,2,2 -> p_e = .5*.25 + .5*.75 = .5 -> 0.5
    assert cohen_kappa([1, 1, 2, 2], [1, 2, 2, 2]) == pytest.approx(0.5)


def test_kappa_perfect_and_opposite():
    assert cohen_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert cohen_kappa([1, 2], [2, 1]) == pytest.approx(-1.0)
    assert cohen_kappa([3, 3], [3, 3]) == 1.0


def test_quadratic_weighting_rewards_near_misses():
    human = [1, 2, 3, 4, 5]
    near = [2, 2, 3, 4, 4]  # off by one twice
    far = [5, 2, 3, 4, 1]   # off by four twice
    scale = [1, 2, 3, 4, 5]
    assert cohen_kappa(human, near, scale, "quadratic") > cohen_kappa(human, far, scale, "quadratic")
    # hand check on a 2-point scale: weights reduce to plain kappa
    assert cohen_kappa([1, 1, 2, 2], [1, 2, 2, 2], [1, 2], "quadratic") == pytest.approx(0.5)


def test_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])

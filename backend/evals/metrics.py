"""pass@k metrics.

pass@k = probability that at least one of k samples drawn from a model solves the
task. Estimating it by literally drawing k samples is high-variance, so Chen et al.
2021 ("Evaluating Large Language Models Trained on Code", the Codex paper) draw
n >= k samples, count the c correct ones, and use the unbiased estimator

    pass@k = 1 - C(n - c, k) / C(n, k)

i.e. one minus the chance that a random size-k subset of the n samples contains no
correct sample. The binomials overflow for large n, so we use the paper's
numerically stable product form:

    C(n - c, k) / C(n, k) = prod_{i = n - c + 1}^{n} (1 - k / i)
"""
from math import prod
from statistics import mean


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one task with n samples, c of them correct."""
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:  # every size-k subset must contain a correct sample
        return 1.0
    return 1.0 - prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def mean_pass_at_k(counts: list[tuple[int, int]], k: int) -> float:
    """Benchmark-level pass@k: average the per-task estimate over (n, c) pairs."""
    return mean(pass_at_k(n, c, k) for n, c in counts)

"""
wci.py

Workload Complexity Index for the AdaHDP prototype.

"""

from __future__ import annotations

import numpy as np


###############################################################################
# Linear algebra
###############################################################################


def singular_values(workload: np.ndarray) -> np.ndarray:
    """
    Return the singular values of a workload matrix.
    """

    return np.linalg.svd(workload, compute_uv=False)


def entropy(sigma: np.ndarray, tolerance: float = 1e-12) -> float:
    """
    Shannon entropy of the normalized singular values.
    """

    sigma = sigma[
        sigma > tolerance
    ]

    if sigma.size == 0:

        return 0.0

    p = sigma / sigma.sum()

    return float(
        -np.sum(
            p * np.log(p)
        )
    )


def effective_rank(workload: np.ndarray) -> float:
    """
    Compute the effective rank.
    """

    sigma = singular_values(workload)
    return float(np.exp(entropy(sigma)))


###############################################################################
# Workload Complexity Index
###############################################################################


def compute_wci(workload: np.ndarray,) -> float:
    """
    Compute the normalized Workload Complexity Index.

    WCI = r_eff / min(m, n)
    """

    m, n = workload.shape
    denominator = min(m, n,)

    if denominator == 0:
        return 0.0

    value = effective_rank(workload,) / denominator
    return float(np.clip(value,0.0,1.0,))


###############################################################################
# Complexity classes
###############################################################################


def complexity_class(wci: float,) -> str:
    """
    Return a qualitative workload complexity.
    """

    if wci < 0.20:
        return "Very Low"

    if wci < 0.40:
        return "Low"

    if wci < 0.60:
        return "Moderate"

    if wci < 0.80:
        return "High"

    return "Very High"


###############################################################################
# Statistics
###############################################################################


def statistics(workload: np.ndarray,) -> dict:
    """
    Compute workload statistics.
    """

    sigma = singular_values(workload)
    erank = effective_rank(workload)
    wci = compute_wci(workload)
    m, n = workload.shape

    return {
        "queries": m,
        "domain": n,
        "rank": int(np.linalg.matrix_rank(workload)),
        "effective_rank": erank,
        "wci": wci,
        "complexity": complexity_class(wci),
        "largest_sv": float(sigma[0]),
        "smallest_sv": float(sigma[-1]),
    }


def compute_complexity(workload):
    """
    Compute workload complexity statistics.
    """

    sigma = singular_values(workload)

    effective_rank_value = float(np.exp(entropy(sigma)))

    m, n = workload.shape

    wci = effective_rank_value / min(m, n)

    rank_factor = (np.linalg.matrix_rank(workload) / m)

    density_factor = (np.count_nonzero(workload)/workload.size)

    unique_queries = np.unique(workload, axis=0).shape[0]

    redundancy_factor = (unique_queries / m)

    complexity = (
            0.50 * wci
            + 0.20 * rank_factor
            + 0.15 * density_factor
            + 0.15 * redundancy_factor
    )

    return {
        "wci": float(np.clip(wci, 0.0, 1.0)),
        "effective_rank": effective_rank_value,
        "rank_factor": rank_factor,
        "density_factor": density_factor,
        "redundancy_factor": redundancy_factor,
        "sigma1": float(sigma[0]),
        "complexity": float(complexity),
        "weight": None,
    }

def print_statistics(workload: np.ndarray,) -> None:
    """
    Print workload statistics.
    """

    stats = statistics(workload)

    print("=" * 80)
    print("Workload Complexity")
    print("=" * 80)

    for key, value in stats.items():
        print(f"{key:18s}: {value}")


###############################################################################
# Example
###############################################################################

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    W = rng.integers(0, 2, (20, 4096)).astype(float)
    print_statistics(W)
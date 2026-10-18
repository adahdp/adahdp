"""Optimal DP/LDP crossover rules for AdaHDP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HDMMFlatCrossover:
    """Diagnostics for the flat-budget HDMM crossover theorem."""

    dp_microbatches: int
    dp_queries: float
    continuous_dp_microbatches: float
    expected_microbatch_size: float
    gamma_hdmm: float
    se_dp: float
    var_ldp: float


@dataclass(frozen=True)
class HDMMAECrossover:
    """Crossover selected by minimizing expected query-weighted MAE."""

    dp_microbatches: int
    expected_mae: float
    objectives: tuple[float, ...]
    dp_queries: float
    expected_microbatch_size: float
    gamma_hdmm: float
    se_dp: float
    var_ldp: float


def optimal_hdmm_mae_crossover(
    *,
    n_dp: int,
    epsilon_dp: float,
    query_counts: Sequence[int],
    gamma_hdmm: float | Sequence[float],
    se_dp: float | Sequence[float],
    var_ldp: float | Sequence[float],
) -> HDMMAECrossover:
    """Choose the flat-budget crossover that minimizes expected MAE.

    For a candidate prefix length ``k``, DP receives ``epsilon_dp / k`` per
    microbatch and therefore has variance

    ``se_t + 6 * gamma_t * (k / (n_dp * epsilon_dp))**2``.

    The objective is the query-weighted Gaussian absolute-error proxy over the
    DP prefix and LDP suffix.  Unlike the older variance-equality rule, this
    directly optimizes the total workload error and supports heterogeneous
    query counts and per-microbatch variance terms.
    """
    if n_dp <= 0 or not math.isfinite(float(epsilon_dp)) or epsilon_dp <= 0.0:
        raise ValueError("n_dp and epsilon_dp must be positive and finite.")
    counts = np.asarray(query_counts, dtype=float)
    if counts.ndim != 1 or counts.size == 0 or np.any(~np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("query_counts must contain positive finite values.")
    count = int(counts.size)

    def _terms(value: float | Sequence[float], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(count, float(array))
        if array.shape != (count,) or np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be a non-negative scalar or length-{count} sequence.")
        return array

    gamma = _terms(gamma_hdmm, "gamma_hdmm")
    sampling = _terms(se_dp, "se_dp")
    ldp = _terms(var_ldp, "var_ldp")
    total_queries = float(np.sum(counts))
    mae_factor = math.sqrt(2.0 / math.pi)
    objectives: list[float] = []
    for k in range(count + 1):
        if k:
            dp_variance = sampling[:k] + 6.0 * gamma[:k] * (
                k / (float(n_dp) * float(epsilon_dp))
            ) ** 2
            dp_error = mae_factor * np.sqrt(dp_variance)
            dp_sum = float(np.dot(counts[:k], dp_error))
        else:
            dp_sum = 0.0
        if k < count:
            ldp_error = mae_factor * np.sqrt(ldp[k:])
            ldp_sum = float(np.dot(counts[k:], ldp_error))
        else:
            ldp_sum = 0.0
        objectives.append((dp_sum + ldp_sum) / total_queries)

    best = int(np.argmin(np.asarray(objectives, dtype=float)))
    expected_size = total_queries / count
    return HDMMAECrossover(
        dp_microbatches=best,
        expected_mae=float(objectives[best]),
        objectives=tuple(float(value) for value in objectives),
        dp_queries=float(np.sum(counts[:best])),
        expected_microbatch_size=expected_size,
        gamma_hdmm=float(np.mean(gamma)),
        se_dp=float(np.mean(sampling)),
        var_ldp=float(np.mean(ldp)),
    )


def estimate_dp_sampling_error(
    *,
    workloads: list[np.ndarray],
    reconstructed_population: np.ndarray,
    n_dp: int,
    population_size: int,
) -> float:
    """Estimate mean query sampling variance using an LDP distribution and FPC."""
    if n_dp <= 0:
        raise ValueError("n_dp must be positive.")
    if population_size < n_dp:
        raise ValueError("population_size must be at least n_dp.")
    if not workloads:
        raise ValueError("workloads must not be empty.")

    reconstructed = np.asarray(reconstructed_population, dtype=float)
    if reconstructed.ndim != 1 or not np.all(np.isfinite(reconstructed)):
        raise ValueError("reconstructed_population must be a finite vector.")
    reconstructed = np.maximum(reconstructed, 0.0)
    reconstructed_total = float(np.sum(reconstructed))
    if reconstructed_total <= 0.0:
        raise ValueError("The reconstructed LDP distribution has zero mass.")

    probabilities = reconstructed / reconstructed_total
    fpc = max(0.0, 1.0 - n_dp / population_size)
    variances: list[np.ndarray] = []
    for workload in workloads:
        workload = np.asarray(workload, dtype=float)
        means = workload @ probabilities
        second_moments = (workload**2) @ probabilities
        variances.append(np.maximum(0.0, second_moments - means**2) / n_dp * fpc)
    return float(np.mean(np.concatenate(variances)))


def optimal_hdmm_flat_crossover(
    *,
    n_dp: int,
    epsilon_dp: float,
    expected_queries: int,
    expected_microbatches: int,
    gamma_hdmm: float,
    se_dp: float,
    var_ldp: float,
) -> HDMMFlatCrossover:
    """Compute Theorem ``Optimal Crossover under Flat HDMM Budget Allocation``.

    The feasible set in the theorem contains at least one DP microbatch.  If
    LDP is no worse than DP sampling (so the square-root argument is not
    positive), the constrained optimum is therefore the lower endpoint.
    """
    if n_dp <= 0:
        raise ValueError("n_dp must be positive.")
    if not math.isfinite(epsilon_dp) or epsilon_dp <= 0.0:
        raise ValueError("epsilon_dp must be finite and positive.")
    if expected_queries <= 0:
        raise ValueError("expected_queries must be positive.")
    if expected_microbatches <= 0:
        raise ValueError("expected_microbatches must be positive.")
    if not math.isfinite(gamma_hdmm) or gamma_hdmm < 0.0:
        raise ValueError("gamma_hdmm must be finite and non-negative.")
    if not math.isfinite(se_dp) or se_dp < 0.0:
        raise ValueError("se_dp must be finite and non-negative.")
    if not math.isfinite(var_ldp) or var_ldp < 0.0:
        raise ValueError("var_ldp must be finite and non-negative.")

    advantage = var_ldp - se_dp
    if advantage <= 0.0:
        continuous = 0.0
    elif gamma_hdmm == 0.0:
        continuous = math.inf
    else:
        continuous = (
            n_dp
            * epsilon_dp
            * math.sqrt(advantage / (6.0 * gamma_hdmm))
        )

    clamped = min(max(1.0, continuous), float(expected_microbatches))
    # Explicit half-up rounding avoids Python's tie-to-even behavior.
    dp_microbatches = int(math.floor(clamped + 0.5))
    expected_size = expected_queries / expected_microbatches

    return HDMMFlatCrossover(
        dp_microbatches=dp_microbatches,
        dp_queries=expected_size * dp_microbatches,
        continuous_dp_microbatches=continuous,
        expected_microbatch_size=expected_size,
        gamma_hdmm=gamma_hdmm,
        se_dp=se_dp,
        var_ldp=var_ldp,
    )

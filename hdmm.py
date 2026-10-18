
"""
hdmm.py

HDMM wrapper for the AdaHDP prototype.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hdmm_matrix import EkteloMatrix
from hdmm_mechanism import HDMM

LAMBDA_MIN_TOL = 1e-8
STATISTIC_TOL = 1e-12


###############################################################################
# Results
###############################################################################


@dataclass
class PreparedHDMM:
    """
    An optimized HDMM mechanism prepared without accessing private data.

    Attributes
    ----------
    workload
        Workload represented as an EkteloMatrix.
    dense_workload
        Dense NumPy representation of the workload.
    mechanism
        Optimized HDMM mechanism.
    sensitivity
        L1 sensitivity of the optimized strategy matrix.
    lambda_min
        Smallest positive eigenvalue of A^T A.
    sigma1
        Largest singular value of the workload matrix.
    estimated_C
        Raw spectral upper-bound constant.
    variance_term
        Exact HDMM reconstruction variance term:
        tr(W (A^T A)^dagger W^T).
    """

    workload: EkteloMatrix
    dense_workload: np.ndarray
    mechanism: HDMM
    sensitivity: float
    lambda_min: float
    sigma1: float
    estimated_C: float
    variance_term: float


@dataclass
class HDMMResult:
    """
    Result produced by executing a prepared HDMM mechanism.
    """

    histogram: np.ndarray
    answers: np.ndarray
    mechanism: HDMM
    sensitivity: float
    lambda_min: float
    sigma1: float
    estimated_C: float
    variance_term: float


###############################################################################
# Matrix conversion
###############################################################################


def to_dense(
    matrix: Any,
) -> np.ndarray:
    """
    Convert an HDMM matrix or array-like object into a dense NumPy matrix.
    """

    if isinstance(matrix, np.ndarray):
        dense = matrix
    elif hasattr(matrix, "dense_matrix"):
        dense = matrix.dense_matrix()
    elif hasattr(matrix, "matrix"):
        dense = matrix.matrix
    else:
        dense = matrix

    dense = np.asarray(
        dense,
        dtype=float,
    )

    if dense.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional matrix, but received "
            f"shape {dense.shape}."
        )

    if not np.all(
        np.isfinite(dense)
    ):
        raise ValueError(
            "The matrix contains non-finite values."
        )

    return dense


def to_ektelo(
    workload: Any,
) -> EkteloMatrix:
    """
    Convert a workload into an EkteloMatrix.
    """

    if isinstance(
        workload,
        EkteloMatrix,
    ):
        return workload

    return EkteloMatrix(
        to_dense(workload)
    )


###############################################################################
# Validation
###############################################################################


def _validate_positive_statistic(
    value: float,
    name: str,
    allow_zero: bool = False,
) -> float:
    """
    Validate and return one scalar HDMM statistic.
    """

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite, but received {value}."
        )

    if allow_zero:
        if value < 0.0:
            raise ValueError(
                f"{name} must be non-negative, but received {value}."
            )
    elif value <= 0.0:
        raise ValueError(
            f"{name} must be positive, but received {value}."
        )

    return value


###############################################################################
# HDMM preparation
###############################################################################


def prepare_hdmm(
    workload: np.ndarray,
    domain_size: int,
    seed: int = 0,
) -> PreparedHDMM:
    """
    Optimize HDMM and compute privacy-free strategy statistics.

    This function analyzes only the workload and optimized strategy. It does
    not access private histogram values and does not add noise.
    """

    dense_workload = to_dense(
        workload
    )

    if domain_size <= 0:
        raise ValueError(
            "domain_size must be positive."
        )

    workload_domain_size = int(
        dense_workload.shape[1]
    )

    if domain_size != workload_domain_size:
        raise ValueError(
            "domain_size must match the workload domain size: "
            f"{domain_size} != {workload_domain_size}."
        )

    if dense_workload.shape[0] <= 0:
        raise ValueError(
            "The workload must contain at least one query."
        )

    ektelo_workload = to_ektelo(
        dense_workload
    )

    dummy_histogram = np.zeros(
        domain_size,
        dtype=float,
    )

    mechanism = HDMM(
        ektelo_workload,
        dummy_histogram,
        1.0,
        seed,
    )

    mechanism.optimize()
    mechanism.compute_statistics()

    sensitivity = _validate_positive_statistic(
        mechanism.delta,
        "HDMM strategy sensitivity",
    )

    lambda_min = max(
        _validate_positive_statistic(
            mechanism.lambda_min,
            "Smallest positive strategy eigenvalue",
        ),
        LAMBDA_MIN_TOL,
    )

    variance_term = _validate_positive_statistic(
        mechanism.variance_term,
        "HDMM variance term",
        allow_zero=True,
    )

    singular_values = np.linalg.svd(
        dense_workload,
        compute_uv=False,
    )

    sigma1 = (
        float(singular_values[0])
        if singular_values.size > 0
        else 0.0
    )

    sigma1 = _validate_positive_statistic(
        sigma1,
        "Largest workload singular value",
        allow_zero=True,
    )

    estimated_c = (
        2.0
        * sensitivity**2
        * sigma1**2
        / lambda_min
    )

    estimated_c = _validate_positive_statistic(
        estimated_c,
        "Raw HDMM spectral constant",
        allow_zero=True,
    )

    return PreparedHDMM(
        workload=ektelo_workload,
        dense_workload=dense_workload,
        mechanism=mechanism,
        sensitivity=sensitivity,
        lambda_min=lambda_min,
        sigma1=sigma1,
        estimated_C=estimated_c,
        variance_term=variance_term,
    )


###############################################################################
# HDMM execution
###############################################################################


def run_prepared_hdmm(
    prepared: PreparedHDMM,
    histogram: np.ndarray,
    epsilon: float,
    seed: int = 0,
) -> HDMMResult:
    """
    Execute an already optimized HDMM strategy.

    The strategy used to compute the variance term is reused when answering
    the workload.
    """

    if not isinstance(
        prepared,
        PreparedHDMM,
    ):
        raise TypeError(
            "prepared must be a PreparedHDMM object."
        )

    histogram = np.asarray(
        histogram,
        dtype=float,
    )

    if histogram.ndim != 1:
        raise ValueError(
            "histogram must be one-dimensional."
        )

    if not np.all(
        np.isfinite(histogram)
    ):
        raise ValueError(
            "histogram contains non-finite values."
        )

    domain_size = int(
        prepared.dense_workload.shape[1]
    )

    if histogram.size != domain_size:
        raise ValueError(
            "Histogram size must match the workload domain size: "
            f"{histogram.size} != {domain_size}."
        )

    epsilon = float(
        epsilon
    )

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            "epsilon must be finite and positive."
        )

    mechanism = prepared.mechanism

    mechanism.x = histogram.copy()
    mechanism.eps = epsilon
    mechanism.prng = np.random.RandomState(
        seed
    )

    estimated_histogram = np.asarray(
        mechanism.run(),
        dtype=float,
    ).reshape(-1)

    if estimated_histogram.size != domain_size:
        raise ValueError(
            "The reconstructed histogram has an unexpected size: "
            f"{estimated_histogram.size} != {domain_size}."
        )

    if not np.all(
        np.isfinite(estimated_histogram)
    ):
        raise ValueError(
            "The reconstructed histogram contains non-finite values."
        )

    answers = np.asarray(
        prepared.workload.dot(
            estimated_histogram
        ),
        dtype=float,
    ).reshape(-1)

    if answers.size != prepared.dense_workload.shape[0]:
        raise ValueError(
            "The answer vector has an unexpected size: "
            f"{answers.size} != {prepared.dense_workload.shape[0]}."
        )

    return HDMMResult(
        histogram=estimated_histogram,
        answers=answers,
        mechanism=mechanism,
        sensitivity=prepared.sensitivity,
        lambda_min=prepared.lambda_min,
        sigma1=prepared.sigma1,
        estimated_C=prepared.estimated_C,
        variance_term=prepared.variance_term,
    )


###############################################################################
# Convenience functions
###############################################################################


def estimate_c(
    workload: np.ndarray,
    seed: int = 0,
) -> float:
    """
    Compute the raw HDMM spectral constant without accessing private data.
    """

    dense_workload = to_dense(
        workload
    )

    prepared = prepare_hdmm(
        workload=dense_workload,
        domain_size=dense_workload.shape[1],
        seed=seed,
    )

    return float(
        prepared.estimated_C
    )


def estimate_variance_term(
    workload: np.ndarray,
    seed: int = 0,
) -> float:
    """
    Compute the exact HDMM variance term without accessing private data.
    """

    dense_workload = to_dense(
        workload
    )

    prepared = prepare_hdmm(
        workload=dense_workload,
        domain_size=dense_workload.shape[1],
        seed=seed,
    )

    return float(
        prepared.variance_term
    )


def answer_hdmm(
    workload: np.ndarray,
    histogram: np.ndarray,
    epsilon: float,
    seed: int = 0,
) -> HDMMResult:
    """
    Optimize and execute HDMM on one workload.
    """

    histogram = np.asarray(
        histogram,
        dtype=float,
    )

    if histogram.ndim != 1:
        raise ValueError(
            "histogram must be one-dimensional."
        )

    prepared = prepare_hdmm(
        workload=workload,
        domain_size=histogram.size,
        seed=seed,
    )

    return run_prepared_hdmm(
        prepared=prepared,
        histogram=histogram,
        epsilon=epsilon,
        seed=seed,
    )


def estimate_histogram(
    workload: np.ndarray,
    histogram: np.ndarray,
    epsilon: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Return only the reconstructed histogram.
    """

    return answer_hdmm(
        workload=workload,
        histogram=histogram,
        epsilon=epsilon,
        seed=seed,
    ).histogram


def answer_workload(
    workload: np.ndarray,
    histogram: np.ndarray,
    epsilon: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Return only the reconstructed workload answers.
    """

    return answer_hdmm(
        workload=workload,
        histogram=histogram,
        epsilon=epsilon,
        seed=seed,
    ).answers


###############################################################################
# Error evaluation
###############################################################################


def expected_error(
    workload: np.ndarray,
    histogram: np.ndarray,
    epsilon: float,
    seed: int = 0,
) -> float:
    """
    Execute HDMM and return the observed workload RMSE.
    """

    dense_workload = to_dense(
        workload
    )

    histogram = np.asarray(
        histogram,
        dtype=float,
    )

    result = answer_hdmm(
        workload=dense_workload,
        histogram=histogram,
        epsilon=epsilon,
        seed=seed,
    )

    true_answers = (
        dense_workload
        @ histogram
    )

    error = (
        result.answers
        - true_answers
    )

    return float(
        np.sqrt(
            np.mean(
                error**2
            )
        )
    )


def theoretical_expected_squared_error(
    prepared: PreparedHDMM,
    epsilon: float,
) -> float:
    """
    Return the HDMM expected squared workload error.

    The value is:

    2 * Delta_A^2 * variance_term / epsilon^2.
    """

    epsilon = float(
        epsilon
    )

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            "epsilon must be finite and positive."
        )

    return float(
        2.0
        * prepared.sensitivity**2
        * prepared.variance_term
        / epsilon**2
    )


###############################################################################
# Example
###############################################################################


if __name__ == "__main__":
    rng = np.random.default_rng(
        42
    )

    histogram = rng.poisson(
        10,
        128,
    ).astype(
        float
    )

    workload = np.eye(
        128,
        dtype=float,
    )

    prepared = prepare_hdmm(
        workload=workload,
        domain_size=histogram.size,
        seed=42,
    )

    result = run_prepared_hdmm(
        prepared=prepared,
        histogram=histogram,
        epsilon=1.0,
        seed=42,
    )

    theoretical_error = theoretical_expected_squared_error(
        prepared=prepared,
        epsilon=1.0,
    )

    print("=" * 80)
    print("HDMM")
    print("=" * 80)
    print(f"Histogram size       : {len(result.histogram)}")
    print(f"Answers              : {len(result.answers)}")
    print(f"Sensitivity          : {result.sensitivity:.6f}")
    print(f"Lambda min           : {result.lambda_min:.6e}")
    print(f"Sigma 1              : {result.sigma1:.6f}")
    print(f"Raw C                : {result.estimated_C:.6f}")
    print(f"Variance term        : {result.variance_term:.6f}")
    print(f"Expected squared err.: {theoretical_error:.6f}")

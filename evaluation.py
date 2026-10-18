"""
evaluation.py

Evaluation utilities for the AdaHDP prototype.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationResult:

    mechanism: str

    n_queries: int

    mae: float

    mse: float

    rmse: float

    max_error: float


###############################################################################
# Metrics
###############################################################################


def mae(
    truth: np.ndarray,
    estimate: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.abs(
                truth - estimate,
            ),
        ),
    )


def mse(
    truth: np.ndarray,
    estimate: np.ndarray,
) -> float:

    return float(
        np.mean(
            (
                truth - estimate
            ) ** 2,
        ),
    )


def rmse(
    truth: np.ndarray,
    estimate: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            mse(
                truth,
                estimate,
            ),
        ),
    )


def max_error(
    truth: np.ndarray,
    estimate: np.ndarray,
) -> float:

    return float(
        np.max(
            np.abs(
                truth - estimate,
            ),
        ),
    )


###############################################################################
# Single microbatch
###############################################################################


def evaluate(
    workload: np.ndarray,
    histogram: np.ndarray,
    answers: np.ndarray,
    mechanism: str,
) -> EvaluationResult:
    """
    Evaluate one microbatch.
    """

    truth = workload @ histogram

    return EvaluationResult(
        mechanism=mechanism,
        n_queries=int(truth.size),
        mae=mae(
            truth,
            answers,
        ),
        mse=mse(
            truth,
            answers,
        ),
        rmse=rmse(
            truth,
            answers,
        ),
        max_error=max_error(
            truth,
            answers,
        ),
    )


###############################################################################
# Complete experiment
###############################################################################


def evaluate_experiment(
    microbatches,
    results,
    histogram,
) -> list[EvaluationResult]:
    """
    Evaluate all microbatches.
    """

    metrics = []

    for workload, result in zip(
        microbatches,
        results,
    ):

        metrics.append(
            evaluate(
                workload,
                histogram,
                result.answers,
                result.mechanism,
            ),
        )

    return metrics


###############################################################################
# Summary
###############################################################################


def summarize(
    metrics: list[EvaluationResult],
) -> dict:

    if not metrics:
        raise ValueError("metrics must contain at least one microbatch.")

    query_counts = np.asarray(
        [m.n_queries for m in metrics],
        dtype=float,
    )
    if np.any(query_counts <= 0.0):
        raise ValueError("Every microbatch must contain at least one query.")
    total_queries = int(np.sum(query_counts))

    maes = np.array(
        [m.mae for m in metrics],
    )

    mses = np.array(
        [m.mse for m in metrics],
    )

    rmses = np.array(
        [m.rmse for m in metrics],
    )

    maximum = np.array(
        [m.max_error for m in metrics],
    )

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.average(values, weights=query_counts))

    def weighted_std(values: np.ndarray) -> float:
        mean = weighted_mean(values)
        variance = np.average((values - mean) ** 2, weights=query_counts)
        return float(np.sqrt(variance))

    mae_mean = weighted_mean(maes)
    mse_mean = weighted_mean(mses)

    hdmm = sum(
        m.mechanism == "HDMM"
        for m in metrics
    )

    oue = sum(
        m.mechanism == "OUE"
        for m in metrics
    )

    return {
        "microbatches": len(metrics),
        "queries": total_queries,
        "hdmm_batches": hdmm,
        "oue_batches": oue,
        "mae_mean": mae_mean,
        "mae_std": weighted_std(maes),
        "mse_mean": mse_mean,
        "mse_std": weighted_std(mses),
        "rmse_mean": float(np.sqrt(mse_mean)),
        "rmse_std": weighted_std(rmses),
        "max_error": float(np.max(maximum)),
    }


###############################################################################
# Printing
###############################################################################


def print_summary(
    summary: dict,
) -> None:

    print("=" * 80)

    print("Evaluation")

    print("=" * 80)

    print(
        f"Microbatches : {summary['microbatches']}",
    )

    print(
        f"Queries      : {summary['queries']}",
    )

    print(
        f"HDMM batches : {summary['hdmm_batches']}",
    )

    print(
        f"OUE batches  : {summary['oue_batches']}",
    )

    print()

    print(
        f"MAE  : {summary['mae_mean']:.6f} ± {summary['mae_std']:.6f}",
    )

    print(
        f"MSE  : {summary['mse_mean']:.6f} ± {summary['mse_std']:.6f}",
    )

    print(
        f"RMSE : {summary['rmse_mean']:.6f} ± {summary['rmse_std']:.6f}",
    )

    print(
        f"MAX  : {summary['max_error']:.6f}",
    )


###############################################################################
# Example
###############################################################################

if __name__ == "__main__":

    rng = np.random.default_rng(
        42,
    )

    workload = rng.integers(
        0,
        2,
        (
            20,
            128,
        ),
    ).astype(
        float,
    )

    histogram = rng.poisson(
        10,
        128,
    ).astype(
        float,
    )

    truth = workload @ histogram

    noisy = truth + rng.normal(
        0,
        2,
        truth.shape,
    )

    metric = evaluate(
        workload,
        histogram,
        noisy,
        "HDMM",
    )

    summary = summarize(
        [metric],
    )

    print_summary(
        summary,
    )

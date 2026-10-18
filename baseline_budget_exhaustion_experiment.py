
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import config
from hdmm import PreparedHDMM, prepare_hdmm


@dataclass(frozen=True)
class ExperimentSettings:
    total_microbatches: int
    expected_microbatches: int
    runs: int = config.N_RUNS
    epsilon_dp: float = config.EPS_DP
    epsilon_ldp: float = config.EPS_LDP
    seed: int = config.SEED


def expected_microbatch_size() -> float:
    if config.VARIABLE_MICROBATCH_SIZE:
        return (config.MICROBATCH_SIZE_MIN + config.MICROBATCH_SIZE_MAX) / 2.0
    return float(config.MICROBATCH_SIZE)


def normalized_population_histogram(dp_histogram: np.ndarray, ldp_histogram: np.ndarray) -> np.ndarray:
    population = np.asarray(dp_histogram + ldp_histogram, dtype=float)
    total = float(population.sum())
    if total <= 0:
        raise ValueError("The combined population must be nonempty.")
    return population / total


def prepare_shared_hdmm_strategies(
    batches: list[np.ndarray], domain_size: int, seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, PreparedHDMM]:
    prepared = {}
    total = len(batches)
    interval = max(1, (total + 9) // 10)
    for index, batch in enumerate(batches):
        prepared[index] = prepare_hdmm(batch, domain_size=domain_size, seed=seed + index)
        completed = index + 1
        if progress_callback and (completed % interval == 0 or completed == total):
            progress_callback(completed, total)
    return prepared


"""
scheduler.py

AdaHDP scheduler.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

import budget
import config
import wci
from crossover import (
    HDMMAECrossover,
    HDMMFlatCrossover,
    estimate_dp_sampling_error,
    optimal_hdmm_mae_crossover,
    optimal_hdmm_flat_crossover,
)
from hdmm import PreparedHDMM, prepare_hdmm, run_prepared_hdmm
from ldp import build_engine

EPSILON_TOL = 1e-10

VARIANCE_NORMALIZATION_MIN = 0.5
VARIANCE_NORMALIZATION_MAX = 2.0
VARIANCE_DAMPING_EXPONENT = 0.5


###############################################################################
# Results
###############################################################################


@dataclass
class BatchResult:
    """
    Results and diagnostic information for one query microbatch.
    """

    microbatch: int
    mechanism: str
    epsilon: float
    remaining_budget: float
    answers: Any

    wci: float | None = None
    complexity: float | None = None
    rank_factor: float | None = None
    density_factor: float | None = None
    redundancy_factor: float | None = None
    effective_rank: float | None = None

    weight: float | None = None

    variance_term: float | None = None
    normalized_variance: float | None = None

    raw_C: float | None = None
    estimated_C: float | None = None

    sensitivity: float | None = None
    lambda_min: float | None = None
    sigma1: float | None = None

    mae: float | None = None
    rmse: float | None = None


###############################################################################
# Scheduler
###############################################################################


class AdaHDP:
    """
    Adaptive scheduler for hybrid DP/LDP query answering.
    """

    def __init__(
        self,
        epsilon_dp: float = config.EPS_DP,
        epsilon_ldp: float = config.EPS_LDP,
        cp: int = config.CP,
        beta: float = config.BETA,
        policy: str = config.BUDGET_POLICY,
        seed: int = config.SEED,
        expected_microbatches: int | None = None,
        expected_queries: int | None = None,
        smart_target_variance: float = config.SMART_TARGET_VARIANCE,
        smart_min_epsilon: float = config.SMART_MIN_EPSILON,
    ) -> None:
        if epsilon_dp < 0.0:
            raise ValueError("epsilon_dp must be non-negative.")

        if epsilon_ldp <= 0.0:
            raise ValueError("epsilon_ldp must be positive.")

        if cp < 0:
            raise ValueError("cp must be non-negative.")

        if policy == "dp_geometric" and not 0.0 < beta < 1.0:
            raise ValueError(
                "beta must lie strictly between 0 and 1 for dp_geometric."
            )

        if expected_microbatches is not None and expected_microbatches <= 0:
            raise ValueError("expected_microbatches must be positive when set.")

        if expected_queries is not None and expected_queries <= 0:
            raise ValueError("expected_queries must be positive when set.")

        self.epsilon_dp = float(epsilon_dp)
        self.epsilon_ldp = float(epsilon_ldp)
        self.cp = int(cp)
        self.beta = float(beta)
        self.policy = policy
        self.seed = int(seed)
        self.expected_microbatches = expected_microbatches
        self.expected_queries = expected_queries
        self.smart_target_variance = float(smart_target_variance)
        self.smart_min_epsilon = float(smart_min_epsilon)
        if not np.isfinite(self.smart_min_epsilon) or self.smart_min_epsilon < 0.0:
            raise ValueError("smart_min_epsilon must be finite and non-negative.")

        self.remaining_budget = float(epsilon_dp)
        self.dp_queries_answered = 0
        self.geometric_query_index = 0
        self.expected_dp_batches = 0
        self._online_dp_batch_count: int | None = None
        self._smart_min_dp_batches: int | None = None
        self.expected_weight: float | None = None

        self.variance_reference: float | None = None
        self.ldp_engine = None
        self.prepared_hdmm: dict[int, PreparedHDMM] = {}
        self.flat_crossover: HDMMFlatCrossover | HDMMAECrossover | None = None

    ###########################################################################
    # Initialization
    ###########################################################################

    def initialize(self, ldp_histogram: np.ndarray) -> None:
        """
        Initialize the LDP estimator.
        """
        self.ldp_engine = build_engine(histogram=ldp_histogram,
                                       epsilon=self.epsilon_ldp)

    ###########################################################################
    # Utility methods
    ###########################################################################

    def _answer_with_prepared_hdmm(
        self,
        workload: np.ndarray,
        dp_histogram: np.ndarray,
        prepared: PreparedHDMM,
        epsilon: float,
    ) -> BatchResult:
        """Answer one microbatch using a prepared HDMM strategy."""
        stats = wci.compute_complexity(workload)
        hdmm_result = run_prepared_hdmm(
            prepared=prepared,
            histogram=dp_histogram,
            epsilon=epsilon,
            seed=self.seed,
        )
        n_dp = float(np.sum(dp_histogram))
        if n_dp <= 0.0:
            raise ValueError("The DP histogram must contain at least one user.")
        answers = np.asarray(hdmm_result.answers, dtype=float) / n_dp
        true_answers = np.asarray(workload @ (dp_histogram / n_dp), dtype=float)
        error = answers - true_answers
        return BatchResult(
            microbatch=0,
            mechanism="HDMM",
            epsilon=epsilon,
            remaining_budget=self.remaining_budget,
            answers=answers,
            wci=float(stats["wci"]),
            complexity=float(stats["complexity"]),
            rank_factor=float(stats["rank_factor"]),
            density_factor=float(stats["density_factor"]),
            redundancy_factor=float(stats["redundancy_factor"]),
            effective_rank=float(stats["effective_rank"]),
            variance_term=float(prepared.variance_term),
            raw_C=float(prepared.estimated_C),
            sensitivity=float(prepared.sensitivity),
            lambda_min=float(prepared.lambda_min),
            sigma1=float(prepared.sigma1),
            mae=float(np.mean(np.abs(error))),
            rmse=float(np.sqrt(np.mean(error ** 2))),
        )

    def _answer_with_ldp(self, workload: np.ndarray,) -> BatchResult:
        """
        Answer one microbatch using the LDP estimator.
        """

        if self.ldp_engine is None:
            raise RuntimeError("The LDP estimator has not been initialized.")

        n_ldp = float(np.sum(self.ldp_engine.histogram))
        if n_ldp <= 0.0:
            raise ValueError("The LDP histogram must contain at least one user.")
        answers = np.asarray(self.ldp_engine.answer(workload), dtype=float) / n_ldp

        return BatchResult(
            microbatch=0,
            mechanism="OUE",
            epsilon=self.epsilon_ldp,
            remaining_budget=self.remaining_budget,
            answers=answers,
        )

    def _prepare_strategy(
        self,
        workload: np.ndarray,
        dp_histogram: np.ndarray,
    ) -> PreparedHDMM:
        """
        Optimize an HDMM strategy without accessing private data.
        """

        prepared = prepare_hdmm(
            workload=workload,
            domain_size=dp_histogram.size,
            seed=self.seed,
        )

        if not hasattr(prepared, "variance_term"):
            raise AttributeError(
                "PreparedHDMM does not contain variance_term. "
                "Update hdmm.py and mechanism.py before using the "
                "variance-based scheduler."
            )

        variance_term = float(prepared.variance_term)

        if not np.isfinite(variance_term) or variance_term <= 0.0:
            raise ValueError(
                "The HDMM variance term must be finite and positive: "
                f"{variance_term}."
            )

        return prepared

    def _normalize_variance(self, variance_term: float,) -> float:
        """Normalize an HDMM variance term."""
        if not np.isfinite(variance_term) or variance_term <= 0.0:
            raise ValueError("The HDMM variance term must be finite and positive.")

        if (
                self.variance_reference is None
                or not np.isfinite(self.variance_reference)
                or self.variance_reference <= 0.0
        ):
            raise RuntimeError("The variance normalization reference has not been initialized.")

        return float(variance_term / self.variance_reference)

    def _compute_online_weight(
        self,
        normalized_variance: float,
        workload_wci: float,
        complexity: float,
        microbatch_size: int,
    ) -> float:
        """
        Compute the current online-allocation weight.

        Supported policies:
        - online_variance
        - online_variance_wci
        - online_variance_complexity
        - online_optimal, kept as an alias for online_variance_complexity
        """
        if normalized_variance <= 0.0 or not np.isfinite(normalized_variance):
            raise ValueError(
                "normalized_variance must be finite and positive."
            )

        if self.policy in {
            "online_variance",
            "hdmm_online_variance",
            "smart",
            "hdmm_smart",
        }:
            factor = normalized_variance
        elif self.policy == "online_variance_wci":
            factor = normalized_variance * workload_wci
        elif self.policy in {"online_variance_complexity", "online_optimal"}:
            factor = normalized_variance * microbatch_size * complexity
        else:
            raise ValueError(f"Unknown online budget policy '{self.policy}'.")

        weight = factor ** (1.0 / 3.0)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Invalid online-allocation weight: {weight}.")
        return float(weight)

    ###########################################################################
    # Answer one microbatch
    ###########################################################################

    def answer_microbatch(
        self,
        workload: np.ndarray,
        dp_histogram: np.ndarray,
        prepared: PreparedHDMM | None = None,
    ) -> BatchResult:
        """
        Answer one microbatch using either HDMM or the LDP estimator.
        """

        #######################################################################
        # Switch to LDP when the DP phase has ended
        #######################################################################

        if self.remaining_budget <= EPSILON_TOL:
            self.remaining_budget = 0.0
            return self._answer_with_ldp(workload)

        if (
            self.expected_dp_batches <= 0
            and self.policy not in {"smart", "hdmm_smart"}
        ):
            return self._answer_with_ldp(workload)

        #######################################################################
        # Workload statistics
        #######################################################################

        stats = wci.compute_complexity(workload)
        workload_wci = float(stats["wci"])
        complexity = float(stats["complexity"])

        #######################################################################
        # Privacy-budget allocation
        #######################################################################

        variance_term: float | None = None
        normalized_variance: float | None = None

        if self.policy.startswith("online_") or self.policy == "hdmm_online_variance":
            if prepared is None:
                prepared = self._prepare_strategy(
                    workload=workload,
                    dp_histogram=dp_histogram,
                )

            variance_term = float(prepared.variance_term)
            normalized_variance = self._normalize_variance(variance_term)

            if self.expected_weight is None:
                raise RuntimeError(
                    "expected_weight was not initialized for "
                    "the online_optimal policy."
                )

            current_weight = self._compute_online_weight(
                normalized_variance=normalized_variance,
                workload_wci=workload_wci,
                complexity=complexity,
                microbatch_size=workload.shape[0],
            )
            epsilon = budget.online_optimal(
                remaining_budget=self.remaining_budget,
                current_weight=current_weight,
                remaining_microbatches=self.expected_dp_batches,
                expected_weight=self.expected_weight,
            )
        elif self.policy == "dp_geometric":
            # Allocate beta * epsilon * (1 - beta)^(i - 1) to the i-th
            # DP microbatch.  The index advances only when a DP answer is
            # attempted, so LDP-only fallbacks do not consume a slot.
            epsilon = (
                self.beta
                * self.epsilon_dp
                * (1.0 - self.beta) ** self.geometric_query_index
            )
            self.geometric_query_index += 1
        elif self.policy in {"smart", "hdmm_smart"}:
            if prepared is None:
                prepared = self._prepare_strategy(
                    workload=workload,
                    dp_histogram=dp_histogram,
                )
            variance_term = float(prepared.variance_term)
            normalized_variance = self._normalize_variance(variance_term)
            current_weight = self._compute_online_weight(
                normalized_variance=normalized_variance,
                workload_wci=workload_wci,
                complexity=complexity,
                microbatch_size=workload.shape[0],
            )
            online_epsilon = budget.online_optimal(
                remaining_budget=self.remaining_budget,
                current_weight=current_weight,
                remaining_microbatches=max(1, self.expected_dp_batches),
                expected_weight=self.expected_weight,
            )
            target_epsilon = budget.smart_min_epsilon(
                sensitivity=float(prepared.sensitivity),
                variance_term=variance_term,
                target_variance=self.smart_target_variance,
                remaining_budget=self.remaining_budget,
            )
            # Smart follows online_variance, but may save budget on an easy
            # microbatch when its target-variance epsilon is smaller.
            epsilon = min(online_epsilon, target_epsilon)
            if (
                self.expected_dp_batches > 0
                and self.policy in {"online_variance", "hdmm_online_variance"}
            ):
                epsilon = min(
                    epsilon,
                    self.remaining_budget / self.expected_dp_batches,
                )
            epsilon = min(
                max(epsilon, self.smart_min_epsilon),
                self.remaining_budget,
            )
            # If the remaining budget is too small for another minimum-sized
            # allocation, consume the complete remainder now rather than
            # leaving unused central-DP budget behind.
            if (
                self.smart_min_epsilon > 0.0
                and (
                    self.remaining_budget + EPSILON_TOL < self.smart_min_epsilon
                    or self.remaining_budget - epsilon < self.smart_min_epsilon
                )
            ):
                epsilon = self.remaining_budget
        else:
            epsilon = budget.allocate_budget(
                epsilon_dp=self.epsilon_dp,
                cp=self.cp,
                microbatch_size=workload.shape[0],
                wci=workload_wci,
                beta=self.beta,
                policy=self.policy,
                remaining_budget=self.remaining_budget,
            )

        epsilon = float(min(max(epsilon, 0.0),
                            self.remaining_budget)
        )

        #######################################################################
        # Fall back to LDP if no usable DP budget is available
        #######################################################################

        if epsilon <= EPSILON_TOL:
            self.remaining_budget = 0.0
            return self._answer_with_ldp(workload)

        #######################################################################
        # Prepare HDMM for baseline policies
        #######################################################################

        if prepared is None:
            prepared = self._prepare_strategy(workload=workload,
                                              dp_histogram=dp_histogram
            )

        if variance_term is None:
            variance_term = float(prepared.variance_term)

        #######################################################################
        # Execute the same strategy used to obtain the variance term
        #######################################################################

        hdmm_result = run_prepared_hdmm(
            prepared=prepared,
            histogram=dp_histogram,
            epsilon=epsilon,
            seed=self.seed,
        )

        n_dp = float(np.sum(dp_histogram))
        if n_dp <= 0.0:
            raise ValueError("The DP histogram must contain at least one user.")
        answers = np.asarray(hdmm_result.answers, dtype=float) / n_dp

        #######################################################################
        # Update remaining DP budget
        #######################################################################

        self.remaining_budget = max(
            0.0,
            self.remaining_budget - epsilon,
        )

        if self.remaining_budget <= EPSILON_TOL:
            self.remaining_budget = 0.0

        #######################################################################
        # Online allocation weight
        #######################################################################

        if (
            self.policy.startswith("online_")
            or self.policy in {"hdmm_online_variance", "smart", "hdmm_smart"}
        ):
            if normalized_variance is None:
                raise RuntimeError("The normalized variance term was not computed.")
            current_weight = self._compute_online_weight(
                normalized_variance=normalized_variance,
                workload_wci=workload_wci,
                complexity=complexity,
                microbatch_size=workload.shape[0],
            )
        else:
            current_weight = None

        #######################################################################
        # Per-microbatch error
        #######################################################################

        true_answers = np.asarray(workload @ (dp_histogram / n_dp), dtype=float)
        error = answers - true_answers
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))

        #######################################################################
        # Result
        #######################################################################

        raw_c = (
            float(prepared.estimated_C)
            if hasattr(prepared, "estimated_C")
            else None
        )

        return BatchResult(
            microbatch=0,
            mechanism="HDMM",
            epsilon=epsilon,
            remaining_budget=self.remaining_budget,
            answers=answers,
            wci=workload_wci,
            complexity=complexity,
            rank_factor=float(stats["rank_factor"]),
            density_factor=float(stats["density_factor"]),
            redundancy_factor=float(stats["redundancy_factor"]),
            effective_rank=float(stats["effective_rank"]),
            weight=current_weight,
            variance_term=variance_term,
            normalized_variance=normalized_variance,
            raw_C=raw_c,
            estimated_C=normalized_variance,
            sensitivity=float(prepared.sensitivity),
            lambda_min=float(prepared.lambda_min),
            sigma1=float(prepared.sigma1),
            mae=mae,
            rmse=rmse,
        )

    ###########################################################################
    # Expected-weight initialization
    ###########################################################################

    def _initialize_online_policy(
        self,
        microbatches: list[np.ndarray],
        dp_histogram: np.ndarray,
        prepared_hdmm_cache: dict[int, PreparedHDMM] | None = None,
    ) -> None:
        """Initialize variance statistics using only the DP portions."""
        dp_portions: list[tuple[int, np.ndarray]] = []
        processed_queries = 0

        for index, workload in enumerate(microbatches):
            remaining_dp_queries = self.cp - processed_queries
            if remaining_dp_queries <= 0:
                break

            dp_size = min(int(workload.shape[0]), remaining_dp_queries)
            if dp_size <= 0:
                continue

            dp_workload = workload[:dp_size]
            dp_portions.append((index, dp_workload))
            processed_queries += dp_size

        if not dp_portions:
            self.expected_weight = None
            self.variance_reference = None
            self.prepared_hdmm = {}
            return

        self.prepared_hdmm = {}
        variance_values: list[float] = []
        batch_statistics: dict[int, dict] = {}

        for index, workload in dp_portions:
            prepared = None
            if prepared_hdmm_cache is not None:
                prepared = prepared_hdmm_cache.get(index)

            if prepared is None:
                prepared = self._prepare_strategy(
                    workload=workload,
                    dp_histogram=dp_histogram,
                )
            elif (
                prepared.dense_workload.shape != workload.shape
                or not np.array_equal(prepared.dense_workload, workload)
            ):
                prepared = self._prepare_strategy(
                    workload=workload,
                    dp_histogram=dp_histogram,
                )

            variance_term = float(prepared.variance_term)
            if not np.isfinite(variance_term) or variance_term <= 0.0:
                raise ValueError(
                    "An invalid HDMM variance term was computed for "
                    f"microbatch {index + 1}: {variance_term}."
                )

            self.prepared_hdmm[index] = prepared
            batch_statistics[index] = wci.compute_complexity(workload)
            variance_values.append(variance_term)

        self.variance_reference = float(np.median(variance_values))
        if (
            not np.isfinite(self.variance_reference)
            or self.variance_reference <= 0.0
        ):
            raise ValueError(
                "The median variance reference is invalid: "
                f"{self.variance_reference}."
            )

        weights: list[float] = []
        for index, workload in dp_portions:
            prepared = self.prepared_hdmm[index]
            stats = batch_statistics[index]
            normalized_variance = self._normalize_variance(
                float(prepared.variance_term)
            )
            weight = self._compute_online_weight(
                normalized_variance=normalized_variance,
                workload_wci=float(stats["wci"]),
                complexity=float(stats["complexity"]),
                microbatch_size=int(workload.shape[0]),
            )
            weights.append(float(weight))

        self.expected_weight = float(np.mean(weights))

    ###########################################################################
    # Execute AdaHDP
    ###########################################################################

    def _expected_microbatch_size(self) -> float:
        """Return the configured expected size of a future microbatch."""
        configured = getattr(config, "EXPECTED_MICROBATCH_SIZE", None)
        if configured is not None:
            value = float(configured)
        elif hasattr(config, "MICROBATCH_SIZE_MIN") and hasattr(
            config,
            "MICROBATCH_SIZE_MAX",
        ):
            value = (
                float(config.MICROBATCH_SIZE_MIN)
                + float(config.MICROBATCH_SIZE_MAX)
            ) / 2.0
        else:
            value = float(config.MICROBATCH_SIZE)

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "The expected microbatch size must be finite and positive."
            )

        return value

    def _estimate_remaining_dp_batches(self, current_dp_size: int) -> int:
        """Estimate remaining DP batches, including the current batch."""
        remaining_after_current = max(
            0,
            self.cp - self.dp_queries_answered - current_dp_size,
        )
        if remaining_after_current == 0:
            return 1

        expected_batch_size = self._expected_microbatch_size()
        return 1 + math.ceil(remaining_after_current / expected_batch_size)

    def _run_ldp_only(self, microbatches: list[np.ndarray]) -> list[BatchResult]:
        """Answer every microbatch using LDP."""
        results: list[BatchResult] = []
        for index, workload in enumerate(microbatches):
            result = self._answer_with_ldp(workload)
            result.microbatch = index + 1
            results.append(result)
            if config.VERBOSE:
                self._print_result(result)
        return results

    def _run_dp_only(
            self,
            microbatches: list[np.ndarray],
            dp_histogram: np.ndarray,
            prepared_hdmm_cache: dict[int, PreparedHDMM] | None,
    ) -> list[BatchResult]:
        """Answer all microbatches using HDMM with uniform budget allocation."""
        if not microbatches:
            return []

        n_microbatches = len(microbatches)
        epsilon_per_batch = self.epsilon_dp / n_microbatches
        results: list[BatchResult] = []

        for index, workload in enumerate(microbatches):
            prepared = (
                prepared_hdmm_cache.get(index)
                if prepared_hdmm_cache is not None
                else None
            )

            if prepared is None or prepared.dense_workload.shape != workload.shape:
                prepared = self._prepare_strategy(
                    workload=workload,
                    dp_histogram=dp_histogram,
                )
            elif not np.array_equal(
                    prepared.dense_workload,
                    workload,
            ):
                prepared = self._prepare_strategy(
                    workload=workload,
                    dp_histogram=dp_histogram,
                )

            if index == n_microbatches - 1:
                epsilon = self.remaining_budget
            else:
                epsilon = min(
                    epsilon_per_batch,
                    self.remaining_budget,
                )

            result = self._execute_fixed_hdmm(
                workload=workload,
                dp_histogram=dp_histogram,
                prepared=prepared,
                epsilon=float(epsilon),
            )

            result.microbatch = index + 1
            results.append(result)

            if config.VERBOSE:
                self._print_result(result)

        return results

    def _sampling_error(
        self,
        workloads: list[np.ndarray],
        dp_histogram: np.ndarray,
        ldp_histogram: np.ndarray,
    ) -> float:
        """Estimate DP sampling error from LDP reconstruction with the FPC.

        For each normalized counting query q this computes

            Var_LDP_reconstruction[q] / N_DP * (1 - N_DP / N),

        then averages over all arriving queries.  The LDP reconstruction is
        used only to estimate the population distribution; its mechanism
        variance is accounted for separately by ``var_ldp``.
        """
        n_dp = float(np.sum(dp_histogram))
        if n_dp <= 0.0:
            raise ValueError("The DP histogram must contain at least one user.")

        n_ldp = float(np.sum(ldp_histogram))
        population_size = n_dp + n_ldp
        if population_size <= 0.0:
            raise ValueError("The population must contain at least one user.")
        if self.ldp_engine is None:
            raise RuntimeError("The LDP estimator has not been initialized.")

        return estimate_dp_sampling_error(
            workloads=workloads,
            reconstructed_population=self.ldp_engine.estimates,
            n_dp=int(n_dp),
            population_size=int(population_size),
        )

    def _ldp_workload_variance(self, workloads: list[np.ndarray]) -> float:
        """Lift the per-cell OUE variance to mean per-query variance."""
        if self.ldp_engine is None:
            raise RuntimeError("The LDP estimator has not been initialized.")
        squared_norms = np.concatenate(
            [np.sum(workload**2, axis=1) for workload in workloads]
        )
        return float(self.ldp_engine.variance() * np.mean(squared_norms))

    def _initialize_flat_crossover(
        self,
        microbatches: list[np.ndarray],
        dp_histogram: np.ndarray,
        ldp_histogram: np.ndarray,
        prepared_hdmm_cache: dict[int, PreparedHDMM] | None,
    ) -> int:
        """Prepare workload statistics and select the theorem's integer B_p."""
        expected_batch_count = (
            self.expected_microbatches
            if self.expected_microbatches is not None
            else len(microbatches)
        )
        estimation_microbatches = microbatches[:expected_batch_count]
        if not estimation_microbatches:
            raise ValueError("At least one microbatch is required for crossover estimation.")

        prepared_by_batch: dict[int, PreparedHDMM] = {}
        gamma_terms: list[float] = []
        for index, workload in enumerate(estimation_microbatches):
            prepared = (
                prepared_hdmm_cache.get(index)
                if prepared_hdmm_cache is not None
                else None
            )
            if prepared is None or (
                prepared.dense_workload.shape != workload.shape
                or not np.array_equal(prepared.dense_workload, workload)
            ):
                prepared = self._prepare_strategy(workload, dp_histogram)
            prepared_by_batch[index] = prepared
            gamma_terms.append(
                prepared.sensitivity**2
                * prepared.variance_term
                / int(workload.shape[0])
            )

        self.prepared_hdmm = prepared_by_batch
        n_dp = int(np.sum(dp_histogram))
        expected_query_count = (
            self.expected_queries
            if self.expected_queries is not None
            else int(round(expected_batch_count * self._expected_microbatch_size()))
            if self.expected_microbatches is not None
            else sum(int(batch.shape[0]) for batch in microbatches)
        )
        self.flat_crossover = optimal_hdmm_flat_crossover(
            n_dp=n_dp,
            epsilon_dp=self.epsilon_dp,
            expected_queries=expected_query_count,
            expected_microbatches=expected_batch_count,
            gamma_hdmm=float(np.mean(gamma_terms)),
            se_dp=self._sampling_error(
                estimation_microbatches,
                dp_histogram,
                ldp_histogram,
            ),
            var_ldp=self._ldp_workload_variance(estimation_microbatches),
        )
        self.cp = int(round(self.flat_crossover.dp_queries))
        return self.flat_crossover.dp_microbatches

    def _initialize_mae_crossover(
        self,
        microbatches: list[np.ndarray],
        dp_histogram: np.ndarray,
        ldp_histogram: np.ndarray,
        prepared_hdmm_cache: dict[int, PreparedHDMM] | None,
    ) -> int:
        """Select the discrete prefix minimizing expected query-weighted MAE."""
        expected_batch_count = (
            self.expected_microbatches
            if self.expected_microbatches is not None
            else len(microbatches)
        )
        estimation_microbatches = microbatches[:expected_batch_count]
        if not estimation_microbatches:
            raise ValueError("At least one microbatch is required for crossover estimation.")

        prepared_by_batch: dict[int, PreparedHDMM] = {}
        gamma_terms: list[float] = []
        for index, workload in enumerate(estimation_microbatches):
            prepared = (
                prepared_hdmm_cache.get(index)
                if prepared_hdmm_cache is not None
                else None
            )
            if prepared is None or (
                prepared.dense_workload.shape != workload.shape
                or not np.array_equal(prepared.dense_workload, workload)
            ):
                prepared = self._prepare_strategy(workload, dp_histogram)
            prepared_by_batch[index] = prepared
            gamma_terms.append(
                prepared.sensitivity**2
                * prepared.variance_term
                / int(workload.shape[0])
            )

        self.prepared_hdmm = prepared_by_batch
        self.flat_crossover = optimal_hdmm_mae_crossover(
            n_dp=int(np.sum(dp_histogram)),
            epsilon_dp=self.epsilon_dp,
            query_counts=[int(batch.shape[0]) for batch in estimation_microbatches],
            gamma_hdmm=gamma_terms,
            se_dp=self._sampling_error(
                estimation_microbatches,
                dp_histogram,
                ldp_histogram,
            ),
            var_ldp=self._ldp_workload_variance(estimation_microbatches),
        )
        self.cp = int(round(self.flat_crossover.dp_queries))
        return self.flat_crossover.dp_microbatches

    def _run_hdmm_flat(
        self,
        microbatches: list[np.ndarray],
        dp_histogram: np.ndarray,
        dp_batch_count: int,
    ) -> list[BatchResult]:
        """Answer the selected whole microbatches with equal DP budget."""
        if dp_batch_count <= 0:
            return self._run_ldp_only(microbatches)
        epsilon_per_batch = self.epsilon_dp / dp_batch_count
        results: list[BatchResult] = []
        for index, workload in enumerate(microbatches):
            if index < dp_batch_count:
                epsilon = (
                    self.remaining_budget
                    if index == dp_batch_count - 1
                    else min(epsilon_per_batch, self.remaining_budget)
                )
                result = self._execute_fixed_hdmm(
                    workload, dp_histogram, self.prepared_hdmm[index], epsilon
                )
                self.dp_queries_answered += int(workload.shape[0])
            else:
                result = self._answer_with_ldp(workload)
            result.microbatch = index + 1
            results.append(result)
            if config.VERBOSE:
                self._print_result(result)
        return results

    def run(
        self,
        microbatches: list[np.ndarray],
        dp_histogram: np.ndarray,
        ldp_histogram: np.ndarray,
        prepared_hdmm_cache: dict[int, PreparedHDMM] | None = None,
    ) -> list[BatchResult]:
        """
        Execute AdaHDP over variable-size microbatches.

        ``hdmm_flat`` computes the theorem-based crossover and switches only
        between whole microbatches. ``hdmm_online_variance`` uses that same
        whole-microbatch crossover, but allocates the DP budget online based
        on each microbatch's variance. Other hybrid policies retain the
        explicit ``cp`` query cutoff and may split the crossover microbatch.
        One result is returned for each original microbatch.
        """
        self.initialize(ldp_histogram)
        self.remaining_budget = float(self.epsilon_dp)
        self.dp_queries_answered = 0
        self.expected_dp_batches = 0
        self._online_dp_batch_count = None
        self._smart_min_dp_batches = None
        self.expected_weight = None
        self.variance_reference = None
        self.prepared_hdmm = {}
        self.flat_crossover = None

        if not microbatches:
            return []

        if self.policy == "ldp_only":
            return self._run_ldp_only(microbatches)

        if self.policy == "dp_only":
            return self._run_dp_only(
                microbatches=microbatches,
                dp_histogram=dp_histogram,
                prepared_hdmm_cache=prepared_hdmm_cache,
            )

        if self.policy == "hdmm_flat":
            if self.epsilon_dp <= EPSILON_TOL:
                return self._run_ldp_only(microbatches)
            dp_batch_count = self._initialize_flat_crossover(
                microbatches,
                dp_histogram,
                ldp_histogram,
                prepared_hdmm_cache,
            )
            return self._run_hdmm_flat(
                microbatches, dp_histogram, dp_batch_count
            )

        if self.policy == "hdmm_mae":
            if self.epsilon_dp <= EPSILON_TOL:
                return self._run_ldp_only(microbatches)
            dp_batch_count = self._initialize_mae_crossover(
                microbatches,
                dp_histogram,
                ldp_histogram,
                prepared_hdmm_cache,
            )
            return self._run_hdmm_flat(
                microbatches, dp_histogram, dp_batch_count
            )

        if self.policy in {"online_variance", "hdmm_online_variance"}:
            if self.epsilon_dp <= EPSILON_TOL:
                return self._run_ldp_only(microbatches)

            # Compute exactly the same theorem-based crossover as hdmm_flat.
            # The online policy then uses the selected prefix as its DP phase,
            # while answer_microbatch performs the variance-based allocation.
            dp_batch_count = self._initialize_flat_crossover(
                microbatches,
                dp_histogram,
                ldp_histogram,
                prepared_hdmm_cache,
            )
            self.cp = sum(
                int(workload.shape[0])
                for workload in microbatches[:dp_batch_count]
            )
            self._initialize_online_policy(
                microbatches=microbatches,
                dp_histogram=dp_histogram,
                prepared_hdmm_cache=prepared_hdmm_cache,
            )
            self._online_dp_batch_count = dp_batch_count

        elif self.policy in {"smart", "hdmm_smart"}:
            if self.epsilon_dp <= EPSILON_TOL:
                return self._run_ldp_only(microbatches)
            dp_batch_count = self._initialize_flat_crossover(
                microbatches,
                dp_histogram,
                ldp_histogram,
                prepared_hdmm_cache,
            )
            self._smart_min_dp_batches = dp_batch_count
            self.cp = sum(
                int(workload.shape[0])
                for workload in microbatches[:dp_batch_count]
            )
            self._initialize_online_policy(
                microbatches=microbatches,
                dp_histogram=dp_histogram,
                prepared_hdmm_cache=prepared_hdmm_cache,
            )
            # The crossover is a reference workload; smart may continue with
            # DP or exhaust the budget before reaching it.
            self.cp = sum(int(workload.shape[0]) for workload in microbatches)

        elif self.policy.startswith("online_"):
            self._initialize_online_policy(
                microbatches=microbatches,
                dp_histogram=dp_histogram,
                prepared_hdmm_cache=prepared_hdmm_cache,
            )
        elif prepared_hdmm_cache is not None:
            self.prepared_hdmm = dict(prepared_hdmm_cache)

        results: list[BatchResult] = []

        for index, workload in enumerate(microbatches):
            batch_size = int(workload.shape[0])
            if batch_size <= 0:
                raise ValueError(f"Microbatch {index + 1} contains no queries.")

            remaining_dp_queries = max(
                0,
                self.cp - self.dp_queries_answered,
            )

            if remaining_dp_queries <= 0 or self.remaining_budget <= EPSILON_TOL:
                result = self._answer_with_ldp(workload)
                result.microbatch = index + 1
                results.append(result)
                if config.VERBOSE:
                    self._print_result(result)
                continue

            dp_size = min(batch_size, remaining_dp_queries)
            dp_workload = workload[:dp_size]
            ldp_workload = workload[dp_size:] if dp_size < batch_size else None
            prepared = self.prepared_hdmm.get(index)

            if prepared is not None and (
                prepared.dense_workload.shape != dp_workload.shape
                or not np.array_equal(prepared.dense_workload, dp_workload)
            ):
                prepared = None

            if (
                self.policy in {
                    "online_variance",
                    "hdmm_online_variance",
                    "smart",
                    "hdmm_smart",
                }
                and self._online_dp_batch_count is not None
            ):
                # The shared flat crossover is a whole-microbatch boundary,
                # so reserve budget for each remaining selected DP batch.
                self.expected_dp_batches = max(
                    1, self._online_dp_batch_count - index
                )
            elif (
                self.policy in {"smart", "hdmm_smart"}
                and self._smart_min_dp_batches is not None
                and index < self._smart_min_dp_batches
            ):
                self.expected_dp_batches = self._smart_min_dp_batches - index
            elif self.policy in {"smart", "hdmm_smart"}:
                self.expected_dp_batches = 0
            else:
                self.expected_dp_batches = self._estimate_remaining_dp_batches(
                    current_dp_size=dp_size
                )

            dp_result = self.answer_microbatch(
                workload=dp_workload,
                dp_histogram=dp_histogram,
                prepared=prepared,
            )

            if dp_result.mechanism != "HDMM":
                dp_result.microbatch = index + 1
                results.append(dp_result)
                if config.VERBOSE:
                    self._print_result(dp_result)
                continue

            self.dp_queries_answered += dp_size

            if ldp_workload is None or ldp_workload.shape[0] == 0:
                dp_result.microbatch = index + 1
                results.append(dp_result)
                if config.VERBOSE:
                    self._print_result(dp_result)
                continue

            ldp_result = self._answer_with_ldp(ldp_workload)
            combined_answers = np.concatenate(
                (
                    np.asarray(dp_result.answers, dtype=float),
                    np.asarray(ldp_result.answers, dtype=float),
                )
            )

            combined_result = BatchResult(
                microbatch=index + 1,
                mechanism="HDMM",
                epsilon=dp_result.epsilon,
                remaining_budget=self.remaining_budget,
                answers=combined_answers,
                wci=dp_result.wci,
                complexity=dp_result.complexity,
                rank_factor=dp_result.rank_factor,
                density_factor=dp_result.density_factor,
                redundancy_factor=dp_result.redundancy_factor,
                effective_rank=dp_result.effective_rank,
                weight=dp_result.weight,
                variance_term=dp_result.variance_term,
                normalized_variance=dp_result.normalized_variance,
                raw_C=dp_result.raw_C,
                estimated_C=dp_result.estimated_C,
                sensitivity=dp_result.sensitivity,
                lambda_min=dp_result.lambda_min,
                sigma1=dp_result.sigma1,
            )
            results.append(combined_result)

            if config.VERBOSE:
                self._print_result(combined_result)

        return results

    def _execute_fixed_hdmm(
            self,
            workload: np.ndarray,
            dp_histogram: np.ndarray,
            prepared: PreparedHDMM,
            epsilon: float,
    ) -> BatchResult:
        """Execute HDMM using a predetermined privacy budget."""
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        stats = wci.compute_complexity(workload)

        hdmm_result = run_prepared_hdmm(
            prepared=prepared,
            histogram=dp_histogram,
            epsilon=epsilon,
            seed=self.seed,
        )

        n_dp = float(np.sum(dp_histogram))
        if n_dp <= 0.0:
            raise ValueError("The DP histogram must contain at least one user.")
        answers = np.asarray(hdmm_result.answers, dtype=float) / n_dp
        true_answers = np.asarray(workload @ (dp_histogram / n_dp), dtype=float)
        error = answers - true_answers

        self.remaining_budget = max(
            0.0,
            self.remaining_budget - epsilon,
        )

        if self.remaining_budget <= EPSILON_TOL:
            self.remaining_budget = 0.0

        return BatchResult(
            microbatch=0,
            mechanism="HDMM",
            epsilon=epsilon,
            remaining_budget=self.remaining_budget,
            answers=answers,
            wci=float(stats["wci"]),
            complexity=float(stats["complexity"]),
            rank_factor=float(stats["rank_factor"]),
            density_factor=float(stats["density_factor"]),
            redundancy_factor=float(stats["redundancy_factor"]),
            effective_rank=float(stats["effective_rank"]),
            variance_term=float(prepared.variance_term),
            normalized_variance=None,
            raw_C=float(prepared.estimated_C),
            estimated_C=None,
            sensitivity=float(prepared.sensitivity),
            lambda_min=float(prepared.lambda_min),
            sigma1=float(prepared.sigma1),
            mae=float(np.mean(np.abs(error))),
            rmse=float(np.sqrt(np.mean(error ** 2))),
        )

    ###########################################################################
    # Output
    ###########################################################################

    @staticmethod
    def _format_optional(value: float | None, precision: int = 3) -> str:
        """
        Format an optional floating-point value.
        """

        if value is None or not np.isfinite(value):
            return "-"

        return f"{value:.{precision}f}"

    def _print_result(self, result: BatchResult) -> None:
        """
        Print one microbatch result.
        """

        if not config.SHOW_SCHEDULER_PROGRESS:
            return

        wci_text = self._format_optional(result.wci)
        weight_text = self._format_optional(result.weight)

        print(
            f"Batch {result.microbatch:3d} | "
            f"{result.mechanism:4s} | "
            f"WCI={wci_text:>5s} | "
            f"eps={result.epsilon:.6f} | "
            f"remaining={result.remaining_budget:.6f} | "
            f"W={weight_text:>5s}"
        )

        if result.mechanism != "HDMM":
            return

        print(
            f"WCI={self._format_optional(result.wci)} | "
            f"Complexity="
            f"{self._format_optional(result.complexity)} | "
            f"Variance="
            f"{self._format_optional(result.variance_term, precision=6)} | "
            f"NormVariance="
            f"{self._format_optional(result.normalized_variance)} | "
            f"RawC="
            f"{self._format_optional(result.raw_C)} | "
            f"Delta="
            f"{self._format_optional(result.sensitivity)} | "
            f"LambdaMin="
            f"{self._format_optional(result.lambda_min, precision=6)}"
        )

    ###########################################################################
    # Statistics
    ###########################################################################

    @property
    def consumed_budget(self) -> float:
        """
        Return the total consumed DP privacy budget.
        """
        return float(self.epsilon_dp - self.remaining_budget)

    @property
    def budget_utilization(self) -> float:
        """
        Return the fraction of the DP budget consumed.
        """

        if self.epsilon_dp <= 0.0:
            return 0.0

        return float(self.consumed_budget / self.epsilon_dp)

"""Point-query crossover experiment used by Fig. 4 and Fig. 6."""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pandas as pd

import config
import baseline_budget_exhaustion_experiment as baseline
import adahdp as flat
from dataset import dataset_domain_size, get_dataset_spec, load_data
from evaluation import evaluate_experiment
from scheduler import AdaHDP
from crossover import optimal_hdmm_flat_crossover


def theorem_mae_crossover(*, n_dp, epsilon_dp, query_counts, gamma_hdmm, se_dp, var_ldp):
    counts = np.asarray(query_counts, dtype=float)
    gamma = np.asarray(gamma_hdmm, dtype=float)
    if gamma.ndim == 0:
        gamma = np.full(counts.size, float(gamma))
    sampling = np.asarray(se_dp, dtype=float)
    if sampling.ndim == 0:
        sampling = np.full(counts.size, float(sampling))
    ldp = np.asarray(var_ldp, dtype=float)
    if ldp.ndim == 0:
        ldp = np.full(counts.size, float(ldp))
    factor = np.sqrt(2.0 / np.pi)
    total = float(np.sum(counts))
    objectives = []
    for prefix in range(counts.size + 1):
        dp_error = factor * np.sqrt(
            sampling[:prefix]
            + 2.0 * gamma[:prefix] * (prefix / (float(n_dp) * float(epsilon_dp))) ** 2
        )
        ldp_error = factor * np.sqrt(ldp[prefix:])
        objectives.append(float(
            (np.dot(counts[:prefix], dp_error) + np.dot(counts[prefix:], ldp_error)) / total
        ))
    best = int(np.argmin(objectives))
    return type("Crossover", (), {"dp_microbatches": best, "objectives": tuple(objectives)})()


class ForcedOUEMAECrossover(AdaHDP):
    """Apply AdaHDP's point-query analysis while forcing a candidate prefix."""
    def __init__(self, *, forced_prefix: int, **kwargs):
        super().__init__(**kwargs)
        self.forced_prefix = forced_prefix
        self.analytical_prefix = None
        self.mae_objectives = None
        self.mae_phase_errors = None
        self.mse_objectives = None
        self.mse_phase_errors = None
        self.mse_crossover = None

    def _initialize_mae_crossover(self, microbatches, dp_histogram, ldp_histogram, prepared_hdmm_cache):
        selected = super()._initialize_mae_crossover(
            microbatches, dp_histogram, ldp_histogram, prepared_hdmm_cache
        )
        self.analytical_prefix = int(selected)
        count = self.expected_microbatches or len(microbatches)
        batches = microbatches[:count]
        query_counts = np.asarray([len(batch) for batch in batches], dtype=float)
        gamma_terms = np.asarray([
            self.prepared_hdmm[i].sensitivity ** 2 * self.prepared_hdmm[i].variance_term / len(batch)
            for i, batch in enumerate(batches)
        ], dtype=float)
        sampling = float(self._sampling_error(batches, dp_histogram, ldp_histogram))
        ldp_variance = float(self._ldp_workload_variance(batches))
        n_dp_epsilon = float(np.sum(dp_histogram)) * self.epsilon_dp
        result = theorem_mae_crossover(
            n_dp=int(np.sum(dp_histogram)), epsilon_dp=self.epsilon_dp,
            query_counts=query_counts, gamma_hdmm=gamma_terms,
            se_dp=sampling, var_ldp=ldp_variance,
        )
        self.analytical_prefix = int(result.dp_microbatches)
        self.mae_objectives = result.objectives
        factor = np.sqrt(2.0 / np.pi)
        total_queries = float(np.sum(query_counts))
        self.mae_phase_errors = []
        self.mse_phase_errors = []
        mse_objectives = []
        for prefix in range(count + 1):
            dp_queries = float(np.sum(query_counts[:prefix]))
            ldp_queries = float(np.sum(query_counts[prefix:]))
            variances = sampling + 2.0 * gamma_terms[:prefix] * (prefix / n_dp_epsilon) ** 2
            dp_mae_sum = float(np.sum(query_counts[:prefix] * factor * np.sqrt(variances)))
            ldp_mae_sum = ldp_queries * factor * np.sqrt(ldp_variance)
            dp_mse_sum = float(np.dot(query_counts[:prefix], variances))
            ldp_mse_sum = ldp_queries * ldp_variance
            mse_objectives.append((dp_mse_sum + ldp_mse_sum) / total_queries)
            self.mae_phase_errors.append({
                "theoretical_dp_phase_mae": dp_mae_sum / dp_queries if dp_queries else np.nan,
                "theoretical_ldp_phase_mae": ldp_mae_sum / ldp_queries if ldp_queries else np.nan,
                "theoretical_total_mae": (dp_mae_sum + ldp_mae_sum) / total_queries,
            })
            self.mse_phase_errors.append({
                "theoretical_dp_phase_mse": dp_mse_sum / dp_queries if dp_queries else np.nan,
                "theoretical_ldp_phase_mse": ldp_mse_sum / ldp_queries if ldp_queries else np.nan,
                "theoretical_total_mse": (dp_mse_sum + ldp_mse_sum) / total_queries,
            })
        self.mse_objectives = tuple(mse_objectives)
        self.mse_crossover = int(np.argmin(mse_objectives))
        return self.forced_prefix


def run_experiment(
    dataset: str, *, runs: int, total_microbatches: int,
    expected_microbatches: int, n_records: int, seed: int,
    query_type: str = "point", report_metric: str = "mse",
    fixed_workload: bool = False, progress_callback: Callable[[str], None] | None = None,
    crossover_min: int = 0, crossover_max: int | None = None, crossover_step: int = 1,
) -> pd.DataFrame:
    if query_type != "point":
        raise ValueError("This minimal copy supports point queries only.")
    if report_metric not in {"mae", "mse"}:
        raise ValueError("report_metric must be 'mae' or 'mse'.")
    crossover_max = expected_microbatches if crossover_max is None else crossover_max
    if crossover_min < 0 or crossover_max > expected_microbatches or crossover_min > crossover_max or crossover_step <= 0:
        raise ValueError("Invalid crossover sweep bounds.")
    candidates = list(range(crossover_min, crossover_max + 1, crossover_step))
    domain_size = dataset_domain_size(dataset)
    spec = get_dataset_spec(dataset)
    settings = baseline.ExperimentSettings(
        total_microbatches=total_microbatches,
        expected_microbatches=expected_microbatches, runs=runs, seed=seed,
    )
    rows = []
    fixed_cache = None
    for run_index in range(runs):
        report_seed = seed + run_index
        workload_seed = seed if fixed_workload else report_seed
        if progress_callback:
            progress_callback(f"{dataset} | run {run_index + 1}/{runs} | loading data")
        _, _, dp_hist, ldp_hist = load_data(n_records, config.DP_RATIO, workload_seed, dataset)
        batches = flat.generate_point_query_microbatches(domain_size, settings, workload_seed)
        cache_batches = batches[:expected_microbatches]
        cache = fixed_cache if fixed_workload and fixed_cache is not None else baseline.prepare_shared_hdmm_strategies(
            cache_batches, domain_size, workload_seed
        )
        if fixed_workload:
            fixed_cache = cache
        expected_queries = int(round(expected_microbatches * baseline.expected_microbatch_size()))
        population = baseline.normalized_population_histogram(dp_hist, ldp_hist)
        options = dict(
            epsilon_dp=config.EPS_DP, epsilon_ldp=config.EPS_LDP, beta=config.BETA,
            seed=report_seed, cp=expected_queries, policy="hdmm_mae",
            expected_microbatches=expected_microbatches, expected_queries=expected_queries,
        )
        np.random.seed(report_seed)
        selector = ForcedOUEMAECrossover(forced_prefix=0, **options)
        selector.run(microbatches=batches, dp_histogram=dp_hist, ldp_histogram=ldp_hist,
                     prepared_hdmm_cache=copy.deepcopy(cache))
        run_rows = []
        empirical_mses = []
        for prefix in candidates:
            np.random.seed(report_seed)
            scheduler = ForcedOUEMAECrossover(forced_prefix=prefix, **options)
            results = scheduler.run(microbatches=batches, dp_histogram=dp_hist,
                                    ldp_histogram=ldp_hist,
                                    prepared_hdmm_cache=copy.deepcopy(cache))
            metrics = evaluate_experiment(batches, results, population)
            weights = np.asarray([metric.n_queries for metric in metrics], dtype=float)
            empirical_mse = float(np.average([metric.mse for metric in metrics], weights=weights))
            empirical_mses.append(empirical_mse)
            row = {
                "dataset": dataset, "query_type": "point", "run": run_index + 1,
                "seed": report_seed, "workload_seed": workload_seed,
                "dp_prefix_microbatches": prefix,
                "analytical_crossover": scheduler.analytical_prefix,
                "mse_crossover": scheduler.mse_crossover,
                "empirical_mse_crossover": None,
                "analytical_mse": scheduler.mse_objectives[prefix],
                **scheduler.mse_phase_errors[prefix],
                "empirical_mse": empirical_mse,
                "dp_only_flat_mae": np.nan,
            }
            run_rows.append(row)
        empirical_optimum = candidates[int(np.argmin(empirical_mses))]
        for row in run_rows:
            row["empirical_mse_crossover"] = empirical_optimum
        rows.extend(run_rows)
    return pd.DataFrame(rows)

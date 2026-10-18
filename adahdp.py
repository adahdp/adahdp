"""Compare DP-only baselines and AdaHDP across supported datasets.

The experiment compares:

* DP-only-Flat: every microbatch is answered by HDMM with a uniform budget;
* DP-only-Geometric: every microbatch is answered by HDMM with a geometric budget;
* LDP-only: every microbatch is answered by OUE;
* AdaHDP: the discrete MSE-minimizing DP prefix with uniform allocation, followed by LDP.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import config
from dataset import dataset_domain_size, get_dataset_spec, load_data
from evaluation import evaluate_experiment
from hdmm import PreparedHDMM
from scheduler import AdaHDP
from workload import microbatches, point_query

import baseline_budget_exhaustion_experiment as baseline


OUTPUT_CSV = "flat_geometric_dataset_results.csv"
OUTPUT_PREFIX = "flat_geometric_comparison"
SUPPORTED_DATASETS = ("adult",)
# Choose datasets here; the --datasets option can override this selection.
DATASETS_TO_RUN = ("adult",)
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
# Number of microbatches shown on the x-axis for each dataset.
DATASET_MICROBATCHES: dict[str, int] = {
    "adult": 30,
}
METHODS = ("DP-only-Flat", "DP-only-Geometric", "LDP-only", "AdaHDP")
METHOD_STYLE = {
    "DP-only-Flat": {"color": "#268F4B", "marker": "o", "linestyle": "-"},
    "DP-only-Geometric": {"color": "#347FC4", "marker": "s", "linestyle": "-"},
    "LDP-only": {"color": "#8A2BE2", "marker": "^", "linestyle": "-"},
    "AdaHDP": {"color": "#CF2E3E", "marker": "D", "linestyle": "-"},
}
MARKER_STEP = 1


@dataclass(frozen=True)
class ExperimentSettings:
    total_microbatches: int
    expected_microbatches: int
    runs: int
    n_records: int
    epsilon_dp: float = config.EPS_DP
    epsilon_ldp: float = config.EPS_LDP
    beta: float = config.BETA
    seed: int = config.SEED

    def __post_init__(self) -> None:
        if self.total_microbatches <= 0 or self.runs <= 0 or self.n_records <= 0:
            raise ValueError("Microbatch, run, and record counts must be positive.")
        if not 0 < self.expected_microbatches <= self.total_microbatches:
            raise ValueError("expected_microbatches must lie within the stream.")
        if self.epsilon_dp <= 0.0 or self.epsilon_ldp <= 0.0:
            raise ValueError("Privacy budgets must be positive.")
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie strictly between 0 and 1.")


def workload_settings(settings: ExperimentSettings) -> baseline.ExperimentSettings:
    """Adapt this experiment's settings to the shared workload generator."""
    return baseline.ExperimentSettings(
        total_microbatches=settings.total_microbatches,
        expected_microbatches=settings.expected_microbatches,
        runs=settings.runs,
        epsilon_dp=settings.epsilon_dp,
        epsilon_ldp=settings.epsilon_ldp,
        seed=settings.seed,
    )


def generate_point_query_microbatches(
    domain_size: int, settings: baseline.ExperimentSettings, seed: int
) -> list[np.ndarray]:
    """Generate point-query-only microbatches, independent of QUERY_TYPE config."""
    rng = np.random.default_rng(seed)
    if config.VARIABLE_MICROBATCH_SIZE:
        n_queries = settings.total_microbatches * config.MICROBATCH_SIZE_MAX
    else:
        n_queries = settings.total_microbatches * config.MICROBATCH_SIZE
    workload = np.vstack(
        [point_query(domain_size, rng) for _ in range(n_queries)]
    )
    if config.VARIABLE_MICROBATCH_SIZE:
        batches = microbatches(
            workload,
            batch_size=config.MICROBATCH_SIZE,
            rng=rng,
            min_batch_size=config.MICROBATCH_SIZE_MIN,
            max_batch_size=config.MICROBATCH_SIZE_MAX,
        )
    else:
        batches = microbatches(workload, batch_size=config.MICROBATCH_SIZE)
    return batches[: settings.total_microbatches]


def build_scheduler(
    method: str,
    batches: list[np.ndarray],
    settings: ExperimentSettings,
    seed: int,
) -> AdaHDP:
    common = {
        "epsilon_dp": settings.epsilon_dp,
        "epsilon_ldp": settings.epsilon_ldp,
        "beta": settings.beta,
        "seed": seed,
    }
    if method == "DP-only-Flat":
        return AdaHDP(
            **common,
            cp=sum(int(batch.shape[0]) for batch in batches),
            policy="dp_only",
        )
    if method == "DP-only-Geometric":
        return AdaHDP(
            **common,
            cp=sum(int(batch.shape[0]) for batch in batches),
            policy="dp_geometric",
        )
    if method == "LDP-only":
        return AdaHDP(**common, cp=0, policy="ldp_only")
    raise ValueError(f"Unknown method: {method!r}.")


def run_method(
    *,
    method: str,
    batches: list[np.ndarray],
    dp_histogram: np.ndarray,
    ldp_histogram: np.ndarray,
    prepared_cache: dict[int, PreparedHDMM],
    settings: ExperimentSettings,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    crossover = None
    if method == "AdaHDP":
        # Use error_analysis's forced-prefix MSE analysis to select the DP prefix.
        from query_crossover_validation import ForcedOUEMAECrossover

        expected_queries = int(round(
            settings.expected_microbatches * baseline.expected_microbatch_size()
        ))
        scheduler_options = {
            "epsilon_dp": settings.epsilon_dp,
            "epsilon_ldp": settings.epsilon_ldp,
            "beta": settings.beta,
            "seed": seed,
            "cp": expected_queries,
            "policy": "hdmm_mae",
            "expected_microbatches": settings.expected_microbatches,
            "expected_queries": expected_queries,
        }
        # The helper computes the MSE objective for every candidate prefix as
        # part of its initialization, regardless of which prefix is forced.
        selector = ForcedOUEMAECrossover(forced_prefix=0, **scheduler_options)
        np.random.seed(seed)
        selector.run(
            microbatches=batches,
            dp_histogram=dp_histogram,
            ldp_histogram=ldp_histogram,
            prepared_hdmm_cache=copy.deepcopy(prepared_cache),
        )
        crossover = int(selector.mse_crossover)
        scheduler = ForcedOUEMAECrossover(
            forced_prefix=crossover, **scheduler_options
        )
    else:
        scheduler = build_scheduler(method, batches, settings, seed)

    np.random.seed(seed)
    results = scheduler.run(
        microbatches=batches,
        dp_histogram=dp_histogram,
        ldp_histogram=ldp_histogram,
        prepared_hdmm_cache=copy.deepcopy(prepared_cache),
    )
    population = baseline.normalized_population_histogram(
        dp_histogram, ldp_histogram
    )
    metrics = evaluate_experiment(
        microbatches=batches,
        results=results,
        histogram=population,
    )
    mse = np.asarray([metric.mse for metric in metrics], dtype=float)
    epsilon = np.asarray(
        [
            float(result.epsilon) if result.mechanism == "HDMM" else np.nan
            for result in results
        ],
        dtype=float,
    )
    return mse, epsilon, crossover


def run_dataset(
    dataset_name: str,
    settings: ExperimentSettings,
) -> pd.DataFrame:
    spec = get_dataset_spec(dataset_name)
    domain_size = dataset_domain_size(dataset_name)
    print(
        f"Dataset={dataset_name} | attributes={spec['attributes']} | "
        f"domains={spec['attribute_domains']} | domain size={domain_size}"
    )

    # Keep the data split and query stream fixed across realizations, as in
    # error_analysis's fixed_workload mode. Only report_seed varies per realization.
    workload_seed = settings.seed
    _, _, dp_histogram, ldp_histogram = load_data(
        n_records=settings.n_records,
        dp_ratio=config.DP_RATIO,
        seed=workload_seed,
        dataset_name=dataset_name,
    )
    dp_histogram = np.asarray(dp_histogram, dtype=float)
    ldp_histogram = np.asarray(ldp_histogram, dtype=float)
    batches = generate_point_query_microbatches(
        domain_size, workload_settings(settings), workload_seed
    )
    # Prepare once and reuse the same HDMM strategies for every method and
    # realization on this fixed workload.
    prepared_cache = baseline.prepare_shared_hdmm_strategies(
        batches, domain_size, workload_seed
    )

    frames: list[pd.DataFrame] = []
    for run_index in range(settings.runs):
        report_seed = settings.seed + run_index
        print(
            f"Run {run_index + 1}/{settings.runs} | "
            f"workload seed={workload_seed} | report seed={report_seed}"
        )
        for method in METHODS:
            mse, epsilon, crossover = run_method(
                method=method,
                batches=batches,
                dp_histogram=dp_histogram,
                ldp_histogram=ldp_histogram,
                prepared_cache=prepared_cache,
                settings=settings,
                seed=report_seed,
            )
            cumulative_mse = np.cumsum(mse)
            crossover_text = (
                f" | crossover={crossover}" if crossover is not None else ""
            )
            print(
                f"  {method:20s} | final cumulative MSE="
                f"{cumulative_mse[-1]:.6f}{crossover_text}"
            )
            frames.append(pd.DataFrame({
                "dataset": dataset_name,
                "run": run_index + 1,
                "seed": report_seed,
                "workload_seed": workload_seed,
                "method": method,
                "microbatch": np.arange(1, len(batches) + 1),
                "mse": mse,
                "cumulative_mse": cumulative_mse,
                "dp_epsilon": epsilon,
                "crossover_microbatch": crossover,
                "attributes": ",".join(spec["attributes"]),
                "domain_size": domain_size,
            }))
    return pd.concat(frames, ignore_index=True)


def mean_crossover(results: pd.DataFrame, method: str) -> float:
    values = (
        results.loc[results.method == method, ["run", "crossover_microbatch"]]
        .drop_duplicates(subset="run")
        ["crossover_microbatch"]
        .dropna()
    )
    if values.empty:
        raise ValueError(f"No crossover values recorded for {method}.")
    return float(values.mean())


def format_crossover(value: float) -> str:
    return str(int(round(value))) if np.isclose(value, round(value)) else f"{value:.1f}"


def horizontal_legend_handles() -> list[Line2D]:
    """Build the one-row legend swatches used by the comparison figures."""
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linewidth=7,
            marker=METHOD_STYLE[method]["marker"],
            markersize=36,
            markerfacecolor="none",
            markeredgewidth=3,
            linestyle=METHOD_STYLE[method]["linestyle"],
            label=method,
        )
        for method in METHODS
    ]


def save_horizontal_legend(output_path: Path, *, fontsize: int = 50) -> None:
    """Save a standalone horizontal legend strip, similar to ``legends_pq.png``."""
    figure = plt.figure(figsize=(36, 2.5))
    figure.legend(
        handles=horizontal_legend_handles(),
        loc="center",
        ncol=len(METHODS),
        frameon=False,
        handlelength=2.8,
        handletextpad=0.7,
        columnspacing=2.0,
        prop={"size": fontsize},
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(figure)


def plot_dataset(
    results: pd.DataFrame,
    settings: ExperimentSettings,
    output_path: Path,
    *,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    x = np.arange(1, settings.total_microbatches + 1)
    cumulative_mean: dict[str, np.ndarray] = {}
    cumulative_std: dict[str, np.ndarray] = {}
    for method in METHODS:
        data = results[results.method == method].pivot(
            index="run", columns="microbatch", values="mse"
        ).reindex(columns=range(1, settings.total_microbatches + 1))
        # Accumulate within each run first, then summarize across runs so the
        # band reflects run-to-run uncertainty in the cumulative metric.
        run_cumulative = data.cumsum(axis=1)
        cumulative_mean[method] = run_cumulative.mean(axis=0).to_numpy(dtype=float)
        cumulative_std[method] = run_cumulative.std(
            axis=0, ddof=1 if len(run_cumulative) > 1 else 0
        ).fillna(0.0).to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(16, 10.5))
    for method, mean_curve in cumulative_mean.items():
        style = METHOD_STYLE[method]
        axis.plot(
            x,
            mean_curve,
            label=method,
            linewidth=7,
            marker=style["marker"],
            markevery=MARKER_STEP,
            markersize=36,
            markerfacecolor="none",
            markeredgecolor=style["color"],
            markeredgewidth=3,
            color=style["color"],
            linestyle=style["linestyle"],
        )
        spread = cumulative_std[method]
        axis.fill_between(
            x,
            np.maximum(0.0, mean_curve - spread),
            mean_curve + spread,
            color=style["color"],
            alpha=0.18,
            linewidth=0,
        )

    cp_avg = mean_crossover(results, "AdaHDP")
    if y_max is None:
        visible_methods = [
            method for method in METHODS if method != "DP-only-Geometric"
        ]
        visible_max = max(
            float(np.nanmax(cumulative_mean[method] + cumulative_std[method]))
            for method in visible_methods
        )
        y_max = visible_max * 1.05
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min.")
    axis.set_ylim(bottom=y_min, top=y_max)
    axis.axvline(cp_avg, linestyle="--", linewidth=6.5, color="black")

    offset = 0.8
    axis.text(
        cp_avg + offset,
        axis.get_ylim()[1] * 0.8,
        rf"$c_p={format_crossover(cp_avg)}$",
        rotation=90,
        va="top",
        fontsize=60,
    )

    axis.set_xlabel("Microbatch Index", fontsize=60)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.set_ylabel("Cumulative MSE", fontsize=60)
    axis.tick_params(axis="both", which="major", labelsize=48)
    axis.yaxis.get_offset_text().set_fontsize(48)
    axis.grid(alpha=0.55)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def normalize_dataset_names(names: list[str]) -> list[str]:
    normalized = [name.lower() for name in names]
    if "all" in normalized:
        return list(SUPPORTED_DATASETS)
    unknown = sorted(set(normalized) - set(SUPPORTED_DATASETS))
    if unknown:
        raise ValueError(
            f"Unknown datasets {unknown}; choose from {SUPPORTED_DATASETS} or 'all'."
        )
    return list(dict.fromkeys(normalized))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS_TO_RUN),
        help="Dataset to run: adult (or all).",
    )
    parser.add_argument("--runs", type=int, default=config.N_RUNS)
    parser.add_argument(
        "--total-microbatches",
        type=int,
        default=None,
        help="Override the per-dataset DATASET_MICROBATCHES setting.",
    )
    parser.add_argument("--expected-microbatches", type=int)
    parser.add_argument(
        "--n-records",
        type=int,
        default=None,
        help=(
            "Override the dataset-specific record count for every dataset; "
            "by default, use config.DATASET_N_RECORDS."
        ),
    )
    parser.add_argument("--beta", type=float, default=config.BETA)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional y-axis upper bound; by default exclude DP-only-Geometric.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    args = parser.parse_args()

    dataset_names = normalize_dataset_names(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / Path(args.output_csv).name
    output_prefix = Path(args.output_prefix).name
    dataset_label = "_".join(dataset_names)
    output_csv = output_csv.with_name(
        f"{output_csv.stem}_{dataset_label}{output_csv.suffix}"
    )
    frames: list[pd.DataFrame] = []
    for dataset_name in dataset_names:
        n_records = (
            args.n_records
            if args.n_records is not None
            else config.DATASET_N_RECORDS[dataset_name]
        )
        total_microbatches = (
            args.total_microbatches
            if args.total_microbatches is not None
            else DATASET_MICROBATCHES[dataset_name]
        )
        expected_microbatches = (
            args.expected_microbatches
            if args.expected_microbatches is not None
            else total_microbatches
        )
        settings = ExperimentSettings(
            total_microbatches=total_microbatches,
            expected_microbatches=expected_microbatches,
            runs=args.runs,
            n_records=n_records,
            beta=args.beta,
            seed=args.seed,
        )
        print(f"Using {settings.n_records:,} records for dataset={dataset_name}")
        dataset_results = run_dataset(dataset_name, settings)
        frames.append(dataset_results)
        output_path = args.output_dir / f"{output_prefix}_{dataset_name}.pdf"
        plot_dataset(
            dataset_results,
            settings,
            output_path,
            y_min=args.y_min,
            y_max=args.y_max,
        )
        print(f"Saved figure to: {output_path}")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

    legend_path = args.output_dir / f"{output_prefix}_legend.png"
    save_horizontal_legend(legend_path)
    print(f"Saved horizontal legend to: {legend_path}")


if __name__ == "__main__":
    main()

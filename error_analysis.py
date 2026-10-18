"""Generate MSE versions of the four point-query crossover plots in Fig. 4.

The experiment reuses ``query_crossover_validation.run_experiment`` for
the query stream, forced crossover sweep, empirical reports, and analytical
MSE model. For every dataset the script saves a crossover curve PDF, one
shared legend, and a combined CSV with all per-run sweep results.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import query_crossover_validation as crossover_experiment

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASETS = ("adult",)
MARKER_STEP = 2
# Toggle the verbose per-crossover c_p MSE lines from the underlying runner.
SHOW_CROSSOVER_DETAILS = False


def report_progress(message: str) -> None:
    """Print stage progress even when detailed crossover output is hidden."""
    print(f"[progress] {message}", file=sys.stderr, flush=True)

# Edit these per-dataset settings to control each crossover sweep.
DATASET_MICROBATCHES = {
    "adult": 30,
}
DATASET_CROSSOVER_STEP = {
    "adult": 1,
}

SERIES = {
    "analytical_mse": ("Total analytical MSE", "#6A35FF", "o"),
    "theoretical_dp_phase_mse": ("DP-phase MSE", "#0077E6", "s"),
    "theoretical_ldp_phase_mse": ("LDP-phase MSE", "#F04400", "^"),
    "empirical_mse": ("Empirical MSE", "#00A878", "D"),
}


def _mean_and_standard_error(values: pd.Series) -> tuple[float, float]:
    observations = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if observations.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(observations))
    standard_error = (
        float(np.std(observations, ddof=1) / np.sqrt(observations.size))
        if observations.size > 1 else 0.0
    )
    return mean, standard_error


def plot_dataset(frame: pd.DataFrame, dataset: str, output_dir: Path) -> tuple[Path, list, list]:
    """Plot mean analytical and empirical MSE curves for one dataset."""
    required = {"dp_prefix_microbatches", *SERIES, "empirical_mse_crossover"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Experiment results for {dataset} lack columns: "
            f"{', '.join(sorted(missing))}"
        )

    summary = frame.groupby("dp_prefix_microbatches", sort=True)[
        list(SERIES)
    ].mean()
    x = summary.index.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(16, 10.5))
    for column, (label, color, marker) in SERIES.items():
        ax.plot(
            x,
            summary[column].to_numpy(float),
            marker=marker,
            markevery=MARKER_STEP,
            markersize=36,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=3,
            linewidth=7,
            color=color,
            label=label,
        )

    analytical_optimum = int(summary["analytical_mse"].idxmin())
    empirical_optimum, empirical_optimum_se = _mean_and_standard_error(
        frame.drop_duplicates("run")["empirical_mse_crossover"]
    )
    ax.axvline(
        analytical_optimum,
        color=SERIES["analytical_mse"][1],
        linestyle=":",
        linewidth=6.5,
        label="Optimal analytical MSE crossover",
    )
    ax.axvline(
        empirical_optimum,
        color=SERIES["empirical_mse"][1],
        linestyle="--",
        linewidth=6.5,
        label="Mean empirical MSE crossover",
    )

    y_values = np.concatenate([summary[column].to_numpy(float) for column in SERIES])
    y_values = y_values[np.isfinite(y_values)]
    y_anchor = float(np.min(y_values) + 0.06 * (np.max(y_values) - np.min(y_values)))
    ax.errorbar(
        empirical_optimum,
        y_anchor,
        xerr=empirical_optimum_se,
        fmt="D",
        markersize=9,
        color=SERIES["empirical_mse"][1],
        ecolor=SERIES["empirical_mse"][1],
        capsize=6,
        elinewidth=2.5,
        label="Empirical crossover ±1 SE",
        zorder=10,
    )

    ax.set_xlabel(r"Crossover point $c_p$", fontsize=60)
    ax.set_ylabel("MSE", fontsize=60)
    ax.tick_params(axis="both", which="major", labelsize=48)
    ax.grid(alpha=0.3)
    ax.set_title(dataset.title(), fontsize=45, pad=18)
    fig.tight_layout()
    output_path = output_dir / f"point_query_crossover_mse_{dataset}.pdf"
    fig.savefig(output_path, dpi=200)
    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)
    return output_path, handles, labels


def save_shared_legend(handles: list, labels: list, path: Path) -> None:
    legend_fig = plt.figure(figsize=(36, 2.5))
    legend_fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        fontsize=50,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )
    legend_fig.tight_layout()
    legend_fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(legend_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--runs", type=int, default=config.N_RUNS)
    parser.add_argument("--n-records", type=int)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "figures")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("runs must be positive")
    unknown = [name for name in args.datasets if name not in config.DATASET_N_RECORDS]
    if unknown:
        parser.error(f"unsupported dataset(s): {', '.join(unknown)}")
    missing_settings = [
        name for name in args.datasets
        if name not in DATASET_MICROBATCHES or name not in DATASET_CROSSOVER_STEP
    ]
    if missing_settings:
        parser.error(
            "add microbatch and crossover-step settings in the code for: "
            + ", ".join(missing_settings)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[pd.DataFrame] = []
    legend_handles: list = []
    legend_labels: list[str] = []
    for dataset in args.datasets:
        microbatch_count = int(DATASET_MICROBATCHES[dataset])
        crossover_step = int(DATASET_CROSSOVER_STEP[dataset])
        if microbatch_count <= 0 or crossover_step <= 0:
            parser.error(
                f"{dataset}: microbatch count and crossover step must be positive"
            )
        print(f"Running point-query MSE crossover experiment for {dataset}...")
        experiment_options = {
            "runs": args.runs,
            "total_microbatches": microbatch_count,
            "expected_microbatches": microbatch_count,
            "n_records": args.n_records or int(config.DATASET_N_RECORDS[dataset]),
            "seed": args.seed,
            "query_type": "point",
            "report_metric": "mse",
            "fixed_workload": True,
            "progress_callback": report_progress,
            "crossover_min": 0,
            "crossover_max": microbatch_count,
            "crossover_step": crossover_step,
        }
        if SHOW_CROSSOVER_DETAILS:
            result = crossover_experiment.run_experiment(dataset, **experiment_options)
        else:
            # The underlying experiment prints each c_p row. Capture that
            # verbose output while preserving errors and this script's
            # per-dataset summary messages.
            with contextlib.redirect_stdout(io.StringIO()):
                result = crossover_experiment.run_experiment(
                    dataset, **experiment_options
                )
        dataset_csv = args.output_dir / f"point_query_crossover_mse_{dataset}.csv"
        result.to_csv(dataset_csv, index=False)
        print(f"Saved {dataset} results CSV: {dataset_csv}")
        all_results.append(result)
        plot_path, handles, labels = plot_dataset(result, dataset, args.output_dir)
        if not legend_handles:
            legend_handles, legend_labels = handles, labels
        mean_crossover, crossover_se = _mean_and_standard_error(
            result.drop_duplicates("run")["empirical_mse_crossover"]
        )
        print(
            f"{dataset}: mean empirical MSE crossover={mean_crossover:.3f} "
            f"± {crossover_se:.3f} SE; saved {plot_path}"
        )

    combined = pd.concat(all_results, ignore_index=True)
    csv_path = args.output_dir / "point_query_crossover_mse_all_datasets.csv"
    combined.to_csv(csv_path, index=False)
    legend_path = args.output_dir / "point_query_crossover_mse_legend.pdf"
    save_shared_legend(legend_handles, legend_labels, legend_path)
    print(f"Saved results CSV: {csv_path}")
    print(f"Saved shared legend: {legend_path}")


if __name__ == "__main__":
    main()

"""
dataset.py

Dataset utilities for the AdaHDP prototype.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import config


###############################################################################
# Dataset loading
###############################################################################


def get_dataset_spec(dataset_name: str | None = None) -> dict:
    """Return a supported dataset specification by case-insensitive name."""
    key = (dataset_name or config.DATASET).lower()
    try:
        return config.DATASET_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; "
            f"choose from {tuple(config.DATASET_SPECS)}."
        ) from exc


def dataset_domain_size(dataset_name: str | None = None) -> int:
    """Return the flattened histogram domain size for a dataset."""
    return int(np.prod(get_dataset_spec(dataset_name)["attribute_domains"]))


def load_dataset(
    n_records: int,
    seed: int,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """
    Load and preprocess the dataset.
    """

    spec = get_dataset_spec(dataset_name)
    attributes = list(spec["attributes"])
    domains = list(spec["attribute_domains"])

    if spec["source"] == "openml":
        dataset = fetch_openml(
            spec["openml_name"],
            version=spec["version"],
            as_frame=True,
        )
        df = dataset.frame.loc[:, attributes].copy()
    elif spec["source"] == "local":
        path = Path(__file__).resolve().parent / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        separator = "," if path.suffix.lower() == ".csv" else r"\s+"
        df = pd.read_csv(path, sep=separator, usecols=attributes)
    else:
        raise ValueError(f"Unsupported dataset source: {spec['source']!r}.")

    if config.REMOVE_UNKNOWN:

        df = df.replace(
            config.UNKNOWN_VALUE,
            np.nan,
        )

    if config.DROP_NA:

        df = df.dropna()

    if spec["source"] == "openml":
        # Preserve the original Adult preprocessing: sample first, then fit
        # the numeric bins on that sampled experiment population.
        df = df.sample(
            n=min(n_records, len(df)),
            random_state=seed,
        ).reset_index(drop=True)

    for attribute, domain in zip(attributes, domains):

        values = pd.to_numeric(
            df[attribute],
            errors="coerce",
        )

        if spec["source"] == "local":
            if not values.notna().all():
                raise ValueError(
                    f"Local dataset column {attribute!r} must contain integers."
                )
            array = values.to_numpy(dtype=float)
            if not np.all(array == np.floor(array)):
                raise ValueError(
                    f"Local dataset column {attribute!r} has non-integer codes."
                )
            if np.any(array < 0):
                raise ValueError(
                    f"Column {attribute!r} contains codes outside [0, {domain})."
                )
            if np.any(array >= domain):
                # Local files may be pre-discretized at a finer resolution
                # than the selected Adult histogram. Re-bin deterministically
                # domain instead of rejecting otherwise valid source codes.
                source_domain = int(array.max()) + 1
                array = np.minimum(
                    domain - 1,
                    np.floor(array * domain / source_domain),
                )
            df[attribute] = array.astype(np.int64)

        elif values.notna().all():

            df[attribute] = pd.cut(
                values,
                bins=domain,
                labels=False,
                include_lowest=True,
            )

        else:

            df[attribute] = (
                df[attribute]
                .astype("category")
                .cat.codes
            )

    if spec["source"] == "local":
        df = df.sample(
            n=min(n_records, len(df)),
            random_state=seed,
        ).reset_index(drop=True)

    df = df.astype(
        np.int64,
    )

    return df


###############################################################################
# DP/LDP split
###############################################################################


def split_dataset(
    dataframe: pd.DataFrame,
    dp_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not 0.0 <= float(dp_ratio) <= 1.0:
        raise ValueError("dp_ratio must lie in [0, 1].")

    # Draw the DP users uniformly without replacement.  The LDP population is
    # formed from the exact complement, so no record can appear in both phases
    # and every record has the same probability of entering the DP group.
    n_records = len(dataframe)
    n_dp = int(n_records * float(dp_ratio))
    rng = np.random.default_rng(seed)
    dp_positions = rng.choice(n_records, size=n_dp, replace=False)
    dp_mask = np.zeros(n_records, dtype=bool)
    dp_mask[dp_positions] = True

    dp = dataframe.iloc[dp_mask].reset_index(drop=True)
    ldp = dataframe.iloc[~dp_mask].reset_index(drop=True)

    return dp, ldp


###############################################################################
# Histogram
###############################################################################


def encode_records(
    dataframe: pd.DataFrame,
    *,
    attributes: list[str] | tuple[str, ...] | None = None,
    attribute_domains: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Encode each record into one histogram cell.
    """

    attributes = list(attributes or config.ATTRIBUTES)
    attribute_domains = list(attribute_domains or config.ATTRIBUTE_DOMAINS)
    if len(attributes) != len(attribute_domains):
        raise ValueError("attributes and attribute_domains must have equal length.")

    values = dataframe[attributes].to_numpy()

    encoded = np.zeros(
        len(values),
        dtype=np.int64,
    )

    multiplier = 1

    for column in range(
        len(attributes) - 1,
        -1,
        -1,
    ):

        encoded += (
            values[:, column]
            * multiplier
        )

        multiplier *= attribute_domains[
            column
        ]

    return encoded


def build_histogram(
    dataframe: pd.DataFrame,
    dataset_name: str | None = None,
) -> np.ndarray:
    """
    Build a flattened histogram.
    """

    spec = get_dataset_spec(dataset_name)
    encoded = encode_records(
        dataframe,
        attributes=spec["attributes"],
        attribute_domains=spec["attribute_domains"],
    )

    histogram = np.bincount(
        encoded,
        minlength=dataset_domain_size(dataset_name),
    )

    return histogram.astype(
        float,
    )


###############################################################################
# Complete pipeline
###############################################################################


def load_data(
    n_records: int,
    dp_ratio: float,
    seed: int,
    dataset_name: str | None = None,
):

    dataframe = load_dataset(
        n_records,
        seed,
        dataset_name,
    )

    dp, ldp = split_dataset(
        dataframe,
        dp_ratio,
        seed,
    )

    dp_histogram = build_histogram(
        dp,
        dataset_name,
    )

    ldp_histogram = build_histogram(
        ldp,
        dataset_name,
    )

    return (
        dp,
        ldp,
        dp_histogram,
        ldp_histogram,
    )


###############################################################################
# Example
###############################################################################

if __name__ == "__main__":

    dp, ldp, h_dp, h_ldp = load_data(
        n_records=config.N_RECORDS,
        dp_ratio=config.DP_RATIO,
        seed=config.SEED,
    )

    print("=" * 80)
    print("Dataset")
    print("=" * 80)
    print(f"Attributes      : {config.ATTRIBUTES}")
    print(f"Attribute sizes : {config.ATTRIBUTE_DOMAINS}")
    print(f"Domain size     : {config.DOMAIN_SIZE}")
    print(f"DP records      : {len(dp)}")
    print(f"LDP records     : {len(ldp)}")
    print(f"Histogram size  : {len(h_dp)}")
    print(f"DP population   : {int(h_dp.sum())}")
    print(f"LDP population  : {int(h_ldp.sum())}")

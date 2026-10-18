"""
ldp.py

Optimized Unary Encoding (OUE) engine for the AdaHDP prototype.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


###############################################################################
# OUE parameters
###############################################################################


def oue_parameters(
    epsilon: float,
) -> tuple[float, float]:
    """
    Return the OUE parameters.
    """

    p = 0.5

    q = 1.0 / (
        np.exp(epsilon) + 1.0
    )

    return p, q


###############################################################################
# OUE Engine
###############################################################################


@dataclass
class OUEEngine:

    histogram: np.ndarray

    epsilon: float

    estimates: np.ndarray | None = None

    p: float | None = None

    q: float | None = None

    def __post_init__(
        self,
    ) -> None:

        self.p, self.q = oue_parameters(
            self.epsilon,
        )

        self.estimates = self._estimate()

    ###########################################################################
    # Internal
    ###########################################################################

    def _perturb(
        self,
    ) -> np.ndarray:

        unary = np.repeat(
            np.arange(
                self.histogram.size,
            ),
            self.histogram.astype(int),
        )

        reports = np.zeros(
            (
                len(unary),
                self.histogram.size,
            ),
            dtype=np.uint8,
        )

        reports[
            np.arange(
                len(unary),
            ),
            unary,
        ] = 1

        random = np.random.random(
            reports.shape,
        )

        return np.where(
            reports == 1,
            random < self.p,
            random < self.q,
        ).astype(
            np.uint8,
        )

    def _aggregate(
        self,
        reports: np.ndarray,
    ) -> np.ndarray:

        n = reports.shape[0]

        counts = reports.sum(
            axis=0,
        )

        estimates = (
            counts
            - n * self.q
        ) / (
            self.p
            - self.q
        )

        if config.OUE_CLIP_NEGATIVE_ESTIMATES:
            estimates = np.maximum(estimates, 0.0)
        return estimates

    def _estimate(
        self,
    ) -> np.ndarray:

        reports = self._perturb()

        return self._aggregate(
            reports,
        )

    ###########################################################################
    # Public
    ###########################################################################

    def answer(
        self,
        workload: np.ndarray,
    ) -> np.ndarray:
        """
        Answer a workload.
        """

        if workload.shape[1] != self.estimates.shape[0]:
            raise ValueError(
                "LDP domain mismatch: "
                f"workload has {workload.shape[1]} columns, "
                f"but estimates has {self.estimates.shape[0]} entries."
            )

        return workload @ self.estimates

    def variance(
        self,
    ) -> float:
        """
        Worst-case OUE variance.
        """

        n = self.histogram.sum()

        if n == 0:

            return 0.0

        e = np.exp(
            self.epsilon,
        )

        return (
            4.0
            * e
            / (
                n
                * (e - 1.0) ** 2
            )
        )


###############################################################################
# Factory
###############################################################################


def build_engine(
    histogram: np.ndarray,
    epsilon: float,
) -> OUEEngine:
    """
    Build an OUE engine.
    """

    return OUEEngine(
        histogram,
        epsilon,
    )


###############################################################################
# Example
###############################################################################

if __name__ == "__main__":

    rng = np.random.default_rng(
        42,
    )

    histogram = rng.poisson(
        10,
        128,
    ).astype(
        float,
    )

    workload = np.eye(
        128,
        dtype=float,
    )

    engine = build_engine(
        histogram,
        epsilon=1.0,
    )

    answers = engine.answer(
        workload,
    )

    print("=" * 80)
    print("OUE")
    print("=" * 80)
    print(f"Histogram size : {histogram.size}")
    print(f"Variance       : {engine.variance():.6f}")
    print(f"Answers        : {len(answers)}")

"""
budget.py

Privacy budget allocation for the AdaHDP prototype.

"""

from __future__ import annotations
import math
import config


###############################################################################
# Budget allocation functions
###############################################################################


def uniform(wci: float) -> float:
    """Return the uniform allocation score."""
    del wci
    return 1.0


def identity(wci: float) -> float:
    """Return the linear allocation score."""
    return float(wci)


def sqrt(wci: float) -> float:
    """Return the square-root allocation score."""
    return float(math.sqrt(wci))


def quadratic(wci: float) -> float:
    """Return the quadratic allocation score."""
    return float(wci**2)


def logarithmic(wci: float, alpha: float = 9.0) -> float:
    """Return the normalized logarithmic allocation score."""
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")
    return float(math.log1p(alpha * wci) / math.log1p(alpha))


###############################################################################
# Dispatcher
###############################################################################


_FUNCTIONS = {
    "uniform": uniform,
    "identity": identity,
    "linear": identity,
    "sqrt": sqrt,
    "quadratic": quadratic,
    "log": logarithmic,
    "logarithmic": logarithmic,
}


def evaluate(wci: float, policy: str) -> float:
    """Evaluate the selected allocation score."""
    if policy not in _FUNCTIONS:
        raise ValueError(f"Unknown budget policy '{policy}'.")
    if not math.isfinite(wci):
        raise ValueError("wci must be finite.")
    bounded_wci = float(min(max(wci, 0.0), 1.0))
    value = float(_FUNCTIONS[policy](bounded_wci))
    if not math.isfinite(value):
        raise ValueError(f"Policy '{policy}' returned a non-finite score.")
    return float(min(max(value, 0.0), 1.0))


###############################################################################
# Heuristic allocation
###############################################################################


def allocate_budget(
    epsilon_dp: float,
    cp: int,
    microbatch_size: int,
    wci: float,
    beta: float,
    policy: str,
    remaining_budget: float,
) -> float:
    """
    Allocate DP budget to the current DP portion of a microbatch.

    The allocation is proportional to the number of queries in the current
    DP portion. Therefore, this function directly supports variable-size
    microbatches and microbatches split at the DP/LDP crossover point.
    """
    if not math.isfinite(epsilon_dp) or epsilon_dp < 0.0:
        raise ValueError("epsilon_dp must be finite and non-negative.")
    if cp <= 0:
        raise ValueError("cp must be positive.")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive.")
    if microbatch_size > cp:
        raise ValueError("microbatch_size cannot exceed cp.")
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and belong to [0, 1].")
    if not math.isfinite(remaining_budget) or remaining_budget < 0.0:
        raise ValueError("remaining_budget must be finite and non-negative.")
    if remaining_budget <= config.EPSILON_TOL:
        return 0.0

    epsilon_base = microbatch_size * epsilon_dp / cp
    scale = beta + (1.0 - beta) * evaluate(wci=wci, policy=policy)
    epsilon = epsilon_base * scale
    return float(min(max(epsilon, 0.0), remaining_budget))


###############################################################################
# Expected optimal online allocation
###############################################################################


def online_optimal(
    remaining_budget: float,
    current_weight: float,
    remaining_microbatches: int,
    expected_weight: float,
) -> float:
    """
    Allocate budget using the expected optimal online rule.

    The scheduler supplies the weight of the current microbatch and an
    estimate of the number of remaining DP microbatches. This function is
    independent of microbatch size because any size dependence is already
    incorporated into the scheduler's selected weight formulation.
    """
    if not math.isfinite(remaining_budget) or remaining_budget < 0.0:
        raise ValueError("remaining_budget must be finite and non-negative.")
    if not math.isfinite(current_weight) or current_weight < 0.0:
        raise ValueError("current_weight must be finite and non-negative.")
    if remaining_microbatches <= 0:
        return 0.0
    if not math.isfinite(expected_weight) or expected_weight < 0.0:
        raise ValueError("expected_weight must be finite and non-negative.")
    if remaining_budget <= config.EPSILON_TOL:
        return 0.0
    if remaining_microbatches == 1:
        return float(remaining_budget)

    denominator = (current_weight + (remaining_microbatches - 1) * expected_weight)
    if denominator <= config.EPSILON_TOL:
        return 0.0

    epsilon = remaining_budget * current_weight / denominator
    return float(min(max(epsilon, 0.0), remaining_budget))


def smart_min_epsilon(
    sensitivity: float,
    variance_term: float,
    target_variance: float,
    remaining_budget: float,
) -> float:
    """Return the smallest epsilon achieving a target HDMM variance.

    HDMM's Gaussian reconstruction variance is
    ``2 * sensitivity**2 * variance_term / epsilon**2``.  Solving this
    expression for epsilon gives the minimum useful budget for one workload;
    the result is capped by the scheduler's remaining central-DP budget.
    """
    values = (sensitivity, variance_term, target_variance, remaining_budget)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Smart-allocation parameters must be finite.")
    if sensitivity <= 0.0 or variance_term <= 0.0 or target_variance <= 0.0:
        raise ValueError("Sensitivity, variance, and target must be positive.")
    if remaining_budget < 0.0:
        raise ValueError("remaining_budget must be non-negative.")
    epsilon_star = smart_required_epsilon(
        sensitivity=sensitivity,
        variance_term=variance_term,
        target_variance=target_variance,
    )
    return float(min(epsilon_star, remaining_budget))


def smart_required_epsilon(
    sensitivity: float,
    variance_term: float,
    target_variance: float,
) -> float:
    """Return the unconstrained epsilon required by the smart utility target."""
    values = (sensitivity, variance_term, target_variance)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Smart-allocation parameters must be finite.")
    if sensitivity <= 0.0 or variance_term <= 0.0 or target_variance <= 0.0:
        raise ValueError("Sensitivity, variance, and target must be positive.")
    return float(math.sqrt(2.0 * sensitivity**2 * variance_term / target_variance))


###############################################################################
# Statistics
###############################################################################


def worst_case_budget(
    epsilon_dp: float,
    cp: int,
    microbatch_size: int,
) -> float:
    """
    Return the base DP budget for a DP portion of a microbatch.

    The returned value scales with the actual number of queries in the DP
    portion and is therefore valid for variable-size microbatches.
    """
    if not math.isfinite(epsilon_dp) or epsilon_dp < 0.0:
        raise ValueError("epsilon_dp must be finite and non-negative.")
    if cp <= 0:
        raise ValueError("cp must be positive.")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive.")
    if microbatch_size > cp:
        raise ValueError("microbatch_size cannot exceed cp.")
    return float(microbatch_size * epsilon_dp / cp)


def savings(
    epsilon_dp: float,
    cp: int,
    microbatch_size: int,
    allocated_budget: float,
) -> float:
    """Return the budget saved relative to the base allocation."""
    if not math.isfinite(allocated_budget) or allocated_budget < 0.0:
        raise ValueError(
            "allocated_budget must be finite and non-negative."
        )
    return float(
        worst_case_budget(
            epsilon_dp=epsilon_dp,
            cp=cp,
            microbatch_size=microbatch_size,
        )
        - allocated_budget
    )


def utilization(epsilon_dp: float, remaining_budget: float) -> float:

    """Return the fraction of the DP privacy budget consumed."""

    if not math.isfinite(epsilon_dp) or epsilon_dp < 0.0:
        raise ValueError("epsilon_dp must be finite and non-negative.")
    if not math.isfinite(remaining_budget) or remaining_budget < 0.0:
        raise ValueError(
            "remaining_budget must be finite and non-negative."
        )
    if epsilon_dp <= config.EPSILON_TOL:
        return 0.0
    consumed_fraction = (epsilon_dp - remaining_budget) / epsilon_dp
    return float(min(max(consumed_fraction, 0.0), 1.0))


###############################################################################
# Example
###############################################################################


if __name__ == "__main__":
    heuristic_epsilon = allocate_budget(
        epsilon_dp=1.0,
        cp=100,
        microbatch_size=17,
        wci=0.60,
        beta=0.20,
        policy="sqrt",
        remaining_budget=1.0,
    )

    online_epsilon = online_optimal(
        remaining_budget=1.0,
        current_weight=1.25,
        remaining_microbatches=5,
        expected_weight=1.0,
    )

    print("=" * 80)
    print("Budget Allocation")
    print("=" * 80)
    print(f"Heuristic allocation : {heuristic_epsilon:.6f}")
    print(f"Online allocation    : {online_epsilon:.6f}")

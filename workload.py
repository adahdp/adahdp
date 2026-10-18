"""Point-query generation and microbatching."""
from __future__ import annotations
import numpy as np


def point_query(domain_size: int, rng: np.random.Generator) -> np.ndarray:
    query = np.zeros(domain_size, dtype=float)
    query[rng.integers(domain_size)] = 1.0
    return query


def microbatches(workload: np.ndarray, batch_size: int,
                 rng: np.random.Generator | None = None,
                 min_batch_size: int | None = None,
                 max_batch_size: int | None = None) -> list[np.ndarray]:
    if rng is None or min_batch_size is None or max_batch_size is None:
        return [workload[i:i + batch_size] for i in range(0, len(workload), batch_size)]
    if min_batch_size <= 0 or max_batch_size < min_batch_size:
        raise ValueError("Invalid microbatch size bounds.")
    batches = []
    start = 0
    while start < len(workload):
        size = int(rng.integers(min_batch_size, max_batch_size + 1))
        end = min(start + size, len(workload))
        batches.append(workload[start:end])
        start = end
    return batches

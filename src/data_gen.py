"""
data_gen.py — synthetic clustered (Gaussian-mixture) vector dataset and
query generator. Deterministic given a seed.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def generate_gaussian_mixture(
    n: int, d: int, n_components: int, seed: int, cluster_std: float = 0.35
) -> Tuple[np.ndarray, np.ndarray]:
    """n vectors drawn from n_components Gaussian blobs scattered in R^d."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=3.0, size=(n_components, d)).astype(np.float32)
    assignments = rng.integers(0, n_components, size=n)
    noise = rng.normal(loc=0.0, scale=cluster_std, size=(n, d)).astype(np.float32)
    vectors = centers[assignments] + noise
    return vectors.astype(np.float32), assignments.astype(np.int64)


def generate_queries(
    n_queries: int, d: int, n_components: int, seed: int, cluster_std: float = 0.35
) -> np.ndarray:
    """Independent draws from the same mixture — realistic, unseen queries."""
    queries, _ = generate_gaussian_mixture(
        n_queries, d, n_components, seed=seed + 999, cluster_std=cluster_std
    )
    return queries


def build_dataset(
    n: int = 50_000,
    d: int = 128,
    n_components: int = 100,
    n_queries: int = 500,
    seed: int = 42,
):
    vectors, assignments = generate_gaussian_mixture(n, d, n_components, seed=seed)
    ids = np.arange(n, dtype=np.int64)
    queries = generate_queries(n_queries, d, n_components, seed=seed)
    return vectors, ids, assignments, queries

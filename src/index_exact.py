"""
index_exact.py — brute-force exact nearest-neighbor index (ground truth).

Cosine similarity is reduced to dot-product similarity by L2-normalizing
vectors on ingestion. All search-path math is vectorized NumPy; no Python
loop ever touches the vector-math/search path.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple

_EPS = 1e-12


class ExactIndex:
    """Brute-force exact index: S = Q @ X.T, top-k via argpartition."""

    def __init__(self, dim: int, initial_capacity: int = 1024, dtype=np.float32):
        self.dim = dim
        self.dtype = dtype
        self._capacity = max(initial_capacity, 1)
        self.vectors = np.zeros((self._capacity, dim), dtype=dtype)
        self.ids = np.full(self._capacity, -1, dtype=np.int64)
        self.active = np.zeros(self._capacity, dtype=bool)
        self.size = 0  # highest used slot (includes tombstoned slots)
        self._id_to_slot: dict[int, int] = {}

    # ---------------- internal ----------------

    def _grow(self, min_extra: int) -> None:
        needed = self.size + min_extra
        if needed <= self._capacity:
            return
        new_capacity = max(self._capacity * 2, needed)
        new_vectors = np.zeros((new_capacity, self.dim), dtype=self.dtype)
        new_ids = np.full(new_capacity, -1, dtype=np.int64)
        new_active = np.zeros(new_capacity, dtype=bool)
        new_vectors[: self.size] = self.vectors[: self.size]
        new_ids[: self.size] = self.ids[: self.size]
        new_active[: self.size] = self.active[: self.size]
        self.vectors, self.ids, self.active = new_vectors, new_ids, new_active
        self._capacity = new_capacity

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, _EPS, None)

    # ---------------- public API ----------------

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=self.dtype)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("vectors/ids length mismatch")

        n = vectors.shape[0]
        self._grow(n)
        normalized = self._l2_normalize(vectors)

        start = self.size
        end = start + n
        self.vectors[start:end] = normalized
        self.ids[start:end] = ids
        self.active[start:end] = True
        self.size = end

        # Bookkeeping only — NOT on the vector-math/search path.
        for slot, _id in zip(range(start, end), ids.tolist()):
            self._id_to_slot[_id] = slot

    def delete(self, ids: List[int]) -> None:
        # Bookkeeping loop over a typically-small id list; O(1) tombstone per id.
        for _id in ids:
            slot = self._id_to_slot.pop(_id, None)
            if slot is not None:
                self.active[slot] = False

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(query, dtype=self.dtype)
        single = query.ndim == 1
        if single:
            query = query[None, :]
        if query.shape[1] != self.dim:
            raise ValueError(f"expected queries of dim {self.dim}, got {query.shape[1]}")

        n_active = int(self.active[: self.size].sum())
        if self.size == 0 or n_active == 0:
            out_ids = np.full((query.shape[0], k), -1, dtype=np.int64)
            out_scores = np.full((query.shape[0], k), -np.inf, dtype=self.dtype)
            return (out_ids[0], out_scores[0]) if single else (out_ids, out_scores)

        q = self._l2_normalize(query)
        X = self.vectors[: self.size]
        S = q @ X.T  # (n_queries, size) — the only O(n*d) step

        inactive = ~self.active[: self.size]
        if inactive.any():
            S[:, inactive] = -np.inf

        k_eff = min(k, n_active)  # never select more than the active count
        # top-k_eff candidates via argpartition (never a full argsort over candidates)
        part_idx = np.argpartition(-S, kth=k_eff - 1, axis=1)[:, :k_eff]
        part_scores = np.take_along_axis(S, part_idx, axis=1)

        # locally sort only the k_eff selected candidates
        order = np.argsort(-part_scores, axis=1)
        top_idx = np.take_along_axis(part_idx, order, axis=1)
        top_scores = np.take_along_axis(part_scores, order, axis=1)
        top_ids = self.ids[: self.size][top_idx]

        if k_eff < k:
            pad_ids = np.full((query.shape[0], k - k_eff), -1, dtype=np.int64)
            pad_scores = np.full((query.shape[0], k - k_eff), -np.inf, dtype=self.dtype)
            top_ids = np.concatenate([top_ids, pad_ids], axis=1)
            top_scores = np.concatenate([top_scores, pad_scores], axis=1)

        if single:
            return top_ids[0], top_scores[0]
        return top_ids, top_scores

    def __len__(self) -> int:
        return int(self.active[: self.size].sum())

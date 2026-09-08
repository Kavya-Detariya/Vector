"""
index_ivf.py — Production-grade IVF-Flat + IVF-SQ8 approximate vector index.

Enhancements:
- Spherical K-Means++ initialization (geodesic dispersion).
- Pure-NumPy mini-batch centroid updates via one-hot GEMM.
- Optional 8-bit Scalar Quantization (IVF-SQ8) for 75% posting memory reduction.
- Dynamic adaptive nprobe via cumulative probability mass routing.
- Tombstone masking in the candidate scoring path without array copying.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

_EPS = 1e-12
_SQ8_SCALE = 127.0


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, _EPS, None)


def _spherical_kmeans_plus_plus(
    vectors: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Initializes k centroids on the unit sphere using geodesic distance dispersion.
    Probability of picking seed i is proportional to (1.0 - max_cosine_sim)^2.
    """
    n, dim = vectors.shape
    centroids = np.empty((k, dim), dtype=np.float32)

    # 1. Pick first centroid uniformly at random
    first_idx = rng.integers(0, n)
    centroids[0] = vectors[first_idx]

    # Keep track of the maximum cosine similarity to any chosen centroid so far
    max_sims = vectors @ centroids[0]

    for c in range(1, k):
        # Geodesic angular distance proxy: d = 1 - sim
        distances = np.clip(1.0 - max_sims, 0.0, 2.0)
        probs = distances ** 2
        total_prob = probs.sum()

        if total_prob < _EPS:
            # Fallback for degenerate duplicate vectors
            next_idx = rng.integers(0, n)
        else:
            probs /= total_prob
            next_idx = rng.choice(n, p=probs)

        centroids[c] = vectors[next_idx]

        if c < k - 1:
            new_sims = vectors @ centroids[c]
            max_sims = np.maximum(max_sims, new_sims)

    return centroids


def _minibatch_kmeans_spherical(
    vectors: np.ndarray,
    k: int,
    n_iters: int = 60,
    batch_size: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Pure-NumPy mini-batch spherical k-means with one-hot GEMM updates."""
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]

    centroids = _spherical_kmeans_plus_plus(vectors, k, rng)
    counts = np.zeros(k, dtype=np.float64)

    for _ in range(n_iters):
        bsz = min(batch_size, n)
        batch_idx = rng.choice(n, size=bsz, replace=False)
        batch = vectors[batch_idx]

        sims = batch @ centroids.T  # (bsz, k)
        assign = np.argmax(sims, axis=1)

        # Vectorized batch accumulation via one-hot matrix multiplication
        one_hot = np.zeros((bsz, k), dtype=np.float64)
        one_hot[np.arange(bsz), assign] = 1.0
        sums = one_hot.T @ batch  # (k, dim)
        batch_counts = one_hot.sum(axis=0)

        hit = batch_counts > 0
        counts[hit] += batch_counts[hit]
        lr = (batch_counts[hit] / counts[hit])[:, None]
        means = sums[hit] / batch_counts[hit][:, None]
        centroids[hit] = (1.0 - lr) * centroids[hit] + lr * means
        centroids = _l2_normalize(centroids)

    return centroids.astype(np.float32)


class IVFFlatIndex:
    """
    IVF Index with optional SQ8 quantization, adaptive probe routing,
    and vectorized posting gather.
    """

    def __init__(
        self,
        dim: int,
        quantize: bool = False,
        dtype=np.float32,
    ):
        self.dim = dim
        self.quantize = quantize
        self.dtype = dtype
        self.centroids: Optional[np.ndarray] = None
        self.k: int = 0
        self.postings: Dict[int, Dict[str, np.ndarray]] = {}
        self._id_location: Dict[int, Tuple[int, int]] = {}
        self._trained = False

    def train(
        self,
        vectors: np.ndarray,
        n_iters: int = 60,
        k: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=self.dtype)
        n = vectors.shape[0]
        self.k = max(1, int(np.floor(np.sqrt(n)))) if k is None else int(k)
        normalized = _l2_normalize(vectors)

        self.centroids = _minibatch_kmeans_spherical(
            normalized, self.k, n_iters=n_iters, seed=seed
        )

        vec_dtype = np.int8 if self.quantize else self.dtype
        self.postings = {
            c: {
                "vectors": np.zeros((0, self.dim), dtype=vec_dtype),
                "ids": np.zeros((0,), dtype=np.int64),
                "active": np.zeros((0,), dtype=bool),
            }
            for c in range(self.k)
        }
        self._id_location.clear()
        self._trained = True

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        if not self._trained or self.centroids is None:
            raise RuntimeError("Index must be trained before calling add()")

        vectors = np.ascontiguousarray(vectors, dtype=self.dtype)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("vectors and ids length mismatch")

        normalized = _l2_normalize(vectors)
        sims = normalized @ self.centroids.T
        assign = np.argmax(sims, axis=1)

        # Quantize if SQ8 enabled: [-1.0, 1.0] float -> [-127, 127] int8
        stored_vectors = (
            np.clip(np.round(normalized * _SQ8_SCALE), -127, 127).astype(np.int8)
            if self.quantize
            else normalized
        )

        order = np.argsort(assign, kind="stable")
        sorted_assign = assign[order]
        sorted_vecs = stored_vectors[order]
        sorted_ids = ids[order]

        unique_clusters, start_idx = np.unique(sorted_assign, return_index=True)
        boundaries = list(start_idx) + [len(sorted_assign)]

        # Group contiguous cluster slices into posting tables
        for pos, c in enumerate(unique_clusters):
            s, e = boundaries[pos], boundaries[pos + 1]
            new_v = sorted_vecs[s:e]
            new_i = sorted_ids[s:e]

            post = self.postings[int(c)]
            base = post["vectors"].shape[0]
            post["vectors"] = np.concatenate([post["vectors"], new_v], axis=0)
            post["ids"] = np.concatenate([post["ids"], new_i], axis=0)
            post["active"] = np.concatenate(
                [post["active"], np.ones(e - s, dtype=bool)], axis=0
            )

            for offset, _id in enumerate(new_i.tolist()):
                self._id_location[int(_id)] = (int(c), base + offset)

    def delete(self, ids: List[int]) -> int:
        deleted = 0
        for _id in ids:
            loc = self._id_location.pop(int(_id), None)
            if loc is not None:
                c, idx = loc
                self.postings[c]["active"][idx] = False
                deleted += 1
        return deleted

    def _resolve_clusters(
        self,
        q: np.ndarray,
        nprobe: int,
        adaptive_threshold: Optional[float] = None,
        temperature: float = 0.05,
    ) -> np.ndarray:
        """Resolves target clusters using either fixed nprobe or cumulative softmax probability."""
        if self.centroids is None:
            return np.empty(0, dtype=np.int64)

        csims = q @ self.centroids.T

        if adaptive_threshold is None:
            nprobe_eff = min(max(1, nprobe), self.k)
            return np.argpartition(-csims, kth=nprobe_eff - 1)[:nprobe_eff]

        # Dynamic routing: evaluate softmax cumulative density
        shift_sims = (csims - np.max(csims)) / max(temperature, 1e-4)
        exp_sims = np.exp(shift_sims)
        probs = exp_sims / np.sum(exp_sims)

        sorted_clusters = np.argsort(-probs)
        cum_probs = np.cumsum(probs[sorted_clusters])
        cutoff = np.searchsorted(cum_probs, adaptive_threshold)
        probe_count = int(np.clip(cutoff + 1, 1, min(max(1, nprobe), self.k)))
        return sorted_clusters[:probe_count]

    def _search_single(
        self,
        q: np.ndarray,
        k: int,
        nprobe: int,
        adaptive_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_clusters = self._resolve_clusters(
            q, nprobe=nprobe, adaptive_threshold=adaptive_threshold
        )

        vec_parts, id_parts, act_parts = [], [], []
        for c in target_clusters.tolist():
            post = self.postings[c]
            if post["vectors"].shape[0] == 0:
                continue
            vec_parts.append(post["vectors"])
            id_parts.append(post["ids"])
            act_parts.append(post["active"])

        if not vec_parts:
            return (
                np.full(k, -1, dtype=np.int64),
                np.full(k, -np.inf, dtype=self.dtype),
            )

        cand_vectors = np.concatenate(vec_parts, axis=0)
        cand_ids = np.concatenate(id_parts, axis=0)
        cand_active = np.concatenate(act_parts, axis=0)

        # Single dot-product over gathered candidate block
        if self.quantize:
            # Float query dot int8 matrix, dequantized with scalar multiply
            scores = (cand_vectors.astype(np.float32) @ q) / _SQ8_SCALE
        else:
            scores = cand_vectors @ q

        scores = np.where(cand_active, scores, -np.inf)
        n_active = int(cand_active.sum())
        k_eff = min(k, n_active)

        if k_eff == 0:
            return (
                np.full(k, -1, dtype=np.int64),
                np.full(k, -np.inf, dtype=self.dtype),
            )

        part_idx = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
        part_scores = scores[part_idx]
        local_order = np.argsort(-part_scores)

        top_idx = part_idx[local_order]
        top_scores = part_scores[local_order]
        top_ids = cand_ids[top_idx]

        if k_eff < k:
            pad_len = k - k_eff
            top_ids = np.concatenate([top_ids, np.full(pad_len, -1, dtype=np.int64)])
            top_scores = np.concatenate(
                [top_scores, np.full(pad_len, -np.inf, dtype=self.dtype)]
            )

        return top_ids, top_scores

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: int = 4,
        adaptive_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained:
            raise RuntimeError("Index must be trained before search()")

        query = np.ascontiguousarray(query, dtype=self.dtype)
        single = query.ndim == 1
        if single:
            query = query[None, :]

        q_norm = _l2_normalize(query)
        nq = q_norm.shape[0]
        out_ids = np.empty((nq, k), dtype=np.int64)
        out_scores = np.empty((nq, k), dtype=self.dtype)

        for i in range(nq):
            top_ids, top_scores = self._search_single(
                q_norm[i],
                k=k,
                nprobe=nprobe,
                adaptive_threshold=adaptive_threshold,
            )
            out_ids[i] = top_ids
            out_scores[i] = top_scores

        if single:
            return out_ids[0], out_scores[0]
        return out_ids, out_scores

    def get_memory_bytes(self) -> int:
        """Returns byte volume consumed by stored posting lists."""
        total = 0
        for p in self.postings.values():
            total += p["vectors"].nbytes + p["ids"].nbytes + p["active"].nbytes
        return total

    def __len__(self) -> int:
        return sum(int(p["active"].sum()) for p in self.postings.values())
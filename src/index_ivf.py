"""
index_ivf.py — IVF-Flat approximate index.

Pipeline: pure-NumPy mini-batch k-means partitions the space into
k = floor(sqrt(N)) clusters. Search probes only the nprobe nearest
centroids, gathers their posting lists into a contiguous candidate
block, and scores that block with a single dot product.

Vectorization notes:
- k-means centroid updates use a one-hot @ batch matmul — no loop over
  clusters or vectors.
- add() sorts by assigned cluster (vectorized argsort) then loops only
  over the *unique clusters present in the batch* (<= k, i.e. <= sqrt(N))
  to slice already-contiguous ranges into per-cluster posting arrays.
  That loop is bookkeeping, not vector math.
- search() loops over queries in a batch because each query probes a
  different, ragged set of clusters — this is standard for IVF (every
  production IVF implementation scores queries independently for this
  reason). The per-query inner work is a single vectorized dot product.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict

_EPS = 1e-12


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, _EPS, None)


def _minibatch_kmeans(
    vectors: np.ndarray,
    k: int,
    n_iters: int = 60,
    batch_size: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Pure-NumPy mini-batch k-means on unit vectors (cosine == dot product)."""
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[init_idx].copy()
    counts = np.zeros(k, dtype=np.float64)

    for _ in range(n_iters):
        bsz = min(batch_size, n)
        batch_idx = rng.choice(n, size=bsz, replace=False)
        batch = vectors[batch_idx]

        sims = batch @ centroids.T  # (bsz, k)
        assign = np.argmax(sims, axis=1)

        # Vectorized centroid update via one-hot matmul — no loop over clusters.
        one_hot = np.zeros((bsz, k), dtype=np.float64)
        one_hot[np.arange(bsz), assign] = 1.0
        sums = one_hot.T @ batch  # (k, d)
        batch_counts = one_hot.sum(axis=0)  # (k,)

        hit = batch_counts > 0
        counts[hit] += batch_counts[hit]
        lr = (batch_counts[hit] / counts[hit])[:, None]
        means = sums[hit] / batch_counts[hit][:, None]
        centroids[hit] = (1 - lr) * centroids[hit] + lr * means

        centroids = _l2_normalize(centroids)

    return centroids.astype(vectors.dtype)


class IVFFlatIndex:
    def __init__(self, dim: int, dtype=np.float32):
        self.dim = dim
        self.dtype = dtype
        self.centroids: np.ndarray | None = None
        self.k: int = 0
        self.postings: Dict[int, Dict[str, np.ndarray]] = {}
        # id -> (cluster_id, row index within that cluster's arrays)
        self._id_location: Dict[int, Tuple[int, int]] = {}
        self._trained = False

    # ---------------- training ----------------

    def train(self, vectors: np.ndarray, n_iters: int = 60, seed: int = 0) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=self.dtype)
        n = vectors.shape[0]
        self.k = max(1, int(np.floor(np.sqrt(n))))
        normalized = _l2_normalize(vectors)
        self.centroids = _minibatch_kmeans(normalized, self.k, n_iters=n_iters, seed=seed)
        self.postings = {
            c: {
                "vectors": np.zeros((0, self.dim), dtype=self.dtype),
                "ids": np.zeros((0,), dtype=np.int64),
                "active": np.zeros((0,), dtype=bool),
            }
            for c in range(self.k)
        }
        self._id_location = {}
        self._trained = True

    # ---------------- mutation ----------------

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        if not self._trained:
            raise RuntimeError("call train() before add()")
        vectors = np.ascontiguousarray(vectors, dtype=self.dtype)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("vectors/ids length mismatch")

        normalized = _l2_normalize(vectors)
        sims = normalized @ self.centroids.T  # (n, k) — vectorized nearest-centroid
        assign = np.argmax(sims, axis=1)

        order = np.argsort(assign, kind="stable")
        sorted_assign = assign[order]
        sorted_vectors = normalized[order]
        sorted_ids = ids[order]

        unique_clusters, start_idx = np.unique(sorted_assign, return_index=True)
        boundaries = list(start_idx) + [len(sorted_assign)]

        # Loop only over clusters *present in this batch* (<= k <= sqrt(N)).
        for pos, c in enumerate(unique_clusters):
            s, e = boundaries[pos], boundaries[pos + 1]
            new_v = sorted_vectors[s:e]
            new_ids = sorted_ids[s:e]

            post = self.postings[int(c)]
            base = post["vectors"].shape[0]
            post["vectors"] = np.concatenate([post["vectors"], new_v], axis=0)
            post["ids"] = np.concatenate([post["ids"], new_ids], axis=0)
            post["active"] = np.concatenate(
                [post["active"], np.ones(e - s, dtype=bool)], axis=0
            )

            for offset, _id in enumerate(new_ids.tolist()):
                self._id_location[_id] = (int(c), base + offset)

    def delete(self, ids: List[int]) -> None:
        for _id in ids:
            loc = self._id_location.pop(_id, None)
            if loc is not None:
                c, idx = loc
                self.postings[c]["active"][idx] = False

    # ---------------- search ----------------

    def _search_single(self, q: np.ndarray, k: int, nprobe: int) -> Tuple[np.ndarray, np.ndarray]:
        csims = q @ self.centroids.T  # (k_clusters,)
        nprobe_eff = min(nprobe, self.k)
        top_clusters = np.argpartition(-csims, kth=nprobe_eff - 1)[:nprobe_eff]

        vec_parts, id_parts, active_parts = [], [], []
        for c in top_clusters.tolist():
            post = self.postings[c]
            if post["vectors"].shape[0] == 0:
                continue
            vec_parts.append(post["vectors"])
            id_parts.append(post["ids"])
            active_parts.append(post["active"])

        if not vec_parts:
            return (
                np.full(k, -1, dtype=np.int64),
                np.full(k, -np.inf, dtype=self.dtype),
            )

        cand_vectors = np.concatenate(vec_parts, axis=0)
        cand_ids = np.concatenate(id_parts, axis=0)
        cand_active = np.concatenate(active_parts, axis=0)

        scores = cand_vectors @ q  # single dot product over the gathered candidate block
        scores = np.where(cand_active, scores, -np.inf)

        n_active_cand = int(cand_active.sum())
        k_eff = min(k, n_active_cand)
        if k_eff == 0:
            return (
                np.full(k, -1, dtype=np.int64),
                np.full(k, -np.inf, dtype=self.dtype),
            )

        part_idx = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
        part_scores = scores[part_idx]
        order = np.argsort(-part_scores)
        top_idx = part_idx[order]
        top_scores = part_scores[order]
        top_ids = cand_ids[top_idx]

        if k_eff < k:
            top_ids = np.concatenate([top_ids, np.full(k - k_eff, -1, dtype=np.int64)])
            top_scores = np.concatenate(
                [top_scores, np.full(k - k_eff, -np.inf, dtype=self.dtype)]
            )
        return top_ids, top_scores

    def search(
        self, query: np.ndarray, k: int = 10, nprobe: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained:
            raise RuntimeError("call train() before search()")
        query = np.ascontiguousarray(query, dtype=self.dtype)
        single = query.ndim == 1
        if single:
            query = query[None, :]

        q_norm = _l2_normalize(query)
        out_ids = np.empty((q_norm.shape[0], k), dtype=np.int64)
        out_scores = np.empty((q_norm.shape[0], k), dtype=self.dtype)
        for i in range(q_norm.shape[0]):
            ids_i, scores_i = self._search_single(q_norm[i], k, nprobe)
            out_ids[i] = ids_i
            out_scores[i] = scores_i

        if single:
            return out_ids[0], out_scores[0]
        return out_ids, out_scores

    def __len__(self) -> int:
        return sum(int(p["active"].sum()) for p in self.postings.values())

"""
evaluate.py — build the exact index over the synthetic dataset, compute
ground-truth top-10 neighbors for the query set, report baseline
latency/QPS, and persist everything IVF-Flat will need later for
recall scoring.
"""
from __future__ import annotations

import time
import numpy as np

from index_exact import ExactIndex
from data_gen import build_dataset

N, D, N_COMPONENTS, N_QUERIES, K, SEED = 50_000, 128, 100, 500, 10, 42


def main():
    vectors, ids, _, queries = build_dataset(N, D, N_COMPONENTS, N_QUERIES, SEED)

    index = ExactIndex(dim=D)
    t0 = time.perf_counter()
    index.add(vectors, ids)
    build_s = time.perf_counter() - t0
    print(f"build: {N} vectors, d={D} in {build_s:.3f}s")

    index.search(queries[:1], k=K)  # warm-up (BLAS thread/cache)

    latencies = np.empty(N_QUERIES, dtype=np.float64)
    gt_ids = np.empty((N_QUERIES, K), dtype=np.int64)
    gt_scores = np.empty((N_QUERIES, K), dtype=np.float32)

    for i in range(N_QUERIES):
        t0 = time.perf_counter()
        ids_i, scores_i = index.search(queries[i], k=K)
        latencies[i] = time.perf_counter() - t0
        gt_ids[i] = ids_i
        gt_scores[i] = scores_i

    p50 = np.percentile(latencies, 50) * 1000
    p99 = np.percentile(latencies, 99) * 1000
    qps = 1.0 / np.mean(latencies)
    print(f"exact search  p50={p50:.3f}ms  p99={p99:.3f}ms  QPS={qps:.1f}")

    np.savez(
        "ground_truth.npz",
        queries=queries,
        gt_ids=gt_ids,
        gt_scores=gt_scores,
        vectors=vectors,
        ids=ids,
    )
    print("saved ground_truth.npz (queries, gt_ids, gt_scores, vectors, ids)")


if __name__ == "__main__":
    main()

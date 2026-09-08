"""
test_exact.py — standalone correctness + performance check for ExactIndex.
Run: python test_exact.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from index_exact import ExactIndex
from data_gen import build_dataset


def brute_force_topk(query, vectors, ids, k):
    q = query / np.clip(np.linalg.norm(query), 1e-12, None)
    X = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    scores = X @ q
    order = np.argsort(-scores)[:k]
    return ids[order], scores[order]


def test_correctness_small():
    rng = np.random.default_rng(0)
    n, d, k = 200, 16, 10
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    index = ExactIndex(dim=d)
    index.add(vectors, ids)

    for _ in range(5):
        q = rng.normal(size=d).astype(np.float32)
        got_ids, got_scores = index.search(q, k=k)
        exp_ids, exp_scores = brute_force_topk(q, vectors, ids, k)
        assert set(got_ids.tolist()) == set(exp_ids.tolist()), (got_ids, exp_ids)
        np.testing.assert_allclose(np.sort(got_scores), np.sort(exp_scores), atol=1e-5)
    print("PASS: correctness matches independent brute-force reference")


def test_insert_and_delete():
    d = 8
    index = ExactIndex(dim=d, initial_capacity=4)  # forces a grow() mid-test
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(10, d)).astype(np.float32)
    ids = np.arange(10, dtype=np.int64)
    index.add(vectors, ids)
    assert len(index) == 10

    index.delete([0, 1, 2])
    assert len(index) == 7

    got_ids, _ = index.search(vectors[0], k=10)
    assert 0 not in got_ids.tolist(), "deleted id leaked into results"
    print("PASS: insert grows capacity correctly, delete tombstones correctly")


def test_batch_queries_match_single():
    rng = np.random.default_rng(2)
    d, n, k = 32, 500, 10
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    index = ExactIndex(dim=d)
    index.add(vectors, ids)

    queries = rng.normal(size=(5, d)).astype(np.float32)
    batch_ids, batch_scores = index.search(queries, k=k)
    for i in range(5):
        single_ids, single_scores = index.search(queries[i], k=k)
        np.testing.assert_array_equal(batch_ids[i], single_ids)
        # float32 BLAS can take a different summation path for batched vs.
        # single-row matmul; tolerate the resulting ULP-level drift.
        np.testing.assert_allclose(batch_scores[i], single_scores, rtol=1e-5, atol=1e-6)
    print("PASS: batched search matches per-row single-query search")


def test_performance_50k():
    d, n_queries, k = 128, 200, 10
    vectors, ids, _, queries = build_dataset(
        n=50_000, d=d, n_components=100, n_queries=n_queries, seed=7
    )

    index = ExactIndex(dim=d)
    t0 = time.perf_counter()
    index.add(vectors, ids)
    build_s = time.perf_counter() - t0

    index.search(queries[:1], k=k)  # warm-up
    t0 = time.perf_counter()
    index.search(queries, k=k)  # batched
    batch_s = time.perf_counter() - t0

    print(
        f"PASS: build 50k x {d} in {build_s:.3f}s, batched search of "
        f"{n_queries} queries in {batch_s * 1000:.1f}ms "
        f"({n_queries / batch_s:.1f} QPS)"
    )
    assert batch_s < 5.0, "search unexpectedly slow"


if __name__ == "__main__":
    test_correctness_small()
    test_insert_and_delete()
    test_batch_queries_match_single()
    test_performance_50k()
    print("\nALL TESTS PASSED")

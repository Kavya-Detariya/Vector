"""
test_ivf.py — standalone correctness + recall sanity check for IVFFlatIndex.
Run: python test_ivf.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from index_ivf import IVFFlatIndex
from index_exact import ExactIndex
from data_gen import build_dataset


def test_train_add_shapes():
    rng = np.random.default_rng(0)
    n, d = 2000, 32
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    index = IVFFlatIndex(dim=d)
    index.train(vectors, n_iters=20)
    assert index.k == int(np.floor(np.sqrt(n))), index.k

    index.add(vectors, ids)
    assert len(index) == n
    total_in_postings = sum(p["vectors"].shape[0] for p in index.postings.values())
    assert total_in_postings == n
    print(f"PASS: trained k={index.k} clusters, all {n} vectors landed in postings")


def test_delete_excludes_from_search():
    rng = np.random.default_rng(1)
    n, d = 2000, 32
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    index = IVFFlatIndex(dim=d)
    index.train(vectors, n_iters=20)
    index.add(vectors, ids)

    got_ids, _ = index.search(vectors[0], k=10, nprobe=index.k)  # nprobe=all clusters
    assert 0 in got_ids.tolist(), "sanity: vector should find itself pre-delete"

    index.delete([0])
    assert len(index) == n - 1

    got_ids, got_scores = index.search(vectors[0], k=10, nprobe=index.k)
    assert 0 not in got_ids.tolist(), "deleted id leaked into IVF results"
    print("PASS: tombstoned id excluded from search even when probing all clusters")


def test_recall_improves_with_nprobe():
    N, D, N_COMPONENTS, N_QUERIES, K, SEED = 20_000, 128, 60, 200, 10, 42
    vectors, ids, _, queries = build_dataset(N, D, N_COMPONENTS, N_QUERIES, SEED)

    exact = ExactIndex(dim=D)
    exact.add(vectors, ids)
    gt_ids, _ = exact.search(queries, k=K)

    ivf = IVFFlatIndex(dim=D)
    t0 = time.perf_counter()
    ivf.train(vectors, n_iters=40)
    train_s = time.perf_counter() - t0
    ivf.add(vectors, ids)
    print(f"trained IVF: k={ivf.k} clusters in {train_s:.2f}s")

    recalls = {}
    for nprobe in (1, 2, 4, 8, 16, 32):
        pred_ids, _ = ivf.search(queries, k=K, nprobe=nprobe)
        hits = 0
        for i in range(N_QUERIES):
            hits += len(set(pred_ids[i].tolist()) & set(gt_ids[i].tolist()))
        recall = hits / (N_QUERIES * K)
        recalls[nprobe] = recall
        print(f"nprobe={nprobe:>2}  recall@{K}={recall:.3f}")

    probes = sorted(recalls)
    assert all(recalls[probes[i]] <= recalls[probes[i + 1]] + 1e-9 for i in range(len(probes) - 1)), (
        "recall should be monotonically non-decreasing with nprobe"
    )
    assert recalls[max(probes)] > 0.9, "recall at nprobe=32 unexpectedly low — check clustering"
    print("PASS: recall is monotonic in nprobe and high at large nprobe")


if __name__ == "__main__":
    test_train_add_shapes()
    test_delete_excludes_from_search()
    test_recall_improves_with_nprobe()
    print("\nALL TESTS PASSED")

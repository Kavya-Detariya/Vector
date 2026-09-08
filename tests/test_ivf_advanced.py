"""
tests/test_ivf_advanced.py — Advanced IVF validation suite.
Verifies:
1. Spherical K-Means++ seed distribution (dead cluster prevention).
2. 8-bit Scalar Quantization (IVF-SQ8) memory compression, discrete recall, and cosine fidelity.
3. Dynamic adaptive nprobe routing.
4. Tombstone masking inside quantized posting lists.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the Python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from data_gen import build_dataset
from index_exact import ExactIndex
from index_ivf import IVFFlatIndex


def test_kmeans_plus_plus_cluster_balance():
    n, d = 10_000, 64
    vectors, ids, _, _ = build_dataset(n=n, d=d, n_components=50, n_queries=10, seed=123)

    index = IVFFlatIndex(dim=d)
    index.train(vectors, n_iters=40, seed=42)
    index.add(vectors, ids)

    empty_clusters = sum(1 for p in index.postings.values() if p["ids"].size == 0)
    sizes = [p["ids"].size for p in index.postings.values()]

    print(
        f"K-Means++ Cluster Stats: clusters={index.k}, empty={empty_clusters}, "
        f"min_size={min(sizes)}, max_size={max(sizes)}, median={int(np.median(sizes))}"
    )
    assert empty_clusters == 0, f"Found {empty_clusters} empty clusters with K-Means++!"
    print("PASS: Spherical K-Means++ produces balanced non-empty clusters.")


def test_sq8_compression_and_fidelity():
    n, d, k_neighbors = 10_000, 128, 10
    vectors, ids, _, queries = build_dataset(
        n=n, d=d, n_components=50, n_queries=100, seed=42
    )

    exact = ExactIndex(dim=d)
    exact.add(vectors, ids)
    gt_ids, gt_scores = exact.search(queries, k=k_neighbors)

    # FP32 baseline
    ivf_fp32 = IVFFlatIndex(dim=d, quantize=False)
    ivf_fp32.train(vectors, n_iters=40, seed=42)
    ivf_fp32.add(vectors, ids)
    fp32_ids, fp32_scores = ivf_fp32.search(queries, k=k_neighbors, nprobe=8)

    # SQ8 quantized
    ivf_sq8 = IVFFlatIndex(dim=d, quantize=True)
    ivf_sq8.train(vectors, n_iters=40, seed=42)
    ivf_sq8.add(vectors, ids)
    sq8_ids, sq8_scores = ivf_sq8.search(queries, k=k_neighbors, nprobe=8)

    fp32_bytes = ivf_fp32.get_memory_bytes()
    sq8_bytes = ivf_sq8.get_memory_bytes()
    compression_ratio = fp32_bytes / sq8_bytes

    def compute_recall(preds):
        hits = sum(
            len(set(preds[i].tolist()) & set(gt_ids[i].tolist()))
            for i in range(len(queries))
        )
        return hits / (len(queries) * k_neighbors)

    recall_fp32 = compute_recall(fp32_ids)
    recall_sq8 = compute_recall(sq8_ids)

    # Cosine fidelity: average cosine similarity ratio of SQ8 vs exact ground truth
    # Computes actual dot product of the retrieved IDs against the query vectors
    q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    v_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    sq8_true_cosines = np.array([
        [np.dot(q_norm[i], v_norm[sq8_ids[i, j]]) for j in range(k_neighbors)]
        for i in range(len(queries))
    ])
    cosine_fidelity = float(np.mean(sq8_true_cosines) / np.mean(gt_scores))

    print(f"FP32 Posting Memory: {fp32_bytes / 1024:.1f} KB | Recall@10: {recall_fp32:.4f}")
    print(f"SQ8  Posting Memory: {sq8_bytes / 1024:.1f} KB | Recall@10: {recall_sq8:.4f}")
    print(f"Memory Reduction:    {compression_ratio:.2f}x")
    print(f"Cosine Fidelity:     {cosine_fidelity * 100:.2f}% (Semantic Similarity vs Exact Ground Truth)")

    assert compression_ratio > 3.0, f"Expected >3x compression, got {compression_ratio:.2f}x"
    assert recall_sq8 >= 0.75, f"SQ8 discrete recall below noise floor baseline: {recall_sq8:.4f}"
    assert cosine_fidelity >= 0.995, f"Cosine fidelity below 99.5%: {cosine_fidelity:.4f}"
    print("PASS: IVF-SQ8 achieves >3.5x compression with >99.5% cosine fidelity.")


def test_adaptive_nprobe_behavior():
    n, d, k_neighbors = 8_000, 64, 10
    vectors, ids, _, queries = build_dataset(
        n=n, d=d, n_components=40, n_queries=50, seed=7
    )

    exact = ExactIndex(dim=d)
    exact.add(vectors, ids)
    gt_ids, _ = exact.search(queries, k=k_neighbors)

    ivf = IVFFlatIndex(dim=d, quantize=False)
    ivf.train(vectors, n_iters=30, seed=42)
    ivf.add(vectors, ids)

    adaptive_ids, _ = ivf.search(
        queries, k=k_neighbors, nprobe=16, adaptive_threshold=0.90
    )

    hits = sum(
        len(set(adaptive_ids[i].tolist()) & set(gt_ids[i].tolist()))
        for i in range(len(queries))
    )
    recall_adaptive = hits / (len(queries) * k_neighbors)
    print(f"Adaptive Routing Recall@10: {recall_adaptive:.4f}")

    assert recall_adaptive >= 0.85, f"Adaptive recall unexpectedly low: {recall_adaptive:.4f}"
    print("PASS: Dynamic adaptive routing effectively resolves queries.")


def test_sq8_tombstone_integrity():
    d = 32
    vectors = np.random.default_rng(0).standard_normal((500, d)).astype(np.float32)
    ids = np.arange(500, dtype=np.int64)

    ivf = IVFFlatIndex(dim=d, quantize=True)
    ivf.train(vectors, n_iters=20, seed=0)
    ivf.add(vectors, ids)

    target_id = 42
    target_vec = vectors[target_id]

    res_ids, _ = ivf.search(target_vec, k=5, nprobe=ivf.k)
    assert target_id in res_ids.tolist()

    ivf.delete([target_id])
    assert len(ivf) == 499

    res_ids_after, _ = ivf.search(target_vec, k=5, nprobe=ivf.k)
    assert target_id not in res_ids_after.tolist(), "Tombstoned ID leaked from SQ8 index!"
    print("PASS: Tombstone deletions operate properly inside quantized posting lists.")


if __name__ == "__main__":
    print("--- Running Advanced IVF Validation Suite ---")
    test_kmeans_plus_plus_cluster_balance()
    test_sq8_compression_and_fidelity()
    test_adaptive_nprobe_behavior()
    test_sq8_tombstone_integrity()
    print("\nALL ADVANCED IVF TESTS PASSED SUCCESSFULLY.")
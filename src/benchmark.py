"""
benchmark.py — IVF-Flat vs exact-index benchmark.

Datasets:
  --dataset synthetic   50k x 128 Gaussian-mixture vectors (default, no
                        external dependencies — always runnable).
  --dataset real        5,000 x 384 sentence embeddings loaded from
                        text_embeddings.npz (produced by embed_text.py,
                        which needs network access to download the model).
                        500 of the 5,000 corpus embeddings are held out
                        as queries against the full indexed set; this
                        measures IVF/exact agreement identically to the
                        synthetic case, just on real embedding geometry.

For each nprobe in {1,2,4,8,16,32}, reports Recall@10 against the exact
ground truth, p50/p99 latency, and QPS, then writes a CSV and a
Recall-vs-QPS Pareto plot.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from index_exact import ExactIndex
from index_ivf import IVFFlatIndex
from data_gen import build_dataset

K = 10
NPROBES = (1, 2, 4, 8, 16, 32)
N_QUERIES = 500


def _timed_search(index, queries, k, **search_kwargs):
    """Per-query timing loop (orchestration, not vector math) + result arrays."""
    n = queries.shape[0]
    latencies = np.empty(n, dtype=np.float64)
    ids = np.empty((n, k), dtype=np.int64)
    scores = np.empty((n, k), dtype=np.float32)
    index.search(queries[:1], k=k, **search_kwargs) if search_kwargs else index.search(queries[:1], k=k)
    for i in range(n):
        t0 = time.perf_counter()
        ids_i, scores_i = index.search(queries[i], k=k, **search_kwargs) if search_kwargs else index.search(queries[i], k=k)
        latencies[i] = time.perf_counter() - t0
        ids[i] = ids_i
        scores[i] = scores_i
    return ids, scores, latencies


def _recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    n = pred_ids.shape[0]
    hits = 0
    for i in range(n):
        hits += len(set(pred_ids[i, :k].tolist()) & set(gt_ids[i, :k].tolist()))
    return hits / (n * k)


def load_synthetic():
    N, D, N_COMPONENTS = 50_000, 128, 100
    vectors, ids, _, queries = build_dataset(N, D, N_COMPONENTS, N_QUERIES, seed=42)
    return vectors, ids, queries, D


def load_real(path: str = "text_embeddings.npz"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. Run embed_text.py first (needs internet access) "
            "to generate it, then re-run this script."
        )
    data = np.load(p, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    n, d = embeddings.shape
    ids = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(42)
    query_idx = rng.choice(n, size=min(N_QUERIES, n), replace=False)
    queries = embeddings[query_idx]
    return embeddings, ids, queries, d


def run(dataset: str):
    if dataset == "synthetic":
        vectors, ids, queries, d = load_synthetic()
    elif dataset == "real":
        vectors, ids, queries, d = load_real()
    else:
        raise ValueError(dataset)

    print(f"[{dataset}] N={vectors.shape[0]} d={d} queries={queries.shape[0]}")

    exact = ExactIndex(dim=d)
    exact.add(vectors, ids)
    exact.search(queries[:1], k=K)  # warm-up
    gt_ids, _, exact_latencies = _timed_search(exact, queries, K)
    exact_p50 = np.percentile(exact_latencies, 50) * 1000
    exact_p99 = np.percentile(exact_latencies, 99) * 1000
    exact_qps = 1.0 / np.mean(exact_latencies)
    print(f"exact   p50={exact_p50:.3f}ms  p99={exact_p99:.3f}ms  QPS={exact_qps:.1f}")

    ivf = IVFFlatIndex(dim=d)
    t0 = time.perf_counter()
    ivf.train(vectors, n_iters=60)
    train_s = time.perf_counter() - t0
    ivf.add(vectors, ids)
    print(f"IVF trained: k={ivf.k} clusters in {train_s:.2f}s")

    rows = [{
        "nprobe": "exact",
        "recall_at_10": 1.0,
        "p50_ms": exact_p50,
        "p99_ms": exact_p99,
        "qps": exact_qps,
        "speedup_vs_exact": 1.0,
    }]

    for nprobe in NPROBES:
        ivf.search(queries[:1], k=K, nprobe=nprobe)  # warm-up
        pred_ids, _, latencies = _timed_search(ivf, queries, K, nprobe=nprobe)
        recall = _recall_at_k(pred_ids, gt_ids, K)
        p50 = np.percentile(latencies, 50) * 1000
        p99 = np.percentile(latencies, 99) * 1000
        qps = 1.0 / np.mean(latencies)
        rows.append({
            "nprobe": nprobe,
            "recall_at_10": recall,
            "p50_ms": p50,
            "p99_ms": p99,
            "qps": qps,
            "speedup_vs_exact": qps / exact_qps,
        })
        print(f"nprobe={nprobe:>2}  recall@10={recall:.3f}  p50={p50:.3f}ms  "
              f"p99={p99:.3f}ms  QPS={qps:.1f}  speedup={qps / exact_qps:.2f}x")

    csv_path = f"benchmark_{dataset}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")

    plot_pareto(rows, dataset)


def plot_pareto(rows, dataset: str):
    ivf_rows = [r for r in rows if r["nprobe"] != "exact"]
    recalls = np.array([r["recall_at_10"] for r in ivf_rows])
    qpss = np.array([r["qps"] for r in ivf_rows])
    nprobes = [r["nprobe"] for r in ivf_rows]

    # Non-dominated frontier: a point is dominated if another point has
    # both >= recall and >= QPS (and is strictly better in at least one).
    order = np.argsort(-qpss)
    frontier_mask = np.zeros(len(ivf_rows), dtype=bool)
    best_recall_so_far = -1.0
    for idx in order:
        if recalls[idx] > best_recall_so_far:
            frontier_mask[idx] = True
            best_recall_so_far = recalls[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(qpss[~frontier_mask], recalls[~frontier_mask], color="tab:gray", label="dominated")
    ax.scatter(qpss[frontier_mask], recalls[frontier_mask], color="tab:red", label="Pareto frontier")
    order_f = np.argsort(qpss[frontier_mask])
    ax.plot(qpss[frontier_mask][order_f], recalls[frontier_mask][order_f], color="tab:red", linewidth=1)
    for i, np_ in enumerate(nprobes):
        ax.annotate(f"nprobe={np_}", (qpss[i], recalls[i]), fontsize=8,
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("QPS")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"IVF-Flat Recall/QPS Pareto — {dataset}")
    ax.legend()
    fig.tight_layout()
    out_path = f"pareto_{dataset}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()
    run(args.dataset)

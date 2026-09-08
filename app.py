from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.data_gen import build_dataset
from src.index_exact import ExactIndex
from src.index_ivf import IVFFlatIndex

st.set_page_config(
    page_title="VectorDB Internals — IVF-Flat vs Exact",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Pure-NumPy PCA (Zero External Dependencies) ----------------

def fit_pca_2d(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes top-2 principal projection components via covariance eigendecomposition.
    Strictly pure NumPy; zero sklearn/scipy imports.
    """
    mean = np.mean(vectors, axis=0, keepdims=True)
    centered = vectors - mean
    # For N >> d, eigendecomposition of (d, d) covariance runs in < 20ms
    cov = (centered.T @ centered) / max(vectors.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    components = eigenvectors[:, -2:][:, ::-1].T  # Shape: (2, d)
    return mean, components


def project_pca_2d(
    vectors: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    centered = vectors - mean
    return (centered @ components.T).astype(np.float32)


# ---------------- Index & Data Initialization (Cached) ----------------

@st.cache_resource(show_spinner="Indexing Real-Text Embeddings...")
def load_text_index() -> Tuple[ExactIndex, IVFFlatIndex, np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:
    path = Path("text_embeddings.npz")
    if not path.exists():
        raise FileNotFoundError("text_embeddings.npz not found")

    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    texts = [str(t) for t in data["texts"].tolist()]
    labels = [str(l) for l in data["labels"].tolist()]
    n, d = embeddings.shape
    ids = np.arange(n, dtype=np.int64)

    exact = ExactIndex(dim=d)
    exact.add(embeddings, ids)

    ivf = IVFFlatIndex(dim=d)
    ivf.train(embeddings, n_iters=60, seed=42)
    ivf.add(embeddings, ids)

    mean, components = fit_pca_2d(embeddings)
    return exact, ivf, embeddings, texts, labels, mean, components


@st.cache_resource(show_spinner="Synthesizing 50,000 Clustered Vectors...")
def load_synthetic_index() -> Tuple[ExactIndex, IVFFlatIndex, np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:
    n, d = 50_000, 128
    vectors, ids, assignments, _ = build_dataset(n=n, d=d, n_components=100, n_queries=10, seed=42)
    texts = [f"Synthetic Vector #{i} (Cluster {assignments[i]})" for i in range(n)]
    labels = [f"Cluster_{assignments[i]}" for i in range(n)]

    exact = ExactIndex(dim=d)
    exact.add(vectors, ids)

    ivf = IVFFlatIndex(dim=d)
    ivf.train(vectors, n_iters=60, seed=42)
    ivf.add(vectors, ids)

    mean, components = fit_pca_2d(vectors[:5000])  # Subsample for PCA projection basis
    return exact, ivf, vectors, texts, labels, mean, components


@st.cache_resource(show_spinner="Loading Embedding Model...")
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


# ---------------- UI & Interaction ----------------

st.title("Vector Database Internals: Exact vs. IVF-Flat")
st.caption("Zero external vector search engines. Pure NumPy vector operations and memory indexing.")

text_db_available = Path("text_embeddings.npz").exists()

with st.sidebar:
    st.header("Index Configuration")
    dataset_choice = st.radio(
        "Active Dataset",
        options=["5,000 Real Texts (d=384)", "50,000 Synthetic Vectors (d=128)"],
        index=0 if text_db_available else 1,
    )

    if dataset_choice.startswith("5,000") and not text_db_available:
        st.error("`text_embeddings.npz` not found. Run `python embed_text.py` first.")
        st.stop()

    is_text = dataset_choice.startswith("5,000")

    if is_text:
        exact, ivf, raw_vectors, corpus_texts, corpus_labels, pca_mean, pca_components = load_text_index()
    else:
        exact, ivf, raw_vectors, corpus_texts, corpus_labels, pca_mean, pca_components = load_synthetic_index()

    st.markdown("---")
    st.subheader("Search Parameters")
    nprobe = st.slider("IVF Probed Clusters (nprobe)", min_value=1, max_value=min(ivf.k, 32), value=4)
    top_k = st.slider("Top-K Neighbors", min_value=1, max_value=20, value=10)

    st.markdown("---")
    st.subheader("Live Tombstone Mutation")
    delete_id_input = st.number_input("Delete Vector ID", min_value=0, max_value=len(raw_vectors) - 1, value=0)
    if st.button("Tombstone ID"):
        exact.delete([delete_id_input])
        ivf.delete([delete_id_input])
        st.warning(f"Vector ID {delete_id_input} tombstoned across Exact and IVF indexes.")

# ---------------- Query Vector Resolution ----------------

query_vector: np.ndarray | None = None
query_label = ""

if is_text:
    model = get_sentence_transformer()
    user_query = st.text_input(
        "Natural Language Semantic Query:",
        value="A sudden storm rolled into downtown causing heavy rain",
    )
    if model is not None and user_query.strip():
        q_emb = model.encode(user_query, convert_to_numpy=True).astype(np.float32)
        query_vector = q_emb
        query_label = user_query
    else:
        st.warning("sentence-transformers unavailable. Reverting to row 0 vector.")
        query_vector = raw_vectors[0]
        query_label = corpus_texts[0]
else:
    vector_idx = st.slider("Synthetic Query Vector Index", 0, len(raw_vectors) - 1, 42)
    query_vector = raw_vectors[vector_idx]
    query_label = f"Vector #{vector_idx}"

# ---------------- Query Execution & Profiling ----------------

# Exact search profiling (averaging over repeated runs for smooth sub-millisecond precision)
n_profile_runs = 5
t0 = time.perf_counter()
for _ in range(n_profile_runs):
    exact_ids, exact_scores = exact.search(query_vector, k=top_k)
exact_time_ms = ((time.perf_counter() - t0) / n_profile_runs) * 1000.0

# IVF search profiling
t0 = time.perf_counter()
for _ in range(n_profile_runs):
    ivf_ids, ivf_scores = ivf.search(query_vector, k=top_k, nprobe=nprobe)
ivf_time_ms = ((time.perf_counter() - t0) / n_profile_runs) * 1000.0

# Metric evaluation
exact_set = set(exact_ids.tolist())
ivf_set = set(ivf_ids.tolist())
hits = len(exact_set & ivf_set)
recall = hits / top_k
speedup = exact_time_ms / max(ivf_time_ms, 1e-6)

# Determine probed cluster assignments for visualization
q_norm = query_vector / np.maximum(np.linalg.norm(query_vector), 1e-12)
centroid_sims = q_norm @ ivf.centroids.T
nprobe_eff = min(nprobe, ivf.k)
probed_cluster_indices = set(np.argpartition(-centroid_sims, kth=nprobe_eff - 1)[:nprobe_eff].tolist())

# ---------------- Metric Dashboard Cards ----------------

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recall@K", f"{recall * 100:.1f}%", help="Fraction of exact ground-truth neighbors captured by IVF-Flat.")
m2.metric("IVF Latency", f"{ivf_time_ms:.2f} ms")
m3.metric("Exact Latency", f"{exact_time_ms:.2f} ms")
m4.metric("Throughput Speedup", f"{speedup:.2f}x", delta=f"{(speedup - 1.0):.2f}x")

# ---------------- Visualization & Inspection Layout ----------------

col_vis, col_results = st.columns([1.1, 0.9])

with col_vis:
    st.subheader("2D Manifold & Cluster Probe Visualization")

    # Sample points for responsive plotting
    sample_size = min(3000, len(raw_vectors))
    sample_idx = np.linspace(0, len(raw_vectors) - 1, sample_size, dtype=int)
    sampled_vectors = raw_vectors[sample_idx]

    pts_2d = project_pca_2d(sampled_vectors, pca_mean, pca_components)
    centroids_2d = project_pca_2d(ivf.centroids, pca_mean, pca_components)
    query_2d = project_pca_2d(query_vector[None, :], pca_mean, pca_components)[0]
    exact_top_2d = project_pca_2d(raw_vectors[exact_ids], pca_mean, pca_components)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    # Background vector projection
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c="#334155", s=8, alpha=0.35, label="Corpus Vectors")

    # Unprobed vs Probed Centroids
    unprobed_idx = [i for i in range(ivf.k) if i not in probed_cluster_indices]
    probed_idx = list(probed_cluster_indices)

    if unprobed_idx:
        ax.scatter(
            centroids_2d[unprobed_idx, 0],
            centroids_2d[unprobed_idx, 1],
            c="#64748B",
            s=35,
            alpha=0.6,
            marker="o",
            label="Unprobed Centroids",
        )
    ax.scatter(
        centroids_2d[probed_idx, 0],
        centroids_2d[probed_idx, 1],
        c="#F59E0B",
        s=110,
        alpha=0.95,
        marker="X",
        edgecolors="white",
        linewidths=1.2,
        label=f"Probed Centroids ({nprobe})",
    )

    # Ground-truth hits and query marker
    ax.scatter(
        exact_top_2d[:, 0],
        exact_top_2d[:, 1],
        c="#10B981",
        s=80,
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        label="Exact Top-K",
    )
    ax.scatter(
        query_2d[0],
        query_2d[1],
        c="#EF4444",
        s=180,
        marker="*",
        edgecolors="white",
        linewidths=1.5,
        label="Query Vector",
    )

    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8, loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)

with col_results:
    st.subheader(f"Top-{top_k} Matched Results")
    st.caption(f"Query: '{query_label}'")

    results_table = []
    for rank in range(top_k):
        ivf_id = int(ivf_ids[rank])
        ivf_score = float(ivf_scores[rank])
        exact_id = int(exact_ids[rank])
        exact_score = float(exact_scores[rank])

        in_exact_topk = ivf_id in exact_set
        marker = "MATCH" if in_exact_topk else "MISS"

        text_preview = corpus_texts[ivf_id] if ivf_id < len(corpus_texts) else "N/A"
        label_preview = corpus_labels[ivf_id] if ivf_id < len(corpus_labels) else "N/A"

        results_table.append({
            "Rank": rank + 1,
            "Recall": marker,
            "IVF ID": ivf_id,
            "Cosine Sim": f"{ivf_score:.4f}",
            "Domain": label_preview,
            "Snippet": text_preview,
        })

    st.dataframe(results_table, width='stretch', height=450)
"""
app.py — Text-first semantic search interface and vector database visualizer.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_gen import build_dataset
from index_exact import ExactIndex
from index_ivf import IVFFlatIndex

st.set_page_config(
    page_title="Semantic Vector Database",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Pure-NumPy Dimensionality Reduction ----------------

def fit_pca_2d(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(vectors, axis=0, keepdims=True)
    centered = vectors - mean
    cov = (centered.T @ centered) / max(vectors.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    components = eigenvectors[:, -2:][:, ::-1].T
    return mean, components


def project_pca_2d(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    centered = vectors - mean
    return (centered @ components.T).astype(np.float32)


# ---------------- Cached Index Containers ----------------

class IndexContainer:
    """Holds mutable index and text corpus state in RAM across Streamlit reruns."""
    def __init__(
        self,
        exact: ExactIndex,
        ivf: IVFFlatIndex,
        vectors: np.ndarray,
        texts: List[str],
        labels: List[str],
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
    ):
        self.exact = exact
        self.ivf = ivf
        self.vectors = vectors
        self.texts = texts
        self.labels = labels
        self.pca_mean = pca_mean
        self.pca_components = pca_components


@st.cache_resource(show_spinner="Loading and Indexing Real-Text Corpus...")
def load_text_index() -> IndexContainer | None:
    path = DATA_DIR / "text_embeddings.npz"
    if not path.exists():
        return None

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
    return IndexContainer(exact, ivf, embeddings, texts, labels, mean, components)


@st.cache_resource(show_spinner="Synthesizing 50,000 Clustered Vectors...")
def load_synthetic_index() -> IndexContainer:
    n, d = 50_000, 128
    vectors, ids, assignments, _ = build_dataset(n=n, d=d, n_components=100, n_queries=10, seed=42)
    texts = [f"Synthetic Record #{i} [Category: Cluster_{assignments[i]}]" for i in range(n)]
    labels = [f"Cluster_{assignments[i]}" for i in range(n)]

    exact = ExactIndex(dim=d)
    exact.add(vectors, ids)

    ivf = IVFFlatIndex(dim=d)
    ivf.train(vectors, n_iters=60, seed=42)
    ivf.add(vectors, ids)

    mean, components = fit_pca_2d(vectors[:5000])
    return IndexContainer(exact, ivf, vectors, texts, labels, mean, components)


@st.cache_resource(show_spinner="Loading Embedding Model (sentence-transformers)...")
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


# ---------------- Sidebar Controls ----------------

text_embeddings_file = DATA_DIR / "text_embeddings.npz"
text_db_available = text_embeddings_file.exists()

with st.sidebar:
    st.header("Database Engine Config")
    dataset_choice = st.radio(
        "Corpus Mode",
        options=["5,000 Real Texts (d=384)", "50,000 Synthetic Vectors (d=128)"],
        index=0 if text_db_available else 1,
    )

    is_text = dataset_choice.startswith("5,000")
    if is_text and not text_db_available:
        st.error("`data/text_embeddings.npz` not found. Run `uv run python src/embed_text.py` first.")
        st.stop()

    container = load_text_index() if is_text else load_synthetic_index()
    if container is None:
        st.error("Failed to load vector index.")
        st.stop()

    exact = container.exact
    ivf = container.ivf

    model = get_sentence_transformer() if is_text else None

    st.markdown("---")
    st.subheader("Tuning Knobs")
    nprobe = st.slider("IVF Probed Clusters (nprobe)", min_value=1, max_value=min(ivf.k, 32), value=4)
    top_k = st.slider("Retrieve Top-K Matches", min_value=1, max_value=20, value=7)

    st.markdown("---")
    st.subheader("Document Deletion")
    del_mode = st.radio("Deletion Method", ["By ID", "By Semantic Similarity"], horizontal=True)

    if del_mode == "By ID":
        delete_id = st.number_input("Delete by ID", min_value=0, max_value=max(len(container.vectors) - 1, 0), value=0)
        if st.button("Delete Document"):
            exact.delete([delete_id])
            ivf.delete([delete_id])
            st.warning(f"Document ID {delete_id} tombstoned from active index.")
    else:
        if is_text:
            del_query = st.text_input("Purge topics matching:", value="heavy rain and storms")
            del_threshold = st.slider("Min Cosine Similarity Threshold", 0.50, 0.95, 0.75, 0.05)
            max_delete_count = st.slider("Max Documents to Purge", 1, 10, 3)

            if st.button("Find & Delete by Similarity"):
                if model is not None and del_query.strip():
                    # 1. Embed deletion prompt
                    del_vec = model.encode(del_query, convert_to_numpy=True).astype(np.float32)
                    
                    # 2. Find nearest candidates via exact search
                    cand_ids, cand_scores = exact.search(del_vec, k=max_delete_count)
                    
                    # 3. Filter candidates above similarity threshold
                    to_delete = [
                        int(cand_ids[i])
                        for i in range(len(cand_ids))
                        if cand_scores[i] >= del_threshold and cand_ids[i] != -1
                    ]

                    if to_delete:
                        # 4. Tombstone in both indices
                        exact.delete(to_delete)
                        ivf.delete(to_delete)
                        st.error(f"Tombstoned {len(to_delete)} documents matching '{del_query}' (Sim >= {del_threshold}):")
                        for d_id in to_delete:
                            st.caption(f"• ID #{d_id}: {container.texts[d_id]}")
                    else:
                        st.info(f"No active documents found with similarity >= {del_threshold}.")
        else:
            st.info("Switch to '5,000 Real Texts' to use semantic similarity deletion.")


# ---------------- Main Page: Custom Document Insertion & Search ----------------

st.title("Zero-Dependency Semantic Vector Database")
st.caption("Search, insert, and retrieve text documents using pure-NumPy IVF-Flat indexing.")

model = get_sentence_transformer() if is_text else None

# Custom Document Insertion Accordion
with st.expander("Insert Custom Text Document into Database", expanded=False):
    col_ins1, col_ins2 = st.columns([3, 1])
    with col_ins1:
        custom_doc_text = st.text_input(
            "Document Text:",
            value="Quantum computing algorithms drastically reduce cryptography simulation time.",
            placeholder="Type any sentence or paragraph here...",
        )
    with col_ins2:
        custom_doc_domain = st.selectbox(
            "Domain Category:",
            ["technology", "science", "health", "weather", "finance", "custom"],
        )

    if st.button("Insert Document into Vector DB"):
        if is_text and model is not None and custom_doc_text.strip():
            # 1. Embed text to float32 vector
            new_vec = model.encode(custom_doc_text, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
            new_id = len(container.texts)
            # 2. Add to both indices
            exact.add(new_vec, np.array([new_id], dtype=np.int64))
            ivf.add(new_vec, np.array([new_id], dtype=np.int64))
            # 3. Add to metadata registry
            container.texts.append(custom_doc_text)
            container.labels.append(custom_doc_domain)
            container.vectors = np.concatenate([container.vectors, new_vec], axis=0)
            st.success(f"Document indexed with ID #{new_id}! Query for it above to test retrieval.")
        elif not is_text:
            st.info("Switch to '5,000 Real Texts' mode in the sidebar to insert real text documents.")

# Custom Text Query Input
st.subheader("🔍 Semantic Neural Search")
st.caption("Query the vector space using natural language concepts, semantics, or keywords.")

if is_text:
    user_query = st.text_input(
        "Enter any search phrase or statement:",
        placeholder="🔍 Type a statement, question, or scenario to find nearest vectors...",
    )
    if model is not None and user_query.strip():
        query_vector = model.encode(user_query, convert_to_numpy=True).astype(np.float32)
        query_label = user_query
    else:
        query_vector = container.vectors[0]
        query_label = container.texts[0]
else:
    vector_idx = st.slider("Select Query Vector Index", 0, len(container.vectors) - 1, 42)
    query_vector = container.vectors[vector_idx]
    query_label = container.texts[vector_idx]
    st.info(f"Using synthetic vector #{vector_idx} as query.")


# ---------------- Search Execution & Profiling ----------------

n_profile_runs = 5
t0 = time.perf_counter()
for _ in range(n_profile_runs):
    exact_ids, exact_scores = exact.search(query_vector, k=top_k)
exact_time_ms = ((time.perf_counter() - t0) / n_profile_runs) * 1000.0

t0 = time.perf_counter()
for _ in range(n_profile_runs):
    ivf_ids, ivf_scores = ivf.search(query_vector, k=top_k, nprobe=nprobe)
ivf_time_ms = ((time.perf_counter() - t0) / n_profile_runs) * 1000.0

exact_set = set(exact_ids.tolist())
ivf_set = set(ivf_ids.tolist())
recall = len(exact_set & ivf_set) / max(top_k, 1)
speedup = exact_time_ms / max(ivf_time_ms, 1e-6)

# Probed cluster evaluation
q_norm = query_vector / np.maximum(np.linalg.norm(query_vector), 1e-12)
centroid_sims = q_norm @ ivf.centroids.T
nprobe_eff = min(nprobe, ivf.k)
probed_cluster_indices = set(np.argpartition(-centroid_sims, kth=nprobe_eff - 1)[:nprobe_eff].tolist())

# Pruning & Accuracy calculation
total_vectors = len(exact)
ivf_candidates_scanned = sum(
    int(ivf.postings[c]["active"].sum()) for c in probed_cluster_indices
)
pruned_percent = (1.0 - (ivf_candidates_scanned / max(total_vectors, 1))) * 100.0

mean_exact_score = float(np.mean(exact_scores[:top_k])) if len(exact_scores) else 1.0
mean_ivf_score = float(np.mean(ivf_scores[:top_k])) if len(ivf_scores) else 1.0
score_fidelity = (mean_ivf_score / max(mean_exact_score, 1e-6)) * 100.0


# ---------------- Search Space & Latency Metrics ----------------

st.markdown("### Search Space Pruning & Performance")

row1_1, row1_2, row1_3 = st.columns(3)
row1_1.metric("Total Indexed Documents", f"{total_vectors:,} docs")
row1_2.metric("Candidates Evaluated", f"{ivf_candidates_scanned:,} docs", delta=f"-{pruned_percent:.1f}% scanned")
row1_3.metric("Search Space Pruned", f"{pruned_percent:.1f}%")

row2_1, row2_2, row2_3 = st.columns(3)
row2_1.metric("Discrete Recall@K", f"{recall * 100:.1f}%", help="Exact ID match agreement with brute force.")
row2_2.metric("Semantic Score Fidelity", f"{score_fidelity:.2f}%", help="Cosine similarity of IVF results relative to true nearest neighbors.")
row2_3.metric("Search Speedup", f"{speedup:.2f}x", delta=f"{exact_time_ms:.2f}ms exact vs {ivf_time_ms:.2f}ms ivf")

st.markdown("---")


# ---------------- Side-by-Side Visualizations ----------------

col_vis_left, col_vis_right = st.columns(2)

with col_vis_left:
    st.subheader("2D Manifold & Cluster Probe Visualization")
    sample_size = min(3000, len(container.vectors))
    sample_idx = np.linspace(0, len(container.vectors) - 1, sample_size, dtype=int)
    sampled_vectors = container.vectors[sample_idx]

    pts_2d = project_pca_2d(sampled_vectors, container.pca_mean, container.pca_components)
    centroids_2d = project_pca_2d(ivf.centroids, container.pca_mean, container.pca_components)
    query_2d = project_pca_2d(query_vector[None, :], container.pca_mean, container.pca_components)[0]
    
    valid_exact_ids = [idx for idx in exact_ids if idx < len(container.vectors)]
    exact_top_2d = project_pca_2d(container.vectors[valid_exact_ids], container.pca_mean, container.pca_components)

    fig, ax = plt.subplots(figsize=(7, 4.8), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c="#334155", s=8, alpha=0.35, label="Indexed Corpus")

    unprobed = [i for i in range(ivf.k) if i not in probed_cluster_indices]
    probed = list(probed_cluster_indices)

    if unprobed:
        ax.scatter(centroids_2d[unprobed, 0], centroids_2d[unprobed, 1], c="#64748B", s=28, alpha=0.5, label="Unprobed Centroids")
    ax.scatter(centroids_2d[probed, 0], centroids_2d[probed, 1], c="#F59E0B", s=110, alpha=0.95, marker="X", edgecolors="white", label=f"Probed ({nprobe})")
    ax.scatter(exact_top_2d[:, 0], exact_top_2d[:, 1], c="#10B981", s=80, marker="o", edgecolors="white", label="Exact Ground Truth")
    ax.scatter(query_2d[0], query_2d[1], c="#EF4444", s=180, marker="*", edgecolors="white", label="Query")

    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8, loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)

with col_vis_right:
    st.subheader("Recall vs. QPS Pareto Frontier")
    pareto_img = DATA_DIR / ("pareto_real.png" if is_text else "pareto_synthetic.png")
    if pareto_img.exists():
        st.image(str(pareto_img), caption="Precomputed IVF vs. Exact Frontier", use_container_width=True)
    else:
        st.info("Run `uv run python src/benchmark.py` to generate the Pareto trade-off plot.")


# ---------------- Full-Width Retrieved Text Results ----------------

st.markdown("---")
st.subheader(f"Retrieved Top-{top_k} Results")
st.caption(f"Semantic Query: '{query_label}'")

results_table = []
for rank in range(min(top_k, len(ivf_ids))):
    ivf_id = int(ivf_ids[rank])
    ivf_score = float(ivf_scores[rank])
    in_exact = ivf_id in exact_set

    text_content = container.texts[ivf_id] if ivf_id < len(container.texts) else "N/A"
    category = container.labels[ivf_id] if ivf_id < len(container.labels) else "N/A"

    results_table.append({
        "Rank": rank + 1,
        "Match Status": "Exact Match" if in_exact else "Approx Match",
        "Similarity": f"{ivf_score:.4f}",
        "Category": category,
        "Document Text": text_content,
        "Internal ID": ivf_id,
    })

st.dataframe(results_table, use_container_width=True, hide_index=True)
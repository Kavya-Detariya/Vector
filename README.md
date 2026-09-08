# VectorDB from Scratch (Pure-NumPy Approximate Nearest Neighbors)

A zero-dependency, high-performance Vector Database built strictly with Python and **NumPy** for all core indexing, clustering, and search-path arithmetic. 

This engine implements an exact brute-force index (L2-normalized cosine via Level-3 BLAS GEMM) alongside an approximate **IVF-Flat** index featuring **Spherical K-Means++**, optional **8-bit Scalar Quantization (IVF-SQ8)**, dynamic adaptive probe routing, and zero-copy tombstone deletions.

---
## Production vs. Mocked Implementation Details

* **Real & Built From Scratch**:
  * Exact brute-force vector search via BLAS GEMM (`Q @ X.T`) and localized partitioning.
  * Spherical K-Means++ clustering and centroid update loop.
  * Inverted File (IVF-Flat) posting list management, dynamic routing, and search pruning.
  * 8-bit Scalar Quantization (SQ8) dynamic scaling and distance evaluation.
  * Live vector and document insertion, tombstone-based deletions, and semantic threshold purging.
* **Mocked / Out-of-Scope (Educational Scope)**:
  * **Persistence**: Indices and posting lists live in process memory (`RAM`); they reload from generated `.npz` files rather than using a memory-mapped disk layout (`np.memmap`) or WAL (Write-Ahead Log).
  * **Networking / RPC**: The engine runs as an in-process library and Streamlit application rather than exposing a standalone gRPC/REST HTTP microservice.
  * **Concurrency**: Thread-safety locks (`RWLock`) for concurrent reader-writer workers are omitted.

---
## Key Architectural Highlights

* **Pure Vectorized Operations**: No Python loops in the search or distance calculation paths. Centroid training utilizes one-hot GEMM updates, and candidate partitioning uses `np.argpartition` for $O(N)$ candidate selection followed by localized sorting.
* **Spherical K-Means++**: Eliminates dead and empty Voronoi cells on anisotropic sentence embeddings by seeding initial centroids proportional to geodesic angular dispersion ($1.0 - \cos \theta$).
* **IVF-SQ8 Quantization**: Reduces memory footprint by **3.8x** (from 4 bytes/dim to 1 byte/dim) via per-vector dynamic range scaling while preserving $>99.5\%$ cosine fidelity against FP32 ground truth.
* **Sub-linear Inverted File Pruning**: Partitions $N$ vectors into $k \approx \lfloor\sqrt{N}\rfloor$ clusters, querying only $nprobe$ nearest Voronoi lists to minimize candidate scoring.
* **Tombstone Masking**: $O(1)$ soft deletions preserve contiguous memory blocks for SIMD BLAS pipelines without triggering per-query array reallocations.

---

## Project Structure

```text
vector-db/
├── pyproject.toml              # Project dependencies and packaging configuration
├── uv.lock                     # Deterministic dependency lockfile
├── .gitignore                  # Ignores venvs, cache, .npz dumps, and plot images
├── README.md                   # System documentation
├── app.py                      # Interactive Streamlit dashboard and semantic visualizer
├── data/                       # Directory for generated embeddings and binary artifacts
├── src/
│   ├── __init__.py
│   ├── index_exact.py          # Ground-truth brute-force index (GEMM dot product)
│   ├── index_ivf.py            # IVF-Flat + IVF-SQ8 approximate index with K-Means++
│   ├── data_gen.py             # Deterministic Gaussian-mixture vector generator
│   ├── text_corpus.py          # Deterministic 10-domain short-text corpus generator
│   ├── embed_text.py           # Text embedding pipeline (MiniLM-L6-v2)
│   ├── evaluate.py             # Exact baseline evaluator (produces ground_truth.npz)
│   └── benchmark.py            # Latency (p50/p99), QPS, Recall@10 & Pareto analyzer
└── tests/
    ├── __init__.py
    ├── test_exact.py           # Correctness, deletion, and batch parity tests for exact index
    ├── test_ivf.py             # Monotonic recall and shape tests for IVF-Flat
    └── test_ivf_advanced.py    # K-Means++ balance, SQ8 fidelity, and dynamic probe tests
Installation & SetupThis project is managed using uv for reproducible, fast virtual environments. Standard pip is also supported.Using uv (Recommended)Bash# Clone the repository
git clone <your-repo-url>
cd vector-db

# Install dependencies (NumPy, Matplotlib, Streamlit, Sentence-Transformers)
uv sync
Using standard pipBashpython -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install numpy matplotlib streamlit sentence-transformers
Verification & Test SuitesRun the standalone verification suites to validate correctness, mathematical equivalence, and deletion lifecycles:Bash# 1. Verify exact index correctness and memory compaction
uv run python tests/test_exact.py

# 2. Verify IVF-Flat basic cluster assignments and monotonic recall
uv run python tests/test_ivf.py

# 3. Verify advanced IVF (K-Means++, SQ8 3.8x compression, >99.5% fidelity, adaptive routing)
uv run python tests/test_ivf_advanced.py
Benchmarks & Evaluation1. Synthetic Clustered Dataset ($N=50,000$, $d=128$)Evaluates exact ground truth versus IVF-Flat across $nprobe \in \{1, 2, 4, 8, 16, 32\}$ using 500 independent queries:Bashuv run python src/benchmark.py --dataset synthetic
Outputs: benchmark_synthetic.csv and pareto_synthetic.png showing the non-dominated Pareto frontier trading off QPS against Recall@10.2. Real-World Semantic Dataset ($N=5,000$, $d=384$)Generates semantic text embeddings for 5,000 multi-domain procedural sentences using sentence-transformers/all-MiniLM-L6-v2, then runs the evaluation harness:Bash# Generate the embeddings (requires initial network access to download weights)
uv run python src/embed_text.py

# Run benchmark and produce Pareto curve on real-world semantic geometry
uv run python src/benchmark.py --dataset real
Outputs: text_embeddings.npz, benchmark_real.csv, and pareto_real.png.Running the Interactive Streamlit DashboardLaunch the visualizer to explore cluster geometries, adjust search probes, inspect latency speedups, and perform live semantic searches:Bashuv run streamlit run app.py
Dashboard FeaturesZero-Dependency 2D Manifold Projection: Uses pure-NumPy sample covariance eigendecomposition (PCA) to render high-dimensional vectors and centroids in 2D.Cluster Probe Visualizer: Highlights probed Voronoi cells in real time as the nprobe slider moves.Live Latency & Speedup Metrics: Compares Exact vs. IVF-Flat p50/p99 search times and computes real-time Recall@K.Natural Language Semantic Search: Accepts arbitrary user text queries and surfaces the closest semantic matches from the indexed corpus.Interactive Tombstone Deletions: Deletes vectors on-the-fly and confirms immediate exclusion from search results.Algorithmic & Implementation DetailsDistance MetricAll ingested vectors and search queries are row-normalized ($L_2$ norm clamped to $\epsilon = 10^{-12}$) upon entry:$$\hat{x} = \frac{x}{\max(\|x\|_2, \epsilon)}$$Cosine similarity simplifies directly to an inner product:$$\cos(q, x) = \hat{q} \cdot \hat{x}^T$$This enables Level-3 BLAS GEMM (cblas_sgemm) to score large batches of queries with maximum cache locality and SIMD pipeline saturation.Top-$K$ SelectionInstead of costly $O(N \log N)$ sorting over the entire candidate pool, the search path uses:np.argpartition(scores, -k) to isolate the top-$k$ unsorted candidate partition in $O(N)$ time.np.argsort applied only locally to the $k$ partitioned elements ($O(k \log k)$).Dynamic Adaptive nprobeQueries evaluate a softmax probability distribution over centroid similarities:$$P(c_i \mid q) = \frac{e^{(q \cdot c_i) / \tau}}{\sum_j e^{(q \cdot c_j) / \tau}}$$Clusters are probed in descending order until cumulative probability exceeds a target threshold ($\sum P \ge 0.90$). Confident, cluster-centered queries probe only 1–2 clusters, whereas boundary queries dynamically expand search depth.

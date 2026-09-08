"""
src/embed_text.py — Memory-safe text embedding generator.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

# Force PyTorch/ONNX to avoid allocating excess thread pools or huge commit buffers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_corpus import generate_text_corpus

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
N_TEXTS = 5000
SEED = 42
OUT_PATH = DATA_DIR / "text_embeddings.npz"


def main():
    try:
        import torch
        # Cap CPU threads so PyTorch doesn't allocate per-thread work buffers
        torch.set_num_threads(2)
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    print("Generating procedural text corpus...")
    texts, labels = generate_text_corpus(N_TEXTS, seed=SEED)

    print(f"Loading {MODEL_NAME} on CPU without memory mapping...")
    # Low-memory instantiation: pass torch_dtype float32, cpu device, no heavy mmap
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"low_cpu_mem_usage": True},
    )

    print(f"Encoding {N_TEXTS} texts in small batches (32)...")
    # Reduced batch_size prevents RAM/pagefile spikes
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    assert embeddings.shape == (N_TEXTS, 384), embeddings.shape

    np.savez(
        OUT_PATH,
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    print(f"Saved {OUT_PATH}: {embeddings.shape[0]} vectors, d={embeddings.shape[1]}")


if __name__ == "__main__":
    main()
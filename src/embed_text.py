"""
embed_text.py — embed the procedural text corpus with
sentence-transformers/all-MiniLM-L6-v2 (d=384) and save to disk.

This script needs network access to download the model on first run.
Run it in your own environment, then copy text_embeddings.npz alongside
the other files for benchmark.py / app.py to use.

    pip install sentence-transformers
    python embed_text.py
"""
from __future__ import annotations

import sys
import numpy as np

from text_corpus import generate_text_corpus

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
N_TEXTS = 5000
SEED = 42
OUT_PATH = "text_embeddings.npz"


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "sentence-transformers is not installed.\n"
            "Run:  pip install sentence-transformers\n"
            "Then re-run this script in an environment with internet access.",
            file=sys.stderr,
        )
        sys.exit(1)

    texts, labels = generate_text_corpus(N_TEXTS, seed=SEED)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    assert embeddings.shape == (N_TEXTS, 384), embeddings.shape

    np.savez(
        OUT_PATH,
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    print(f"saved {OUT_PATH}: {embeddings.shape[0]} vectors, d={embeddings.shape[1]}")


if __name__ == "__main__":
    main()

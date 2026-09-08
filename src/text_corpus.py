"""
src/text_corpus.py — Procedural short-text corpus generator.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np

from pathlib import Path
import sys
SRC_DIR = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(SRC_DIR))

from corpus_vocab import DOMAINS


def _domain_sentences(vocab: dict, count: int, rng: np.random.Generator) -> List[str]:
    subs, verbs, objs = vocab["subject"], vocab["verb"], vocab["object"]
    combos = [
        (i, j, l)
        for i in range(len(subs))
        for j in range(len(verbs))
        for l in range(len(objs))
    ]
    if count > len(combos):
        raise ValueError("Not enough unique combinations for requested count")
    chosen = rng.permutation(len(combos))[:count]
    return [
        f"{subs[combos[c][0]]} {verbs[combos[c][1]]} {objs[combos[c][2]]}."
        for c in chosen
    ]


def generate_text_corpus(n: int = 5000, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    domain_names = list(DOMAINS.keys())
    n_domains = len(domain_names)
    base, extra = divmod(n, n_domains)

    texts: List[str] = []
    labels: List[str] = []
    for i, name in enumerate(domain_names):
        count = base + (1 if i < extra else 0)
        sentences = _domain_sentences(DOMAINS[name], count, rng)
        texts.extend(sentences)
        labels.extend([name] * count)

    perm = rng.permutation(len(texts))
    return [texts[i] for i in perm], [labels[i] for i in perm]


if __name__ == "__main__":
    texts, labels = generate_text_corpus(5000, seed=42)
    print(f"Generated {len(texts)} texts across {len(set(labels))} domains")
    for t in texts[:5]:
        print(" -", t)
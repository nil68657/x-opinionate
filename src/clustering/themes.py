"""Cluster a list of complaint-like tweets and ask Claude to name each cluster.

Embeddings come from sentence-transformers when available (better quality) and
fall back to TF-IDF (no GPU, no extra deps).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ..llm_client import call_json


def _embed(texts: Sequence[str]) -> np.ndarray:
    """Sentence-transformers if installed, otherwise TF-IDF dense vectors."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return np.asarray(model.encode(list(texts), show_progress_bar=False))
    except Exception:
        vec = TfidfVectorizer(
            max_features=2048, stop_words="english", ngram_range=(1, 2)
        )
        return vec.fit_transform(texts).toarray()


def cluster(texts: list[str], k: int | None = None) -> list[int]:
    """Return a cluster label per text. ``k`` is auto-picked if not given."""
    if not texts:
        return []
    if k is None:
        k = max(2, min(6, max(1, len(texts) // 4)))
    k = min(k, len(texts))
    X = _embed(texts)
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    return km.fit_predict(X).tolist()


THEME_SYSTEM = """You are a customer-experience analyst. Given a cluster of
customer complaints or comments, write:
  - a concise 3-6 word THEME label (specific, not generic), and
  - a one-sentence summary of the underlying issue or topic.
Avoid filler words like "issues" or "problems" on their own."""

THEME_USER = """Here are {n} representative tweets from one cluster:
{joined}

Return JSON:
{{
  "theme": "...",
  "summary": "..."
}}"""


def label_clusters(
    texts: list[str],
    labels: list[int],
    max_per_cluster: int = 8,
) -> dict[int, dict]:
    """For each cluster id, return ``{theme, summary, size, examples}``."""
    out: dict[int, dict] = {}
    for cid in sorted(set(labels)):
        members = [t for t, lbl in zip(texts, labels) if lbl == cid][:max_per_cluster]
        joined = "\n".join(f"- {t}" for t in members)
        try:
            result = call_json(
                THEME_SYSTEM,
                THEME_USER.format(n=len(members), joined=joined),
                max_tokens=300,
            )
        except Exception as e:
            result = {"theme": f"Cluster {cid}", "summary": f"(LLM error: {e})"}
        result["size"] = sum(1 for lbl in labels if lbl == cid)
        result["examples"] = members[:3]
        out[cid] = result
    return out

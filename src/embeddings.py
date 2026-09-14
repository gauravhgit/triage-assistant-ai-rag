"""
Pluggable embedding backends for the RAG pipeline.

Design note (see README "Design Decisions"):
The default backend is TF-IDF (scikit-learn), which runs fully offline with
no API key and no model download - important for a portfolio project anyone
can clone and run immediately. It's also a legitimate retrieval baseline for
QE corpora, where exact terms (exception class names, error codes, function
names) carry a lot of signal that dense embedding models can sometimes
under-weight.

The `Embedder` interface is intentionally small so a production deployment
can swap in a real embedding model (OpenAI text-embedding-3, Voyage AI,
sentence-transformers, etc.) without touching retrieval or triage logic.
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(abc.ABC):
    """Common interface every embedding backend must implement."""

    @abc.abstractmethod
    def fit(self, texts: List[str]) -> None:
        """Fit the embedder on the full corpus (call once at index-build time)."""

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) float array of embeddings."""


class TfidfEmbedder(Embedder):
    """
    Default offline embedding backend.

    Uses word 1-2gram TF-IDF, which works well for short technical text
    (stack traces, error messages, ticket titles) where exact vocabulary
    overlap is highly informative.
    """

    def __init__(self, max_features: int = 4000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed().")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


def get_default_embedder() -> Embedder:
    """
    Factory so the rest of the codebase never hardcodes a specific backend.

    Extend this to check for an API key (e.g. OPENAI_API_KEY, VOYAGE_API_KEY)
    and return a real embedding-model-backed Embedder in production. See
    README.md -> "Swapping in a production embedding model".
    """
    return TfidfEmbedder()

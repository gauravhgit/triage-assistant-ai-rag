"""
Hybrid retrieval: vector similarity + exact-match keyword boosting.

Why hybrid: pure semantic similarity can under-rank a document that shares
an exact, highly diagnostic token with the query (an exception class name
like `StaleElementReferenceException`, an error code like `429`, a SKU, a
commit sha) if the surrounding language differs. QE artifacts are full of
exactly these high-precision tokens, so we boost documents that share them
with the query on top of the base vector score.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.embeddings import Embedder
from src.vector_store import VectorStore

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"

# Tokens worth an exact-match boost: CamelCase/PascalCase identifiers,
# ALL_CAPS error codes, HTTP status-like 3-digit codes, hex-ish commit shas.
_TOKEN_PATTERN = re.compile(
    r"[A-Z][a-zA-Z]*(?:Error|Exception|Timeout)\b|\b[45]\d{2}\b|\b[0-9a-f]{7}\b"
)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, keyword_boost: float = 0.15):
        self.store = store
        self.embedder = embedder
        self.keyword_boost = keyword_boost

    @classmethod
    def from_disk(cls, index_dir: str = str(INDEX_DIR)) -> "Retriever":
        store = VectorStore.load(index_dir)
        with open(Path(index_dir) / "embedder.pkl", "rb") as f:
            embedder = pickle.load(f)
        return cls(store, embedder)

    def retrieve(self, query: str, top_k: int = 5,
                 source_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed([query])[0]
        # Over-fetch, then re-rank with the keyword boost applied.
        candidates = self.store.search(query_embedding, top_k=top_k * 4,
                                        source_types=source_types)

        query_tokens = set(_TOKEN_PATTERN.findall(query))
        for c in candidates:
            doc_tokens = set(_TOKEN_PATTERN.findall(c["text"]))
            overlap = query_tokens & doc_tokens
            c["matched_keywords"] = sorted(overlap)
            c["score"] = c["score"] + self.keyword_boost * len(overlap)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_k]

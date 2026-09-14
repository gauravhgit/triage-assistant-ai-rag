"""
A minimal, dependency-light vector store.

For a corpus this size (a few dozen to a few thousand documents, typical of
a single team's test-failure knowledge base), brute-force cosine similarity
over an in-memory numpy matrix is fast (<10ms) and far easier to audit than
a full vector database. The interface mirrors what you'd get from
Chroma/FAISS/pgvector, so swapping in a real vector DB at scale is a drop-in
replacement documented in README.md -> "Scaling beyond a single team".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str
    source_type: str  # "historical_failure" | "defect_ticket" | "runbook" | "commit"
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add(self, documents: List[Document], embeddings: np.ndarray) -> None:
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               source_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        candidate_idx = list(range(len(self.documents)))
        if source_types:
            candidate_idx = [i for i in candidate_idx
                              if self.documents[i].source_type in source_types]
        if not candidate_idx:
            return []

        candidate_matrix = self.embeddings[candidate_idx]
        sims = _cosine_similarity(query_embedding, candidate_matrix)

        ranked = sorted(zip(candidate_idx, sims), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            doc = self.documents[idx]
            results.append({
                "doc_id": doc.doc_id,
                "text": doc.text,
                "source_type": doc.source_type,
                "metadata": doc.metadata,
                "score": float(score),
            })
        return results

    def save(self, path: str) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.embeddings)
        docs_serializable = [
            {"doc_id": d.doc_id, "text": d.text, "source_type": d.source_type,
             "metadata": d.metadata}
            for d in self.documents
        ]
        with open(out_dir / "documents.json", "w") as f:
            json.dump(docs_serializable, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        in_dir = Path(path)
        store = cls()
        store.embeddings = np.load(in_dir / "embeddings.npy")
        with open(in_dir / "documents.json") as f:
            docs_raw = json.load(f)
        store.documents = [
            Document(doc_id=d["doc_id"], text=d["text"],
                     source_type=d["source_type"], metadata=d["metadata"])
            for d in docs_raw
        ]
        return store


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_vec = query_vec.reshape(1, -1)
    q_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
    m_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1e-8
    m_norm[m_norm == 0] = 1e-8
    sims = (matrix @ query_vec.T) / (m_norm * q_norm)
    return sims.flatten()

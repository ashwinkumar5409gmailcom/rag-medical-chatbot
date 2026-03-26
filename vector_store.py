"""
Stores embeddings in FAISS and performs similarity search.
"""

from __future__ import annotations

from typing import Dict, List

import faiss
import numpy as np


class FAISSVectorStore:
    """
    Simple in-memory FAISS store.
    IndexFlatIP uses inner product, which works well with normalized vectors.
    """

    def __init__(self, embedding_dimension: int) -> None:
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.documents: List[Dict[str, str]] = []

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, str]],
    ) -> None:
        """
        Adds document embeddings and keeps the original document text in memory.
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings and documents must match.")

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, object]]:
        """
        Returns the top_k most similar chunks with similarity scores.
        """
        scores, indices = self.index.search(query_embedding, top_k)
        results: List[Dict[str, object]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                {
                    "score": float(score),
                    "document": self.documents[idx],
                }
            )

        return results

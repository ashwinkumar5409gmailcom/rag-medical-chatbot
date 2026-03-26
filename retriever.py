"""
Retrieves relevant medical chunks for a user symptom query.
"""

from __future__ import annotations

from typing import Dict, List

from embedder import MedicalEmbedder
from vector_store import FAISSVectorStore


class MedicalRetriever:
    """
    Connects the embedder and vector store to retrieve context.
    """

    def __init__(self, embedder: MedicalEmbedder, vector_store: FAISSVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        """
        Converts the query into an embedding and searches in FAISS.
        """
        query_embedding = self.embedder.encode_query(query)
        return self.vector_store.search(query_embedding, top_k=top_k)

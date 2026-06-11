"""Plain RAG: hybrid dense + BM25 retrieval, top-k straight to the LLM."""
from __future__ import annotations

from .base import Approach, RetrievedChunk, TraceStep
from ..config import SETTINGS
from ..ollama_client import embed_one


class PlainRAG(Approach):
    name = "plain"

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        qvec = embed_one(query, role="query")
        hits = self.index.hybrid_search(query, qvec, k=SETTINGS.top_k, bm25_weight=SETTINGS.bm25_weight)
        trace.append(
            TraceStep("Hybrid retrieval", f"dense(embeddinggemma)+BM25, top {SETTINGS.top_k}")
        )
        return [RetrievedChunk(self.index.get(i), s, "hybrid") for i, s in hits]

"""RAG + Reranker: retrieve a wide candidate pool, then re-order with a Jina
cross-encoder and keep the top-k."""
from __future__ import annotations

from .base import Approach, RetrievedChunk, TraceStep
from ..config import SETTINGS
from ..ollama_client import embed_one
from ..reranker import rerank, backend_name


class RerankRAG(Approach):
    name = "rerank"

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        qvec = embed_one(query, role="query")
        cand = self.index.hybrid_search(
            query, qvec, k=SETTINGS.candidate_k, bm25_weight=SETTINGS.bm25_weight
        )
        chunks = [self.index.get(i) for i, _ in cand]
        trace.append(TraceStep("Candidate pool", f"hybrid retrieval, {len(chunks)} candidates"))

        scores = rerank(query, [c.text for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        trace.append(TraceStep("Cross-encoder rerank", f"model: {backend_name()}"))
        return [RetrievedChunk(c, s, "rerank") for c, s in ranked[: SETTINGS.top_k]]

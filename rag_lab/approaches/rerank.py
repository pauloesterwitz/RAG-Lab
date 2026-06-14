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
        hyb = [s for _, s in cand]
        trace.append(TraceStep("Candidate pool", f"hybrid retrieval, {len(chunks)} candidates"))

        ce = rerank(query, [c.text for c in chunks])

        # Blend cross-encoder + hybrid (both min-max normalized): the cross-encoder
        # leads, but the proven hybrid signal guards against a noisy CE score
        # evicting a strong hybrid hit (e.g. the gold chunk) from the top-k.
        def _mm(xs: list[float]) -> list[float]:
            lo, hi = min(xs), max(xs)
            return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]

        ce_n, hyb_n = _mm(ce), _mm(hyb)
        blended = [(c, 0.7 * ce_n[j] + 0.3 * hyb_n[j]) for j, c in enumerate(chunks)]
        ranked = sorted(blended, key=lambda x: x[1], reverse=True)
        trace.append(TraceStep("Cross-encoder rerank (blended w/ hybrid)", f"model: {backend_name()}"))
        return [RetrievedChunk(c, s, "rerank") for c, s in ranked[: SETTINGS.top_k]]

"""HyDE (Hypothetical Document Embeddings): ask the LLM to draft a plausible
answer, embed that hypothetical passage, and retrieve documents near it. BM25
still runs on the literal query, so we get the best of both."""
from __future__ import annotations

from .base import Approach, RetrievedChunk, TraceStep
from ..config import SETTINGS
from ..ollama_client import embed_one, generate


HYDE_SYSTEM = (
    "You are helping a search system. Write a concise, factual passage (4-6 sentences) "
    "that would plausibly answer the question as if it were an excerpt from a relevant "
    "document. Do not say you are unsure; just write the passage."
)


class HydeRAG(Approach):
    name = "hyde"

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        hypothetical = generate(
            f"Question: {query}\n\nWrite the hypothetical answer passage:",
            system=HYDE_SYSTEM,
            num_predict=256,
            temperature=0.3,
        ).strip()
        trace.append(TraceStep("Hypothetical document", hypothetical[:400]))

        hyde_vec = embed_one(hypothetical, role="document")
        hits = self.index.hybrid_search(
            query, hyde_vec, k=SETTINGS.top_k, bm25_weight=SETTINGS.bm25_weight
        )
        trace.append(TraceStep("Retrieve on HyDE vector", f"top {SETTINGS.top_k} (BM25 on literal query)"))
        return [RetrievedChunk(self.index.get(i), s, "hyde") for i, s in hits]

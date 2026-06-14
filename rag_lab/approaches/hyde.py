"""HyDE (Hypothetical Document Embeddings): ask the LLM to draft a plausible
answer, embed that hypothetical passage, and retrieve documents near it. BM25
still runs on the literal query, so we get the best of both."""
from __future__ import annotations

import numpy as np

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

        # Blend the hypothetical-document vector WITH the literal query vector so a
        # generic/hallucinated draft can't drift retrieval off the real query.
        hyde_vec = np.asarray(embed_one(hypothetical, role="document"), dtype=np.float32)
        qvec = np.asarray(embed_one(query, role="query"), dtype=np.float32)
        blended = hyde_vec + qvec  # hybrid_search re-normalizes the query vector
        hits = self.index.hybrid_search(
            query, blended, k=SETTINGS.top_k, bm25_weight=SETTINGS.bm25_weight
        )
        trace.append(TraceStep("Retrieve on blended HyDE+query vector", f"top {SETTINGS.top_k} (BM25 on literal query)"))
        return [RetrievedChunk(self.index.get(i), s, "hyde") for i, s in hits]

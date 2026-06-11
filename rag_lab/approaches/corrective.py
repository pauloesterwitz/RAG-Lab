"""Corrective RAG (CRAG): retrieve, grade each doc for relevance, and take a
corrective action. We lack web search, so the corrective action is query
rewriting + decomposition followed by re-retrieval, then knowledge filtering."""
from __future__ import annotations

import json

from .base import Approach, RetrievedChunk, TraceStep
from ..config import SETTINGS
from ..ollama_client import embed_one, generate

_GRADE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score"],
}
_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}


class CorrectiveRAG(Approach):
    name = "corrective"
    relevance_threshold = 0.5
    min_good = 3

    def _grade(self, query: str, text: str) -> float:
        prompt = (
            f"Question: {query}\n\nRetrieved passage:\n{text[:1500]}\n\n"
            "Grade how well this passage helps answer the question, from 0.0 (irrelevant) "
            'to 1.0 (directly answers). Reply JSON: {"score": <0-1>, "reason": "<short>"}.'
        )
        try:
            raw = generate(prompt, fmt=_GRADE_SCHEMA, num_predict=120, temperature=0.0)
            return max(0.0, min(1.0, float(json.loads(raw).get("score", 0.0))))
        except Exception:
            return 0.0

    def _rewrite(self, query: str) -> list[str]:
        prompt = (
            f"The search results for this question were weak: \"{query}\".\n"
            "Rewrite it into 3 better, more specific search queries (decompose if needed). "
            'Reply JSON: {"queries": ["...", "...", "..."]}.'
        )
        try:
            raw = generate(prompt, fmt=_REWRITE_SCHEMA, num_predict=200, temperature=0.3)
            qs = json.loads(raw).get("queries", [])
            return [q for q in qs if isinstance(q, str) and q.strip()][:3]
        except Exception:
            return []

    def _search(self, query: str, k: int):
        qvec = embed_one(query, role="query")
        return self.index.hybrid_search(query, qvec, k=k, bm25_weight=SETTINGS.bm25_weight)

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        hits = self._search(query, SETTINGS.candidate_k)
        graded: dict[int, float] = {}
        for i, _ in hits[:8]:  # grade the head of the pool
            graded[i] = self._grade(query, self.index.get(i).text)
        good = {i: s for i, s in graded.items() if s >= self.relevance_threshold}
        trace.append(
            TraceStep(
                "Grade documents",
                f"{len(good)}/{len(graded)} passed (≥{self.relevance_threshold}); "
                f"max score {max(graded.values()) if graded else 0:.2f}",
            )
        )

        if len(good) < self.min_good:
            rewrites = self._rewrite(query)
            trace.append(TraceStep("Corrective action", "weak results → rewrite + re-retrieve: " + " | ".join(rewrites)))
            for rq in rewrites:
                for i, _ in self._search(rq, 6):
                    if i not in graded:
                        graded[i] = self._grade(query, self.index.get(i).text)
            good = {i: s for i, s in graded.items() if s >= self.relevance_threshold}
            if not good:  # accept best-effort
                good = dict(sorted(graded.items(), key=lambda x: x[1], reverse=True)[: self.min_good])

        ranked = sorted(good.items(), key=lambda x: x[1], reverse=True)[: SETTINGS.top_k]
        return [RetrievedChunk(self.index.get(i), s, "graded") for i, s in ranked]

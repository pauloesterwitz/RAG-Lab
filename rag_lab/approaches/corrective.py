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
    "properties": {"scores": {"type": "array", "items": {"type": "number"}}},
    "required": ["scores"],
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

    def _grade_batch(self, query: str, texts: list[str]) -> list[float]:
        """Grade ALL passages in ONE LLM call (was one call per passage → 26+/query
        and ~339 s latency). Returns a 0-1 score per passage, in order."""
        if not texts:
            return []
        blocks = "\n\n".join(f"[{j}] {t[:900]}" for j, t in enumerate(texts))
        prompt = (
            f"Question: {query}\n\nGrade how well EACH passage helps answer the question, "
            f"from 0.0 (irrelevant) to 1.0 (directly answers).\n\n{blocks}\n\n"
            f'Reply JSON: {{"scores": [<one number per passage, {len(texts)} total, in order>]}}.'
        )
        try:
            raw = generate(prompt, fmt=_GRADE_SCHEMA, num_predict=256, temperature=0.0)
            sc = [max(0.0, min(1.0, float(s))) for s in json.loads(raw).get("scores", [])]
        except Exception:
            sc = []
        sc = sc[: len(texts)] + [0.0] * (len(texts) - len(sc))
        return sc

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
        hscore: dict[int, float] = {i: s for i, s in hits}

        head = [i for i, _ in hits[:8]]
        scores = self._grade_batch(query, [self.index.get(i).text for i in head])
        graded: dict[int, float] = dict(zip(head, scores))
        good = {i for i, s in graded.items() if s >= self.relevance_threshold}
        trace.append(TraceStep(
            "Grade documents (batched)",
            f"{len(good)}/{len(graded)} passed (≥{self.relevance_threshold}); "
            f"max {max(graded.values()) if graded else 0:.2f}",
        ))

        if len(good) < self.min_good:
            rewrites = self._rewrite(query)
            trace.append(TraceStep("Corrective action", "weak results → rewrite + re-retrieve: " + " | ".join(rewrites)))
            new: list[int] = []
            for rq in rewrites:
                for i, s in self._search(rq, 5):
                    hscore.setdefault(i, s)
                    if i not in graded and i not in new:
                        new.append(i)
            for i, s in zip(new, self._grade_batch(query, [self.index.get(i).text for i in new])):
                graded[i] = s
            good = {i for i, s in graded.items() if s >= self.relevance_threshold}

        # Rank by a BLEND of grade + retrieval score over graded ∪ hybrid hits, so a
        # confident retrieval (e.g. the gold chunk) is never dropped by a noisy grade,
        # and we always return top_k.
        def blended(i: int) -> float:
            return 0.6 * graded.get(i, 0.0) + 0.4 * hscore.get(i, 0.0)

        pool = set(graded) | set(hscore)
        ranked = sorted(pool, key=blended, reverse=True)[: SETTINGS.top_k]
        trace.append(TraceStep("Rank by grade+retrieval blend", f"{len(good)} passed grade; returning top {len(ranked)}"))
        return [RetrievedChunk(self.index.get(i), blended(i), "graded") for i in ranked]

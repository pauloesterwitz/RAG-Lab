"""Agentic RAG: an LLM agent decomposes the question into sub-queries, issues
them against the retriever (its single 'search' tool), reflects on whether the
gathered evidence is sufficient, and issues follow-ups until satisfied or a
budget is hit."""
from __future__ import annotations

import json

from .base import Approach, RetrievedChunk, TraceStep
from ..config import SETTINGS
from ..ollama_client import embed_one, generate

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {"subqueries": {"type": "array", "items": {"type": "string"}}},
    "required": ["subqueries"],
}
_REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "followups": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sufficient"],
}


class AgenticRAG(Approach):
    name = "agentic"
    max_iters = 3
    per_query_k = 4

    def _search(self, query: str, k: int):
        qvec = embed_one(query, role="query")
        return self.index.hybrid_search(query, qvec, k=k, bm25_weight=SETTINGS.bm25_weight)

    def _plan(self, query: str) -> list[str]:
        prompt = (
            f"You are a research agent. Break this question into 2-3 focused search "
            f'sub-queries.\nQuestion: {query}\nReply JSON: {{"subqueries": ["...", "..."]}}'
        )
        try:
            raw = generate(prompt, fmt=_PLAN_SCHEMA, num_predict=200, temperature=0.2)
            subs = [q for q in json.loads(raw).get("subqueries", []) if isinstance(q, str) and q.strip()]
            return subs[:3] or [query]
        except Exception:
            return [query]

    def _reflect(self, query: str, gathered: list[str]) -> tuple[bool, list[str]]:
        joined = "\n---\n".join(t[:500] for t in gathered[:8])
        prompt = (
            f"Question: {query}\n\nEvidence gathered so far:\n{joined}\n\n"
            "Is this evidence sufficient to answer the question fully? If not, list up to 2 "
            'follow-up search queries that would fill the gaps. Reply JSON: '
            '{"sufficient": <bool>, "followups": ["..."]}'
        )
        try:
            raw = generate(prompt, fmt=_REFLECT_SCHEMA, num_predict=200, temperature=0.1)
            d = json.loads(raw)
            return bool(d.get("sufficient", True)), [
                q for q in d.get("followups", []) if isinstance(q, str) and q.strip()
            ][:2]
        except Exception:
            return True, []

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        subqueries = self._plan(query)
        trace.append(TraceStep("Plan sub-queries", " | ".join(subqueries)))

        collected: dict[int, float] = {}
        pending = list(subqueries)
        for it in range(self.max_iters):
            if not pending:
                break
            for sq in pending:
                for i, s in self._search(sq, self.per_query_k):
                    collected[i] = max(collected.get(i, 0.0), s)
            gathered = [self.index.get(i).text for i in collected]
            sufficient, followups = self._reflect(query, gathered)
            trace.append(
                TraceStep(
                    f"Iteration {it+1}",
                    f"searched {len(pending)} queries → {len(collected)} chunks; "
                    + ("sufficient" if sufficient else "need more: " + " | ".join(followups)),
                )
            )
            if sufficient or not followups:
                break
            pending = followups

        ranked = sorted(collected.items(), key=lambda x: x[1], reverse=True)[: SETTINGS.top_k]
        return [RetrievedChunk(self.index.get(i), s, "agentic") for i, s in ranked]

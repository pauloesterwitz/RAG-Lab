"""PageIndex: reasoning-based retrieval. An LLM navigates each document's prebuilt
hierarchical tree index (root -> section -> subsection, ToC/heading-derived) by
reading node titles and summaries, the way a reader flips to the right section of
a report. Structure decides which sections to read; a fused BM25 + dense score
picks the best chunks within them and orders the final merge."""
from __future__ import annotations

import json

import numpy as np

from .base import Approach, RetrievedChunk, TraceStep
from ..config import SETTINGS
from ..llamaswap_client import embed_one, generate
from ..pageindex_build import PITree, load_doc_summaries, load_tree, pageindex_exists

_ROOT_SCHEMA = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc": {"type": "string"},
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["doc", "relevant"],
            },
        }
    },
    "required": ["documents"],
}

_DESCEND_SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {"type": "array", "items": {"type": "string"}},
        "stop_here": {"type": "boolean"},
    },
    "required": ["selected", "stop_here"],
}


class PageIndexRAG(Approach):
    name = "pageindex"

    def __init__(self, index):
        super().__init__(index)
        self._chunk_by_id = {c.id: c for c in index.chunks}
        self._pos_by_id = {c.id: i for i, c in enumerate(index.chunks)}
        self._trees: dict[str, PITree] = {}
        self._doc_summaries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not pageindex_exists():
            return
        self._doc_summaries = load_doc_summaries()
        for doc in self._doc_summaries:
            tree = load_tree(doc)
            if tree is not None:
                self._trees[doc] = tree

    # --- root-level document fan-out (also the multi-hop mechanism: this one
    # call can mark >1 doc relevant, each navigated independently below) -----
    def _select_documents(self, query: str, trace: list[TraceStep]) -> list[str]:
        listing = "\n".join(
            f'- "{doc}": {info.get("title") or doc} — {info.get("summary", "")[:300]}'
            for doc, info in self._doc_summaries.items()
        )
        prompt = (
            "You are choosing which documents to search to answer a question, based only on "
            "each document's title and summary (like scanning a library shelf). Some questions "
            "combine information from TWO OR MORE documents — mark EVERY document that could "
            "plausibly contribute PART of the answer, not just the single closest match. When "
            "genuinely unsure, mark a document relevant rather than excluding it.\n\n"
            f"Documents:\n{listing}\n\nQuestion: {query}\n\n"
            'Reply JSON: {"documents": [{"doc": "<exact name>", "relevant": true|false, "reason": ".."}]}'
        )
        picked: list[str] = []
        try:
            # Scales with doc count: some local models write a verbose "reason" per
            # entry, and a too-tight budget truncates mid-string -> invalid JSON
            # (observed with qwen38fn: num_predict=400 truncated on 11 docs; 1500 didn't).
            budget = max(1500, 150 * len(self._doc_summaries) + 400)
            data = json.loads(generate(
                prompt, model=SETTINGS.pageindex_model, fmt=_ROOT_SCHEMA, num_predict=budget, temperature=0.0,
            ))
            picked = [d["doc"] for d in data.get("documents", [])
                      if d.get("relevant") and d.get("doc") in self._doc_summaries]
            trace.append(TraceStep(
                "Root selection",
                "; ".join(f'{d["doc"]}: {d.get("reason", "")}' for d in data.get("documents", [])
                          if d.get("relevant")) or "(none relevant)",
            ))
        except Exception:
            trace.append(TraceStep("Root selection", "parse failed"))
        # REVERTED 2026-09-14: an embedding-based document-summary fallback used to
        # live here for the "found nothing relevant" case. A/B'd against the exact
        # same 100 goldens: it made things WORSE (composite 0.659->0.627, gold-chunk
        # hit 50%->45%), because it's strictly weaker than the retrieve()-level
        # _hybrid_fallback below — no BM25/keyword signal, only a coarse per-document
        # summary embedding instead of per-chunk, and no recovery once it commits to
        # the wrong 1-2 documents. Traced multiple regressed cases directly to this:
        # e.g. "Which background service runs while Eppie CLI hangs..." — gold doc
        # is "Agents of Chaos.pdf", full hybrid search found it via keyword match,
        # this fallback instead picked HyperAgents + Chollet's book by theme and
        # never recovered. So: an empty `picked` now falls straight through to
        # retrieve()'s _hybrid_fallback (full corpus, hybrid search), same as before
        # this approach existed — that IS the better fallback for this failure mode.
        return picked[: SETTINGS.pageindex_max_docs]

    # --- per-hop tree descent -------------------------------------------------
    def _choose_children(self, query: str, tree: PITree, node, trace: list[TraceStep], doc: str,
                         scores: np.ndarray) -> list[str]:
        options = "\n".join(
            f'- id="{cid}" "{tree.nodes[cid].title}" (p.{tree.nodes[cid].page_start}-{tree.nodes[cid].page_end}): '
            f'{tree.nodes[cid].summary[:200]}'
            for cid in node.children
        )
        prompt = (
            f'You are navigating the structure of "{doc}" to answer a question, the way a reader '
            "flips to the right section using a table of contents.\n\n"
            f'Current section: "{node.title}" — {node.summary[:200]}\n\nSubsections:\n{options}\n\n'
            f"Question: {query}\n\n"
            "For EACH subsection, judge whether it's worth descending into to answer the question, "
            "or say this section itself already answers it (stop_here=true, selected=[]). Reply JSON: "
            '{"selected": ["<id>", ...], "stop_here": true|false}'
        )
        try:
            # Same truncation risk as root selection, smaller (no "reason" field here) but
            # still scales with child count for the same reason.
            budget = max(600, 100 * len(node.children) + 200)
            data = json.loads(generate(
                prompt, model=SETTINGS.pageindex_model, fmt=_DESCEND_SCHEMA, num_predict=budget, temperature=0.0,
            ))
            picked = [cid for cid in data.get("selected", []) if cid in tree.nodes]
            stop_here = bool(data.get("stop_here", False)) or not picked
        except Exception:
            trace.append(TraceStep(f"[{doc}] Descend failed: {node.title}", "call or parse failed"))
            return []
        if stop_here:
            trace.append(TraceStep(f"[{doc}] Stopped at: {node.title}", f"p.{node.page_start}-{node.page_end}"))
            return []
        # The model lists its picks in document order; keep the best-scoring ones, not the first ones.
        picked.sort(key=lambda cid: self._node_best(tree.nodes[cid], scores), reverse=True)
        selected = picked[: SETTINGS.pageindex_max_breadth]
        trace.append(TraceStep(f"[{doc}] Descend from: {node.title}",
                                f"model picked {len(picked)}, kept: " + ", ".join(tree.nodes[c].title for c in selected)))
        return selected

    def _node_best(self, node, scores: np.ndarray) -> float:
        return max((float(scores[self._pos_by_id[c]]) for c in node.chunk_ids if c in self._pos_by_id),
                   default=float("-inf"))

    def _navigate(self, query: str, tree: PITree, node, trace: list[TraceStep], doc: str,
                  scores: np.ndarray, depth: int = 0) -> list[str]:
        if not node.children or depth >= SETTINGS.pageindex_max_depth:
            trace.append(TraceStep(f"[{doc}] Leaf: {node.title}", f"p.{node.page_start}-{node.page_end}"))
            return self._top_chunks(node.chunk_ids, scores)
        selected = self._choose_children(query, tree, node, trace, doc, scores)
        if not selected:
            return self._top_chunks(node.chunk_ids, scores)
        gathered: list[str] = []
        for cid in selected:
            gathered.extend(self._navigate(query, tree, tree.nodes[cid], trace, doc, scores, depth + 1))
        return gathered

    def _top_chunks(self, chunk_ids: list[str], scores: np.ndarray) -> list[str]:
        """Cap ONE navigated location's contribution to the candidate pool by its
        own top-k score, before it's merged with other locations. A broad stop/leaf
        node (a whole section) would otherwise dump every one of its chunks into the
        pool, diluting the final top-k with off-topic siblings. Structure still
        decides *which locations* to visit; this only trims *within* a location."""
        scored = [(cid, float(scores[self._pos_by_id[cid]])) for cid in chunk_ids if cid in self._pos_by_id]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[: SETTINGS.top_k]]

    def _merge_with_doc_floor(self, chunk_ids: list[str], scores: np.ndarray) -> list[tuple[str, float]]:
        """Final cross-location merge, with a per-document floor. A flat global
        top_k cut over every navigated document's candidates let one document's
        higher raw scores fully starve another: of multi-hop cases where both gold
        documents were correctly navigated, over half still had one document's
        chunks zeroed out of the final top_k. Reserve floor = max(1, top_k // n_docs)
        slots per document (best chunks first, by the same score), then fill any
        remaining slots by global rank over what's left. Structure still decided the
        candidate pool; this only changes how the top_k budget is split across it."""
        by_doc: dict[str, list[tuple[str, float]]] = {}
        for cid in chunk_ids:
            if cid not in self._pos_by_id:
                continue
            doc = self._chunk_by_id[cid].doc
            by_doc.setdefault(doc, []).append((cid, float(scores[self._pos_by_id[cid]])))
        for scored in by_doc.values():
            scored.sort(key=lambda x: x[1], reverse=True)

        floor = max(1, SETTINGS.top_k // len(by_doc)) if by_doc else 0
        top: list[tuple[str, float]] = []
        used: set[str] = set()
        for scored in by_doc.values():
            for cid, score in scored[:floor]:
                top.append((cid, score))
                used.add(cid)

        remaining = SETTINGS.top_k - len(top)
        if remaining > 0:
            leftover = sorted(
                (pair for scored in by_doc.values() for pair in scored if pair[0] not in used),
                key=lambda x: x[1], reverse=True,
            )
            top.extend(leftover[:remaining])

        top.sort(key=lambda x: x[1], reverse=True)
        return top[: SETTINGS.top_k]

    # --- resilience: never worse than plain hybrid ----------------------------
    def _hybrid_fallback(self, query: str, trace: list[TraceStep], reason: str) -> list[RetrievedChunk]:
        trace.append(TraceStep(reason, "hybrid retrieval only"))
        qvec = embed_one(query, role="query")
        hits = self.index.hybrid_search(query, qvec, k=SETTINGS.candidate_k, bm25_weight=SETTINGS.bm25_weight)
        return [RetrievedChunk(self.index.get(i), s, "fallback") for i, s in hits[: SETTINGS.top_k]]

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        if not self._trees:
            return self._hybrid_fallback(query, trace, "PageIndex unavailable")

        docs = self._select_documents(query, trace)
        if not docs:
            return self._hybrid_fallback(query, trace, "No document selected")

        # Fused BM25 + dense score, the same fusion hybrid search uses: caps each
        # navigated location and orders the final merge. Dense-only trimming cut gold
        # chunks that BM25 ranked in their section's top 5 (7 of 11 reached-but-cut cases).
        qvec = embed_one(query, role="query")
        scores = self.index.hybrid_scores(query, qvec, SETTINGS.bm25_weight)

        gathered_ids: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            tree = self._trees.get(doc)
            if tree is None:
                continue
            for cid in self._navigate(query, tree, tree.nodes[tree.root_id], trace, doc, scores):
                if cid not in seen:
                    seen.add(cid)
                    gathered_ids.append(cid)

        if not gathered_ids:
            return self._hybrid_fallback(query, trace, "Tree navigation returned nothing")

        return self._final_ranking(docs, gathered_ids, scores, trace)

    def _final_ranking(self, docs: list[str], gathered_ids: list[str], scores: np.ndarray,
                       trace: list[TraceStep]) -> list[RetrievedChunk]:
        top = self._merge_with_doc_floor(gathered_ids, scores)
        trace.append(TraceStep("Final ranking",
                                f"{len(gathered_ids)} candidates from structure → top {len(top)} by hybrid score"))
        return [RetrievedChunk(self._chunk_by_id[cid], s, "pageindex") for cid, s in top]

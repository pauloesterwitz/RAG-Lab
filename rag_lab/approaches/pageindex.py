"""PageIndex: vectorless, reasoning-based retrieval. Instead of embedding
similarity, an LLM navigates each document's prebuilt hierarchical tree index
(root -> section -> subsection, ToC/heading-derived) by reading node titles
and summaries — the way a reader flips to the right section of a report.
Candidate selection is 100% structural; only the final display order among
selected candidates uses a cheap dense-similarity tie-break."""
from __future__ import annotations

import json

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
            "each document's title and summary (like scanning a library shelf).\n\n"
            f"Documents:\n{listing}\n\nQuestion: {query}\n\n"
            'Reply JSON: {"documents": [{"doc": "<exact name>", "relevant": true|false, "reason": ".."}]}'
        )
        try:
            data = json.loads(generate(
                prompt, model=SETTINGS.pageindex_model, fmt=_ROOT_SCHEMA, num_predict=400, temperature=0.0,
            ))
            picked = [d["doc"] for d in data.get("documents", [])
                      if d.get("relevant") and d.get("doc") in self._doc_summaries]
            trace.append(TraceStep(
                "Root selection",
                "; ".join(f'{d["doc"]}: {d.get("reason", "")}' for d in data.get("documents", [])
                          if d.get("relevant")) or "(none relevant)",
            ))
        except Exception:
            picked = list(self._doc_summaries)
            trace.append(TraceStep("Root selection", "parse failed — searching all documents"))
        return picked[: SETTINGS.pageindex_max_docs]

    # --- per-hop tree descent -------------------------------------------------
    def _choose_children(self, query: str, tree: PITree, node, trace: list[TraceStep], doc: str) -> list[str]:
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
            "Pick up to 2 subsections worth descending into, OR say this section itself already "
            'answers it (stop_here=true, selected=[]). Reply JSON: '
            '{"selected": ["<id>", ...], "stop_here": true|false}'
        )
        try:
            data = json.loads(generate(
                prompt, model=SETTINGS.pageindex_model, fmt=_DESCEND_SCHEMA, num_predict=200, temperature=0.0,
            ))
            selected = [cid for cid in data.get("selected", []) if cid in tree.nodes][: SETTINGS.pageindex_max_breadth]
            stop_here = bool(data.get("stop_here", False)) or not selected
        except Exception:
            selected, stop_here = [], True
        if stop_here:
            trace.append(TraceStep(f"[{doc}] Stopped at: {node.title}", f"p.{node.page_start}-{node.page_end}"))
            return []
        trace.append(TraceStep(f"[{doc}] Descend from: {node.title}",
                                ", ".join(tree.nodes[c].title for c in selected)))
        return selected

    def _navigate(self, query: str, tree: PITree, node, trace: list[TraceStep], doc: str, depth: int = 0) -> list[str]:
        if not node.children or depth >= SETTINGS.pageindex_max_depth:
            trace.append(TraceStep(f"[{doc}] Leaf: {node.title}", f"p.{node.page_start}-{node.page_end}"))
            return node.chunk_ids
        selected = self._choose_children(query, tree, node, trace, doc)
        if not selected:
            return node.chunk_ids
        gathered: list[str] = []
        for cid in selected:
            gathered.extend(self._navigate(query, tree, tree.nodes[cid], trace, doc, depth + 1))
        return gathered

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

        gathered_ids: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            tree = self._trees.get(doc)
            if tree is None:
                continue
            for cid in self._navigate(query, tree, tree.nodes[tree.root_id], trace, doc):
                if cid not in seen:
                    seen.add(cid)
                    gathered_ids.append(cid)

        if not gathered_ids:
            return self._hybrid_fallback(query, trace, "Tree navigation returned nothing")

        # structure decided the candidates; a cheap dense tie-break decides display order
        qvec = embed_one(query, role="query")
        dense = self.index.dense_scores(qvec)
        scored = sorted(
            ((cid, float(dense[self._pos_by_id[cid]])) for cid in gathered_ids if cid in self._pos_by_id),
            key=lambda x: x[1], reverse=True,
        )
        top = scored[: SETTINGS.top_k]
        trace.append(TraceStep("Final ranking",
                                f"{len(gathered_ids)} candidates from structure → top {len(top)} by similarity"))
        return [RetrievedChunk(self._chunk_by_id[cid], s, "pageindex") for cid, s in top]

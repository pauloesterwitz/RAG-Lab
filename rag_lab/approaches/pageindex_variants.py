"""Round-3 navigation variants, measured against PageIndex's single greedy pass.

Each subclasses PageIndexRAG and keeps name = "pageindex" so the retrieval harness
pairs it against the existing baselines; scripts/pi_variants_run.py swaps one into
the registry. Per-query state stays inside retrieve(): one instance is shared across
worker threads. Trace labels carry "Descend " so the harness counts the model calls.
"""
from __future__ import annotations

import json

import numpy as np

from .base import RetrievedChunk, TraceStep
from .pageindex import PageIndexRAG, _DESCEND_SCHEMA
from ..config import SETTINGS
from ..llamaswap_client import embed_one, generate

_NODES_SCHEMA = {
    "type": "object",
    "properties": {"nodes": {"type": "array", "items": {"type": "integer"}}},
    "required": ["nodes"],
}
_GRADE_SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "array", "items": {"type": "number"}}},
    "required": ["scores"],
}


class WholeTreePageIndex(PageIndexRAG):
    """V1: one call per document over the whole tree, upstream's documented prompt
    shape, instead of a hop-by-hop descent. Sections are listed by local integer
    index: node ids average 34 characters, which wastes output tokens and invites
    hallucinated ids."""

    def _navigate(self, query: str, tree, node, trace: list[TraceStep], doc: str,
                  scores: np.ndarray, depth: int = 0) -> list[str]:
        nodes = [n for n in tree.nodes.values() if n.id != tree.root_id]
        if not nodes:
            return self._top_chunks(tree.nodes[tree.root_id].chunk_ids, scores)
        listing = "\n".join(
            f'[{k}] {"  " * max(0, n.level - 1)}"{n.title}" (p.{n.page_start}-{n.page_end})'
            for k, n in enumerate(nodes)
        )
        prompt = (
            f'You are given a question and the section structure of "{doc}".\n\n'
            f"Sections:\n{listing}\n\nQuestion: {query}\n\n"
            "List the index numbers of ALL sections likely to contain the answer, most "
            'promising first. Reply JSON: {"nodes": [<index numbers>]}'
        )
        try:
            budget = max(1200, 4 * len(nodes) + 400)
            data = json.loads(generate(prompt, model=SETTINGS.pageindex_model, fmt=_NODES_SCHEMA,
                                       num_predict=budget, temperature=0.0))
            picked = [nodes[i] for i in data.get("nodes", []) if isinstance(i, int) and 0 <= i < len(nodes)]
        except Exception:
            trace.append(TraceStep(f"[{doc}] Descend failed: whole tree", "parse failed"))
            return self._top_chunks(tree.nodes[tree.root_id].chunk_ids, scores)
        if not picked:
            trace.append(TraceStep(f"[{doc}] Stopped at: whole tree", "no section picked"))
            return self._top_chunks(tree.nodes[tree.root_id].chunk_ids, scores)
        trace.append(TraceStep(f"[{doc}] Descend from: whole tree",
                                f"{len(nodes)} sections listed, picked {len(picked)}: "
                                + ", ".join(n.title[:40] for n in picked[:5])))
        gathered: list[str] = []
        for n in picked:
            gathered.extend(self._top_chunks(n.chunk_ids, scores))
        return gathered


class TreePriorPageIndex(WholeTreePageIndex):
    """V4: the tree never excludes anything. Corpus-wide hybrid search proposes the
    candidates; chunks inside the sections the model picked get a score bonus. The
    only variant not bounded by the document-scoped pool."""

    alpha = 0.10

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        if not self._trees:
            return self._hybrid_fallback(query, trace, "PageIndex unavailable")
        docs = self._select_documents(query, trace)
        qvec = embed_one(query, role="query")
        scores = self.index.hybrid_scores(query, qvec, SETTINGS.bm25_weight)

        boosted: set[str] = set()
        for doc in docs:
            tree = self._trees.get(doc)
            if tree is not None:
                boosted.update(super()._navigate(query, tree, tree.nodes[tree.root_id], trace, doc, scores))

        pool = np.argsort(-scores)[: SETTINGS.candidate_k]
        ranked = sorted(
            ((self.index.chunks[i].id, float(scores[i]) + (self.alpha if self.index.chunks[i].id in boosted else 0.0))
             for i in pool),
            key=lambda x: x[1], reverse=True,
        )[: SETTINGS.top_k]
        hits = sum(cid in boosted for cid, _ in ranked)
        trace.append(TraceStep("Final ranking",
                                f"{len(pool)} corpus-wide candidates, {len(boosted)} chunks in picked sections "
                                f"(+{self.alpha}) → {hits} of {len(ranked)} boosted"))
        return [RetrievedChunk(self._chunk_by_id[cid], s, "pageindex") for cid, s in ranked]


class _PickAllPageIndex(PageIndexRAG):
    """Shared by the two iterative variants: the base class cuts the model's picks to
    max_breadth inside _choose_children and discards the rest. Both variants need the
    full score-sorted list, so they ask for it here instead."""

    def _choose_all(self, query: str, tree, node, trace: list[TraceStep], doc: str,
                    scores: np.ndarray, tag: str = "") -> list[str] | None:
        """Score-sorted list of every section the model judged worth entering, or None
        to stop at this node."""
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
            budget = max(600, 100 * len(node.children) + 200)
            data = json.loads(generate(prompt, model=SETTINGS.pageindex_model, fmt=_DESCEND_SCHEMA,
                                       num_predict=budget, temperature=0.0))
            picked = [cid for cid in data.get("selected", []) if cid in tree.nodes]
            if data.get("stop_here") or not picked:
                trace.append(TraceStep(f"[{doc}] Stopped at: {node.title}", f"p.{node.page_start}-{node.page_end}"))
                return None
        except Exception:
            trace.append(TraceStep(f"[{doc}] Descend failed: {node.title}", "call or parse failed"))
            return None
        picked.sort(key=lambda cid: self._node_best(tree.nodes[cid], scores), reverse=True)
        trace.append(TraceStep(f"[{doc}] Descend from: {node.title}{tag}",
                                f"model picked {len(picked)}: " + ", ".join(tree.nodes[c].title[:40] for c in picked[:4])))
        return picked


class BeamPageIndex(_PickAllPageIndex):
    """V3: one best-first frontier across all selected documents instead of a
    depth-first walk that keeps 2 children per hop. Backtracking is intrinsic: a
    branch deprioritised early stays on the frontier and can be expanded later.
    Frontier scoring is free, since the fused scores are corpus-wide."""

    beam_calls = 6

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        if not self._trees:
            return self._hybrid_fallback(query, trace, "PageIndex unavailable")
        docs = self._select_documents(query, trace)
        if not docs:
            return self._hybrid_fallback(query, trace, "No document selected")
        qvec = embed_one(query, role="query")
        scores = self.index.hybrid_scores(query, qvec, SETTINGS.bm25_weight)

        frontier: list[tuple[float, str, str, int]] = []  # (score, doc, node id, depth)
        for doc in docs:
            tree = self._trees.get(doc)
            if tree is not None:
                root = tree.nodes[tree.root_id]
                frontier.append((self._node_best(root, scores), doc, root.id, 0))
        gathered: list[str] = []
        seen: set[str] = set()

        def add(ids):
            for cid in ids:
                if cid not in seen:
                    seen.add(cid)
                    gathered.append(cid)

        for i in range(self.beam_calls):
            frontier.sort(key=lambda e: e[0], reverse=True)
            nxt = next((e for e in frontier
                        if self._trees[e[1]].nodes[e[2]].children and e[3] < SETTINGS.pageindex_max_depth), None)
            if nxt is None:
                break
            frontier.remove(nxt)
            _, doc, nid, depth = nxt
            tree = self._trees[doc]
            picked = self._choose_all(query, tree, tree.nodes[nid], trace, doc, scores,
                                      f" (beam {i + 1}/{self.beam_calls})")
            if picked is None:
                add(self._top_chunks(tree.nodes[nid].chunk_ids, scores))
                continue
            for cid in picked:
                child = tree.nodes[cid]
                if child.children and depth + 1 < SETTINGS.pageindex_max_depth:
                    frontier.append((self._node_best(child, scores), doc, cid, depth + 1))
                else:
                    add(self._top_chunks(child.chunk_ids, scores))

        if not gathered:
            if not frontier:
                return self._hybrid_fallback(query, trace, "Tree navigation returned nothing")
            best = max(frontier, key=lambda e: e[0])
            add(self._top_chunks(self._trees[best[1]].nodes[best[2]].chunk_ids, scores))
        return self._final_ranking(docs, gathered, scores, trace)


class SufficiencyReentryPageIndex(_PickAllPageIndex):
    """V2: navigate, grade what came back in one batched call, and on failure re-enter
    the tree from the sections the breadth cut discarded in round 1. Grading fails
    CLOSED: a parse error counts as insufficient, otherwise the variant would
    reproduce the premature commitment it exists to test."""

    max_rounds = 3
    reentry_per_round = 2
    relevance_threshold = 0.5
    min_good = 3

    def _navigate(self, query, tree, node, trace, doc, scores, depth=0, frontier=None):
        if not node.children or depth >= SETTINGS.pageindex_max_depth:
            trace.append(TraceStep(f"[{doc}] Leaf: {node.title}", f"p.{node.page_start}-{node.page_end}"))
            return self._top_chunks(node.chunk_ids, scores)
        picked = self._choose_all(query, tree, node, trace, doc, scores)
        if picked is None:
            return self._top_chunks(node.chunk_ids, scores)
        selected = picked[: SETTINGS.pageindex_max_breadth]
        if frontier is not None:
            for cid in picked[SETTINGS.pageindex_max_breadth:]:
                frontier.append((self._node_best(tree.nodes[cid], scores), doc, cid, tree.nodes[cid].level))
        gathered: list[str] = []
        for cid in selected:
            gathered.extend(self._navigate(query, tree, tree.nodes[cid], trace, doc, scores, depth + 1, frontier))
        return gathered

    def _grade(self, query: str, chunk_ids: list[str], trace: list[TraceStep], rnd: int) -> int:
        texts = [self._chunk_by_id[cid].text for cid in chunk_ids[: 2 * SETTINGS.top_k]]
        if not texts:
            return 0
        blocks = "\n\n".join(f"[{j}] {t[:900]}" for j, t in enumerate(texts))
        prompt = (
            f"Question: {query}\n\nGrade how well EACH passage helps answer the question, "
            f"from 0.0 (irrelevant) to 1.0 (directly answers).\n\n{blocks}\n\n"
            f'Reply JSON: {{"scores": [<one number per passage, {len(texts)} total, in order>]}}.'
        )
        try:
            raw = generate(prompt, model=SETTINGS.pageindex_model, fmt=_GRADE_SCHEMA,
                           num_predict=max(256, 8 * len(texts) + 100), temperature=0.0)
            sc = [max(0.0, min(1.0, float(s))) for s in json.loads(raw).get("scores", [])]
        except Exception:
            sc = []  # fail closed: no grades means not sufficient
        good = sum(s >= self.relevance_threshold for s in sc)
        trace.append(TraceStep(f"[sufficiency] Descend gate: round {rnd}",
                                f"{good} of {len(texts)} passages >= {self.relevance_threshold}"))
        return good

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        if not self._trees:
            return self._hybrid_fallback(query, trace, "PageIndex unavailable")
        docs = self._select_documents(query, trace)
        if not docs:
            return self._hybrid_fallback(query, trace, "No document selected")
        qvec = embed_one(query, role="query")
        scores = self.index.hybrid_scores(query, qvec, SETTINGS.bm25_weight)

        frontier: list[tuple[float, str, str, int]] = []
        gathered: list[str] = []
        seen: set[str] = set()

        def add(ids):
            for cid in ids:
                if cid not in seen:
                    seen.add(cid)
                    gathered.append(cid)

        for doc in docs:
            tree = self._trees.get(doc)
            if tree is not None:
                add(self._navigate(query, tree, tree.nodes[tree.root_id], trace, doc, scores, 0, frontier))

        for rnd in range(1, self.max_rounds + 1):
            if not gathered or self._grade(query, gathered, trace, rnd) >= self.min_good:
                break
            frontier.sort(key=lambda e: e[0], reverse=True)
            batch, frontier = frontier[: self.reentry_per_round], frontier[self.reentry_per_round:]
            if not batch:
                break
            for _, doc, nid, level in batch:
                tree = self._trees[doc]
                add(self._navigate(query, tree, tree.nodes[nid], trace, doc, scores, level, frontier))

        if not gathered:
            return self._hybrid_fallback(query, trace, "Tree navigation returned nothing")
        return self._final_ranking(docs, gathered, scores, trace)

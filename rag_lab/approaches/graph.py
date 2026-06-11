"""GraphRAG retrieval over the prebuilt entity graph. Extract entities from the
query, anchor them in the graph, expand to neighbours, and gather the chunks
those entities appear in — scored by entity overlap + dense similarity, with a
hybrid backfill so recall stays healthy."""
from __future__ import annotations

import json
from typing import Optional

import networkx as nx
import numpy as np

from .base import Approach, RetrievedChunk, TraceStep
from ..config import SETTINGS
from ..ollama_client import embed_one, generate
from ..graph_build import GRAPH_FILE, MAP_FILE, COMM_FILE, graph_exists, _norm

_QENT_SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
}


class GraphRAG(Approach):
    name = "graph"

    def __init__(self, index):
        super().__init__(index)
        self._G: Optional[nx.Graph] = None
        self._entity_chunks: dict[str, list[str]] = {}
        self._communities: dict = {}
        self._id2pos = {c.id: i for i, c in enumerate(index.chunks)}
        self._load()

    def _load(self):
        if not graph_exists():
            return
        self._G = nx.node_link_graph(json.loads(GRAPH_FILE.read_text()), edges="links")
        self._entity_chunks = json.loads(MAP_FILE.read_text())
        if COMM_FILE.exists():
            self._communities = json.loads(COMM_FILE.read_text())

    def _query_entities(self, query: str) -> list[str]:
        prompt = (
            f"List the key entities/concepts in this question for a knowledge-graph lookup.\n"
            f'Question: {query}\nReply JSON: {{"entities": ["..", ".."]}}'
        )
        try:
            raw = generate(prompt, fmt=_QENT_SCHEMA, num_predict=120, temperature=0.0)
            return [_norm(e) for e in json.loads(raw).get("entities", []) if e.strip()]
        except Exception:
            return [_norm(query)]

    def _match_nodes(self, qents: list[str]) -> list[str]:
        if self._G is None:
            return []
        nodes = list(self._G.nodes())
        matched: set[str] = set()
        for qe in qents:
            if not qe:
                continue
            if self._G.has_node(qe):
                matched.add(qe)
                continue
            qtok = set(qe.split())
            for n in nodes:
                if qe in n or n in qe or (qtok & set(n.split())):
                    matched.add(n)
        return list(matched)

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        qvec = embed_one(query, role="query")

        if self._G is None:
            trace.append(TraceStep("Graph unavailable", "falling back to hybrid retrieval"))
            hits = self.index.hybrid_search(query, qvec, k=SETTINGS.top_k, bm25_weight=SETTINGS.bm25_weight)
            return [RetrievedChunk(self.index.get(i), s, "fallback") for i, s in hits]

        qents = self._query_entities(query)
        seeds = self._match_nodes(qents)
        trace.append(TraceStep("Query entities", ", ".join(qents) or "(none)"))

        # expand to 1-hop neighbours
        expanded: set[str] = set(seeds)
        for s in seeds:
            if self._G.has_node(s):
                expanded.update(self._G.neighbors(s))
        trace.append(TraceStep("Graph anchor", f"{len(seeds)} seed entities → {len(expanded)} with neighbours"))

        # gather candidate chunks + entity-overlap counts
        chunk_overlap: dict[str, int] = {}
        for e in expanded:
            for cid in self._entity_chunks.get(e, []):
                chunk_overlap[cid] = chunk_overlap.get(cid, 0) + 1

        dense = self.index.dense_scores(qvec)
        scored: list[tuple[int, float]] = []
        for cid, ov in chunk_overlap.items():
            pos = self._id2pos.get(cid)
            if pos is None:
                continue
            score = 0.6 * float(dense[pos]) + 0.4 * min(ov / 3.0, 1.0)
            scored.append((pos, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        # community context for transparency
        comm_ids = {self._G.nodes[s].get("community") for s in seeds if self._G.has_node(s)}
        comm_ids.discard(None)
        for c in list(comm_ids)[:2]:
            summ = self._communities.get(str(c), {}).get("summary")
            if summ:
                trace.append(TraceStep(f"Community {c} summary", summ[:300]))

        picked = scored[: SETTINGS.top_k]
        # backfill with hybrid search if the graph was too sparse
        if len(picked) < SETTINGS.top_k:
            have = {p for p, _ in picked}
            for i, s in self.index.hybrid_search(query, qvec, k=SETTINGS.top_k * 2, bm25_weight=SETTINGS.bm25_weight):
                if i not in have:
                    picked.append((i, s * 0.5))
                    if len(picked) >= SETTINGS.top_k:
                        break
            trace.append(TraceStep("Hybrid backfill", "graph was sparse; topped up with hybrid hits"))

        return [RetrievedChunk(self.index.get(i), s, "graph") for i, s in picked[: SETTINGS.top_k]]

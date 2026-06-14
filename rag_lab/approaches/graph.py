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
        self._chunk_entities: dict[str, set[str]] = {}  # inverse map
        self._communities: dict = {}
        self._id2pos = {c.id: i for i, c in enumerate(index.chunks)}
        self._load()

    def _load(self):
        if not graph_exists():
            return
        self._G = nx.node_link_graph(json.loads(GRAPH_FILE.read_text()), edges="links")
        self._entity_chunks = json.loads(MAP_FILE.read_text())
        # invert to chunk -> entities so we can boost hybrid candidates by connectivity
        for ent, cids in self._entity_chunks.items():
            for cid in cids:
                self._chunk_entities.setdefault(cid, set()).add(ent)
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
        """Match query entities to graph nodes by containment or strong token
        overlap — NOT bare substring/single-token, which floods with noise."""
        if self._G is None:
            return []
        nodes = list(self._G.nodes())
        matched: set[str] = set()
        for qe in qents:
            if not qe or len(qe) < 3:
                continue
            if self._G.has_node(qe):
                matched.add(qe)
                continue
            qtok = set(qe.split())
            if not qtok:
                continue
            for n in nodes:
                ntok = set(n.split())
                inter = qtok & ntok
                if not inter:
                    continue
                # full containment either way, or >=60% token overlap
                if qtok <= ntok or ntok <= qtok or len(inter) / min(len(qtok), len(ntok)) >= 0.6:
                    matched.add(n)
        return list(matched)

    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        qvec = embed_one(query, role="query")

        # ALWAYS start from a wide hybrid pool — this guarantees graph recall is
        # never worse than plain. The graph then RE-RANKS by entity connectivity.
        cand = self.index.hybrid_search(
            query, qvec, k=SETTINGS.candidate_k, bm25_weight=SETTINGS.bm25_weight
        )
        if self._G is None:
            trace.append(TraceStep("Graph unavailable", "hybrid retrieval only"))
            return [RetrievedChunk(self.index.get(i), s, "fallback") for i, s in cand[: SETTINGS.top_k]]

        qents = self._query_entities(query)
        seeds = self._match_nodes(qents)
        trace.append(TraceStep("Query entities", ", ".join(qents) or "(none)"))

        # expand seeds to 1-hop neighbours → the set of query-relevant entities
        expanded: set[str] = set(seeds)
        for s in seeds:
            if self._G.has_node(s):
                expanded.update(self._G.neighbors(s))
        trace.append(TraceStep("Graph anchor", f"{len(seeds)} seed entities → {len(expanded)} with neighbours"))

        # Boost each hybrid candidate by how many of ITS entities are connected to
        # the query entities. Hybrid (dense+BM25) stays dominant; graph adds a bonus.
        boosted: list[tuple[int, float, int]] = []
        for i, hscore in cand:
            ents = self._chunk_entities.get(self.index.get(i).id, ())
            overlap = sum(1 for e in ents if e in expanded)
            score = hscore + 0.30 * min(overlap / 3.0, 1.0)
            boosted.append((i, score, overlap))
        boosted.sort(key=lambda x: x[1], reverse=True)

        # community context for transparency
        comm_ids = {self._G.nodes[s].get("community") for s in seeds if self._G.has_node(s)}
        comm_ids.discard(None)
        for c in list(comm_ids)[:2]:
            summ = self._communities.get(str(c), {}).get("summary")
            if summ:
                trace.append(TraceStep(f"Community {c} summary", summ[:300]))
        trace.append(TraceStep(
            "Entity-boosted rerank",
            f"{sum(1 for _, _, o in boosted if o > 0)}/{len(boosted)} hybrid candidates boosted by graph",
        ))

        return [RetrievedChunk(self.index.get(i), s, "graph") for i, s, _ in boosted[: SETTINGS.top_k]]

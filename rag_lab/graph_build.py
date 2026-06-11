"""GraphRAG offline build: LLM-extract entities + relationships per chunk, build
an entity graph, detect communities, and summarize them. Bounded by
graph_max_chunks (round-robin across documents) and cached to disk so it only
runs during (re-)embedding."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import networkx as nx

from .config import INDEX_DIR, SETTINGS
from .ollama_client import generate, generate_many
from .store import BaseIndex

GRAPH_DIR = INDEX_DIR / "graph"
GRAPH_FILE = GRAPH_DIR / "graph.json"
COMM_FILE = GRAPH_DIR / "communities.json"
MAP_FILE = GRAPH_DIR / "entity_chunks.json"
META_FILE = GRAPH_DIR / "meta.json"

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                "required": ["name"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "target"],
            },
        },
    },
    "required": ["entities"],
}

_EXTRACT_PROMPT = (
    "Extract the key entities (concepts, methods, systems, people, organizations) and the "
    "relationships between them from the passage below. Keep entity names short and canonical.\n\n"
    "Passage:\n{text}\n\n"
    'Reply JSON: {{"entities":[{{"name":"..","type":".."}}], '
    '"relationships":[{{"source":"..","target":"..","description":".."}}]}}'
)

_STOP = {"the", "a", "an", "this", "that", "it", "they", "we", "he", "she", "i", "you"}


def _norm(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip().lower())
    name = name.strip(" .,:;\"'()[]")
    return name


def _select_chunks(index: BaseIndex, cap: int) -> list[int]:
    """Round-robin across documents so the graph covers every PDF."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(index.chunks):
        by_doc[c.doc].append(i)
    order: list[int] = []
    docs = list(by_doc.values())
    pos = 0
    while len(order) < min(cap, len(index.chunks)):
        progressed = False
        for lst in docs:
            if pos < len(lst):
                order.append(lst[pos])
                progressed = True
                if len(order) >= cap:
                    break
        if not progressed:
            break
        pos += 1
    return order


def build_graph(index: BaseIndex, progress: Optional[Callable[[str, float, str], None]] = None) -> dict:
    progress = progress or (lambda s, f, m: print(f"[graph {f*100:.0f}%] {m}"))
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sel = _select_chunks(index, SETTINGS.graph_max_chunks)
    progress("graph", 0.0, f"Extracting entities from {len(sel)} chunks…")

    prompts = [_EXTRACT_PROMPT.format(text=index.get(i).text[:1600]) for i in sel]
    raws = generate_many(
        prompts,
        concurrency=SETTINGS.graph_concurrency,
        fmt=_EXTRACT_SCHEMA,
        num_predict=512,
        temperature=0.0,
    )

    G = nx.Graph()
    entity_chunks: dict[str, set[str]] = defaultdict(set)
    entity_type: dict[str, str] = {}

    for k, (i, raw) in enumerate(zip(sel, raws)):
        chunk = index.get(i)
        try:
            data = json.loads(raw)
        except Exception:
            continue
        ents = []
        for e in data.get("entities", []):
            nm = _norm(e.get("name", ""))
            if len(nm) < 3 or nm in _STOP:
                continue
            ents.append(nm)
            entity_chunks[nm].add(chunk.id)
            entity_type.setdefault(nm, (e.get("type") or "concept").lower())
            if not G.has_node(nm):
                G.add_node(nm, type=entity_type[nm], count=0)
            G.nodes[nm]["count"] += 1
        # relationships
        for r in data.get("relationships", []):
            s, t = _norm(r.get("source", "")), _norm(r.get("target", ""))
            if len(s) < 3 or len(t) < 3 or s == t:
                continue
            for nm in (s, t):
                if not G.has_node(nm):
                    G.add_node(nm, type=entity_type.get(nm, "concept"), count=0)
                entity_chunks[nm].add(chunk.id)
            if G.has_edge(s, t):
                G[s][t]["weight"] += 1
            else:
                G.add_edge(s, t, weight=1, description=(r.get("description") or "")[:200])
        if progress and k % 10 == 0:
            progress("graph", 0.0 + (k / max(len(sel), 1)) * 0.7, f"Parsed {k}/{len(sel)} chunks")

    # prune singletons with no edges and count==1 to reduce noise
    for n in [n for n, d in G.degree() if d == 0 and G.nodes[n].get("count", 0) <= 1]:
        G.remove_node(n)

    progress("graph", 0.75, f"Detecting communities over {G.number_of_nodes()} entities…")
    communities = _detect_communities(G)
    comm_summaries = _summarize_communities(G, communities, index, entity_chunks, progress)

    # persist
    nx.write_gml  # noqa (keep import warm)
    GRAPH_FILE.write_text(json.dumps(nx.node_link_data(G, edges="links")))
    MAP_FILE.write_text(json.dumps({k: sorted(v) for k, v in entity_chunks.items()}))
    COMM_FILE.write_text(json.dumps(comm_summaries, indent=2))
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks_processed": len(sel),
        "entities": G.number_of_nodes(),
        "relationships": G.number_of_edges(),
        "communities": len(comm_summaries),
        "build_seconds": round(time.time() - t0, 1),
        "model": SETTINGS.gen_model,
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    progress("graph", 1.0, f"Graph: {meta['entities']} entities, {meta['relationships']} edges, {meta['communities']} communities.")
    return meta


def _detect_communities(G: nx.Graph) -> list[list[str]]:
    if G.number_of_nodes() == 0:
        return []
    try:
        from networkx.algorithms.community import louvain_communities

        comms = louvain_communities(G, weight="weight", seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities

        comms = greedy_modularity_communities(G, weight="weight")
    return [sorted(c) for c in comms]


def _summarize_communities(G, communities, index, entity_chunks, progress) -> dict:
    # assign community id on nodes
    out: dict[str, dict] = {}
    # summarize the largest ~20 communities to bound cost
    ranked = sorted(enumerate(communities), key=lambda x: -len(x[1]))[:20]
    chunk_by_id = {c.id: c for c in index.chunks}
    for rank, (cid, members) in enumerate(ranked):
        for m in members:
            if G.has_node(m):
                G.nodes[m]["community"] = cid
        if len(members) < 2:
            continue
        top_entities = sorted(members, key=lambda n: -G.nodes[n].get("count", 0))[:12]
        snippet_ids = set()
        for e in top_entities[:4]:
            snippet_ids.update(list(entity_chunks.get(e, []))[:2])
        snippets = [chunk_by_id[i].text[:300] for i in list(snippet_ids)[:3] if i in chunk_by_id]
        prompt = (
            "Summarize this community of related concepts in 2-3 sentences for a knowledge "
            f"graph index.\nKey entities: {', '.join(top_entities)}\n\n"
            "Representative text:\n" + "\n---\n".join(snippets)
        )
        try:
            summary = generate(prompt, num_predict=180, temperature=0.2).strip()
        except Exception:
            summary = "Entities: " + ", ".join(top_entities)
        out[str(cid)] = {"entities": top_entities, "summary": summary, "size": len(members)}
        if progress:
            progress("graph", 0.85 + (rank / max(len(ranked), 1)) * 0.14, f"Summarized community {rank+1}/{len(ranked)}")
    return out


def graph_exists() -> bool:
    return GRAPH_FILE.exists() and MAP_FILE.exists()


def get_graph_meta() -> Optional[dict]:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return None

"""PageIndex Hybrid: tree navigation plus a document-scoped hybrid search over the
same selected documents, merged by reciprocal rank fusion (the "hybrid tree search"
VectifyAI documents for PageIndex). Ranking both candidate lists by the same fused
score would let the value list alone decide the top k; RRF rewards chunks that both
searches surface."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .base import RetrievedChunk, TraceStep
from .pageindex import PageIndexRAG
from ..config import SETTINGS


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    out: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            out[cid] = out.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return out


class PageIndexHybridRAG(PageIndexRAG):
    name = "pageindex_hybrid"
    use_tree = True  # False = value-only control: same selected documents, no tree candidates

    def __init__(self, index):
        super().__init__(index)
        by_doc: dict[str, list[int]] = {}
        for i, c in enumerate(index.chunks):
            by_doc.setdefault(c.doc, []).append(i)
        self._doc_pos = {doc: np.array(pos) for doc, pos in by_doc.items()}

    def _final_ranking(self, docs: list[str], gathered_ids: list[str], scores: np.ndarray,
                       trace: list[TraceStep]) -> list[RetrievedChunk]:
        tree = sorted(gathered_ids, key=lambda cid: scores[self._pos_by_id[cid]], reverse=True) if self.use_tree else []
        values = []
        for doc in docs:
            pos = self._doc_pos.get(doc)
            if pos is not None:
                values.append([self.index.chunks[i].id for i in pos[np.argsort(-scores[pos])[: SETTINGS.candidate_k]]])
        fused = rrf([tree, *values])
        rrf_scores = np.zeros(len(scores))
        for cid, s in fused.items():
            rrf_scores[self._pos_by_id[cid]] = s
        top = self._merge_with_doc_floor(list(fused), rrf_scores)

        tree_set, value_set = set(tree), {cid for v in values for cid in v}
        stage = {cid: "pageindex:both" if cid in tree_set and cid in value_set
                 else "pageindex:tree" if cid in tree_set else "pageindex:value" for cid, _ in top}
        counts = Counter(s.split(":")[1] for s in stage.values())
        trace.append(TraceStep("Final ranking",
                                f"{len(tree_set)} tree + {len(value_set)} value candidates → top {len(top)} by RRF: "
                                + ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))))
        return [RetrievedChunk(self._chunk_by_id[cid], s, stage[cid]) for cid, s in top]

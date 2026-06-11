"""BaseIndex: the shared substrate every approach retrieves over.

Holds the chunk list, a normalized embedding matrix (dense), and a BM25 index
(sparse). Provides dense / sparse / hybrid search. Persisted to disk so the Vue
app can re-embed once and have every approach reuse the same vectors."""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

_TOK = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class BaseIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        assert len(chunks) == embeddings.shape[0]
        self.chunks = chunks
        self.embeddings = _normalize(embeddings.astype(np.float32))
        self._tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokens) if chunks else None

    # --- search ---------------------------------------------------------
    def dense_scores(self, query_vec: list[float] | np.ndarray) -> np.ndarray:
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        return self.embeddings @ q

    def bm25_scores(self, query: str) -> np.ndarray:
        if self.bm25 is None:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    def dense_search(self, query_vec, k: int) -> list[tuple[int, float]]:
        scores = self.dense_scores(query_vec)
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def hybrid_search(
        self, query: str, query_vec, k: int, bm25_weight: float = 0.35
    ) -> list[tuple[int, float]]:
        dense = _minmax(self.dense_scores(query_vec))
        sparse = _minmax(self.bm25_scores(query))
        fused = (1 - bm25_weight) * dense + bm25_weight * sparse
        idx = np.argsort(-fused)[:k]
        return [(int(i), float(fused[i])) for i in idx]

    def get(self, i: int) -> Chunk:
        return self.chunks[i]

    def __len__(self) -> int:
        return len(self.chunks)

    # --- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "BaseIndex":
        emb = np.load(path / "embeddings.npy")
        chunks: list[Chunk] = []
        with open(path / "chunks.jsonl") as f:
            for line in f:
                d = json.loads(line)
                chunks.append(Chunk(**d))
        return cls(chunks, emb)

    @staticmethod
    def exists(path: Path) -> bool:
        return (path / "embeddings.npy").exists() and (path / "chunks.jsonl").exists()

"""Common types and base class for every RAG approach.

An approach is a retrieval strategy over the shared BaseIndex plus a (shared)
generation step. `retrieve()` returns the chunks; `run()` produces the answer
together with the chunks and a transparency trace the UI renders."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from ..ingest import Chunk
from ..store import BaseIndex
from ..config import SETTINGS
from ..generate import answer_with_context


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    stage: str = "retrieve"  # which step surfaced this chunk

    def to_dict(self) -> dict:
        return {
            "id": self.chunk.id,
            "doc": self.chunk.doc,
            "page_start": self.chunk.page_start,
            "page_end": self.chunk.page_end,
            "citation": self.chunk.citation(),
            "text": self.chunk.text,
            "score": round(float(self.score), 4),
            "stage": self.stage,
        }


@dataclass
class TraceStep:
    label: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RAGResult:
    approach: str
    query: str
    answer: str
    contexts: list[RetrievedChunk]
    trace: list[TraceStep] = field(default_factory=list)
    latency_s: float = 0.0

    def context_texts(self) -> list[str]:
        return [rc.chunk.text for rc in self.contexts]

    def to_dict(self) -> dict:
        return {
            "approach": self.approach,
            "query": self.query,
            "answer": self.answer,
            "contexts": [c.to_dict() for c in self.contexts],
            "trace": [t.to_dict() for t in self.trace],
            "latency_s": round(self.latency_s, 2),
        }


class Approach:
    name: str = "base"

    def __init__(self, index: BaseIndex):
        self.index = index

    # subclasses implement retrieve(); default run() does shared generation
    def retrieve(self, query: str, trace: list[TraceStep]) -> list[RetrievedChunk]:
        raise NotImplementedError

    def run(self, query: str, *, generate_answer: bool = True) -> RAGResult:
        import time

        t0 = time.time()
        trace: list[TraceStep] = []
        contexts = self.retrieve(query, trace)
        contexts = contexts[: SETTINGS.top_k]
        answer = ""
        if generate_answer:
            answer = answer_with_context(query, [c.chunk for c in contexts])
        return RAGResult(
            approach=self.name,
            query=query,
            answer=answer,
            contexts=contexts,
            trace=trace,
            latency_s=time.time() - t0,
        )

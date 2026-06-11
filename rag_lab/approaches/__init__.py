"""Approach registry + factory."""
from __future__ import annotations

from ..store import BaseIndex
from .base import Approach, RAGResult, RetrievedChunk, TraceStep
from .plain import PlainRAG
from .rerank import RerankRAG
from .hyde import HydeRAG
from .corrective import CorrectiveRAG
from .agentic import AgenticRAG
from .graph import GraphRAG

REGISTRY: dict[str, type[Approach]] = {
    cls.name: cls
    for cls in (PlainRAG, RerankRAG, HydeRAG, CorrectiveRAG, AgenticRAG, GraphRAG)
}


def get_approach(name: str, index: BaseIndex) -> Approach:
    if name not in REGISTRY:
        raise ValueError(f"Unknown approach '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name](index)


__all__ = ["Approach", "RAGResult", "RetrievedChunk", "TraceStep", "REGISTRY", "get_approach"]

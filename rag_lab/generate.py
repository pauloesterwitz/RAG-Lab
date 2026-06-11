"""Shared answer generation. Every approach funnels its final retrieved chunks
through here, so the answer style (grounded, cited, quoting) is consistent."""
from __future__ import annotations

from .ollama_client import generate
from .ingest import Chunk

SYSTEM = (
    "You are a precise research assistant. Answer the user's question using ONLY the "
    "numbered sources provided. Ground every claim in the sources and cite them inline "
    "as [1], [2], etc. When you state a key fact, include a short verbatim quote from the "
    "source in quotation marks. If the sources do not contain the answer, say so plainly. "
    "Do not invent citations or facts."
)


def format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c.citation()})\n{c.text}")
    return "\n\n".join(blocks)


def answer_with_context(query: str, chunks: list[Chunk], *, model: str | None = None) -> str:
    if not chunks:
        return "I could not retrieve any relevant context to answer this question."
    context = format_context(chunks)
    prompt = (
        f"Sources:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Write a clear, well-grounded answer. Cite sources inline as [n] and include short "
        "verbatim quotes for key facts."
    )
    return generate(prompt, model=model, system=SYSTEM).strip()

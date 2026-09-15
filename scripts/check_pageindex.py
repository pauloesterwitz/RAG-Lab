"""Self-checks for PageIndex heuristics. No LLM, index or corpus needed.

  .venv/bin/python scripts/check_pageindex.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_lab.pageindex_build import _sentence_like, _toc_usable  # noqa: E402

# Table of contents: bookmarks pointing nowhere or only at the first pages carry no structure.
assert not _toc_usable([-1] * 7 + [1, 1], 10)
assert not _toc_usable([1, 1, 2, 2, 2, 1, 2, 1, 2], 9)
assert _toc_usable([1, 3, 7, 12, 30, 55], 60)

# Heading candidates: glossary lines and pull quotes out, real headings in.
assert _sentence_like("Grounding: Grounding is the process of connecting a model's outputs to verifiable")
assert _sentence_like("(NSP), where it determines if two sentences logically follow each other.")
assert _sentence_like("deployment, Guardrails are implemented as a final safety layer")
assert not _sentence_like("A Thought Leader's Perspective: Power and Responsibility")
assert not _sentence_like("Chapter 12: Exception Handling and Recovery in Multi-Agent Systems at Production Scale")
assert not _sentence_like("Index of Terms")

print("pageindex checks ok")

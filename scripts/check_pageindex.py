"""Self-checks for PageIndex heuristics. No LLM, index or corpus needed.

  .venv/bin/python scripts/check_pageindex.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_lab.pageindex_build import _FRONT_MATTER_RE, _leaf_body, _sentence_like, _toc_usable  # noqa: E402

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

# Leaf text starts at the section's own heading, even across a line break; else at the top.
page = "end of the DoS case. Case Study #6: Agents\nReflect Provider Values. The agent..."
assert _leaf_body("Case Study #6: Agents Reflect Provider Values", page, 1000).startswith("Case Study #6")
assert _leaf_body("Not On This Page", page, 1000).startswith("end of the DoS case")
assert len(_leaf_body("Case Study #6", page, 20)) == 20

# Front matter stays out of root summaries and section lists.
assert all(_FRONT_MATTER_RE.match(t) for t in ("Brief Contents", "Preface", "Acknowledgment", "About this Book"))
assert not any(_FRONT_MATTER_RE.match(t) for t in ("Chapter 1: Prompt Chaining", "Conclusion", "References"))

print("pageindex checks ok")

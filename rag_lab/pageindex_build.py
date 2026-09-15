"""PageIndex offline build: derive a hierarchical tree index per document — a
real PDF table of contents when the file has bookmarks, else a font-size
heading heuristic — summarize each node bottom-up (LLM calls cached to disk so
a multi-hour build resumes), and assign every node the flat chunks whose page
range overlaps it. Bolts on next to the flat BaseIndex / GraphRAG's graph,
same INDEX_DIR-sibling-directory pattern as graph_build.py."""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF

from .config import INDEX_DIR, SETTINGS
from .ingest import list_pdfs
from .llamaswap_client import generate
from .store import BaseIndex

PAGEINDEX_DIR = INDEX_DIR / "pageindex"
TREES_DIR = PAGEINDEX_DIR / "trees"
DOC_SUMMARY_FILE = PAGEINDEX_DIR / "doc_summaries.json"
META_FILE = PAGEINDEX_DIR / "meta.json"

_MIN_HEADING_CANDIDATES = 3
_HEADING_SIZE_RATIO = 1.15
# Large-font copyright/license text reads as a heading candidate by size alone —
# observed collapsing a doc's real content into one node with license-text-fragment
# "children" (Types of Machine Learning Algorithms.pdf). Filter before bucketing so
# boilerplate can't skew which sizes get picked as heading levels.
_BOILERPLATE_RE = re.compile(
    r"©|\ball rights reserved\b|\blicense[ds]?\b|\bterms of\b|\bshare-?alike\b|\bcc[- ]by\b"
    r"|\bcreative commons\b|\bnon-?commercial\b|\bdistributed under\b|\bpermits use\b"
    r"|\belectronic library\b|\brecommended citation\b|\btmp\.\d+",
    re.IGNORECASE,
)
_MIN_HEADING_CHARS = 3  # drop drop-caps/decorative single letters ("X") a font-size pass mistakes for a heading
_MAX_HEADING_CHARS = 120
_ALWAYS_HEADING_RE = re.compile(r"^(chapter|appendix|part)\b", re.IGNORECASE)


# Large-font body text (a glossary, pull quotes) passes the size test; real headings here run 3-7 words.
def _sentence_like(text: str) -> bool:
    if _ALWAYS_HEADING_RE.match(text):
        return False
    return len(text.split()) >= 10 or text[:1].islower() or text[:1] == "("


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


@dataclass
class PITreeNode:
    id: str
    title: str
    page_start: int
    page_end: int
    level: int = 0
    summary: str = ""
    children: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PITreeNode":
        return cls(**d)


@dataclass
class PITree:
    doc: str
    root_id: str
    nodes: dict[str, PITreeNode]

    def to_dict(self) -> dict:
        return {"doc": self.doc, "root_id": self.root_id, "nodes": {k: v.to_dict() for k, v in self.nodes.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "PITree":
        return cls(doc=d["doc"], root_id=d["root_id"],
                    nodes={k: PITreeNode.from_dict(v) for k, v in d["nodes"].items()})


# --- structure extraction ----------------------------------------------------
def _toc_entries(pdf_path: Path) -> Optional[list[tuple[int, str, int]]]:
    """[(level, title, page_1based), ...] from the PDF's own bookmarks, or None
    if it has no usable table of contents."""
    with fitz.open(pdf_path) as doc:
        toc = doc.get_toc(simple=True)
        page_count = doc.page_count
    if not toc or not _toc_usable([int(page) for _, _, page in toc], page_count):
        return None
    return [(int(lvl), (title or "").strip(), max(1, int(page))) for lvl, title, page in toc]


# Bookmarks that point nowhere (page -1) or only at the first pages carry no structure
# (observed: Using AI, 7 of 9 entries at page -1; AI Readiness, all 9 on pages 1-2 of 9).
def _toc_usable(pages: list[int], page_count: int) -> bool:
    valid = [p for p in pages if p >= 1]
    return len(valid) >= 0.7 * len(pages) and len(set(valid)) >= 3 and max(valid) >= 0.25 * page_count


def _heuristic_entries(pdf_path: Path) -> list[tuple[int, str, int]]:
    """Fallback for PDFs with no bookmarks: spans notably larger than the doc's
    modal (body) text size are heading candidates, bucketed into up to 3 levels
    by size rank. Free (reuses PyMuPDF's own span metadata), deterministic, no
    LLM call. Degrades to [] (a single flat root node) if too few candidates."""
    sizes: Counter = Counter()
    spans: list[tuple[int, float, str]] = []  # (page, size, text)
    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        text = sp.get("text", "").strip()
                        if not text:
                            continue
                        size = round(float(sp.get("size", 0)), 1)
                        sizes[size] += len(text)
                        spans.append((pno, size, text))
    if not sizes:
        return []
    body_size = sizes.most_common(1)[0][0]
    threshold = body_size * _HEADING_SIZE_RATIO
    candidates = [
        (pno, size, text) for pno, size, text in spans
        if size >= threshold and _MIN_HEADING_CHARS <= len(text) <= _MAX_HEADING_CHARS
        and not text.isdigit() and not _BOILERPLATE_RE.search(text)
    ]
    if len(candidates) < _MIN_HEADING_CANDIDATES:
        return []
    distinct_sizes = sorted({s for _, s, _ in candidates}, reverse=True)[:3]
    level_of = {s: i + 1 for i, s in enumerate(distinct_sizes)}
    entries = [(level_of[size], text, pno) for pno, size, text in candidates if size in level_of]

    # Merge line-wrapped headings: a heading spanning two PDF lines becomes two
    # same-page, same-level candidate spans — one real section torn into two
    # siblings (observed: "Chapter 12: Exception Handling and" / "Recovery",
    # same page, Agentic_Design_Patterns.pdf). If a candidate has no terminal
    # punctuation and the next one is the same page/level, it's a continuation.
    merged: list[tuple[int, str, int]] = []
    for level, text, pno in entries:
        prev = merged[-1] if merged else None
        if prev and prev[0] == level and prev[2] == pno and not re.search(r"[.!?:]$", prev[1]):
            merged[-1] = (level, f"{prev[1]} {text}", pno)
        else:
            merged.append((level, text, pno))
    return [e for e in merged if not _sentence_like(e[1])]


def _build_hierarchy(doc_name: str, doc_slug: str, entries: list[tuple[int, str, int]], page_count: int) -> PITree:
    root = PITreeNode(id=f"{doc_slug}#root", title=doc_name, page_start=1, page_end=page_count, level=0)
    nodes: dict[str, PITreeNode] = {root.id: root}
    ordered_ids: list[str] = []

    stack: list[PITreeNode] = [root]
    for i, (level, title, page) in enumerate(entries):
        nid = f"{doc_slug}#{i}"
        node = PITreeNode(id=nid, title=(title[:200] or f"Section {i + 1}"),
                           page_start=page, page_end=page_count, level=max(1, level))
        while len(stack) > 1 and stack[-1].level >= node.level:
            stack.pop()
        stack[-1].children.append(nid)
        nodes[nid] = node
        stack.append(node)
        ordered_ids.append(nid)

    # page_end = one before the next node at level <= this node's level, else page_count
    for i, nid in enumerate(ordered_ids):
        node = nodes[nid]
        end = page_count
        for j in range(i + 1, len(ordered_ids)):
            other = nodes[ordered_ids[j]]
            if other.level <= node.level:
                end = max(node.page_start, other.page_start - 1)
                break
        node.page_end = end

    # Real PDF bookmarks aren't always page-monotonic (a subsection can be
    # stamped one page past where its lookahead-computed parent range ends) —
    # widen each node to cover its children so parent ranges always stay a
    # superset of their descendants'.
    def _expand(node: PITreeNode) -> None:
        for cid in node.children:
            child = nodes[cid]
            _expand(child)
            node.page_end = max(node.page_end, child.page_end)

    _expand(root)
    return PITree(doc=doc_name, root_id=root.id, nodes=nodes)


def _assign_chunks(tree: PITree, index: BaseIndex, doc_name: str) -> None:
    """A chunk belongs to a node iff their page ranges overlap. Nodes nest by
    construction, so parents naturally end up with the union of their
    descendants' chunks — no separate bottom-up merge needed."""
    doc_chunks = [c for c in index.chunks if c.doc == doc_name]
    for node in tree.nodes.values():
        node.chunk_ids = [
            c.id for c in doc_chunks
            if not (c.page_end < node.page_start or c.page_start > node.page_end)
        ]


# --- bottom-up node summarization (LLM, per-item disk-cached) ---------------
_LEAF_PROMPT = (
    "Summarize this section of a document in 2-3 sentences, for a table-of-contents "
    "style index a reader would use to decide whether to read this section.\n\n"
    "Section title: {title}\n\nText:\n{text}\n\nSummary:"
)
_ROLLUP_PROMPT = (
    "Summarize this section of a document in 2-3 sentences, based on its subsections, "
    "for a table-of-contents style index.\n\n"
    "Section title: {title}\n\nSubsections:\n{children}\n\nSummary:"
)


def _subtree_depth(tree: PITree, node: PITreeNode) -> int:
    d, frontier = 0, list(node.children)
    while frontier:
        d += 1
        frontier = [cid for c in frontier for cid in tree.nodes[c].children]
    return d


def _summarize_tree(tree: PITree, index: BaseIndex, doc_name: str, cache_dir: Path,
                     node_done: Optional[Callable[[str], None]] = None) -> None:
    chunk_by_id = {c.id: c for c in index.chunks if c.doc == doc_name}
    cache_dir.mkdir(parents=True, exist_ok=True)

    limit = SETTINGS.pageindex_summary_chars
    for node in sorted(tree.nodes.values(), key=lambda n: _subtree_depth(tree, n)):  # leaves first
        if node.children:
            kids = [tree.nodes[cid] for cid in node.children]
            if node.id == tree.root_id:
                kids = [k for k in kids if not _FRONT_MATTER_RE.match(k.title)] or kids
            # Clip each child rather than the joined text, so late chapters still reach the rollup.
            clip = max(80, limit // len(kids))
            rollup = "\n".join(f"- {k.title}: {k.summary[:clip]}" for k in kids)
            prompt = _ROLLUP_PROMPT.format(title=node.title, children=rollup)
        else:
            texts = [chunk_by_id[cid].text for cid in node.chunk_ids if cid in chunk_by_id]
            body = _leaf_body(node.title, "\n\n".join(texts), limit)
            prompt = _LEAF_PROMPT.format(title=node.title, text=body) if body.strip() else None
        if prompt is None:
            node.summary = node.title
        else:
            # Keyed by content, not node position: a changed tree can't inherit another section's summary.
            cache_f = cache_dir / f"{_slug(tree.doc)}__{hashlib.sha1(prompt.encode()).hexdigest()[:16]}.txt"
            if cache_f.exists():
                node.summary = cache_f.read_text()
            else:
                try:
                    node.summary = generate(prompt, model=SETTINGS.pageindex_model,
                                            num_predict=160, temperature=0.2).strip()
                    cache_f.write_text(node.summary)
                except Exception:
                    node.summary = node.title
        if node_done:
            node_done(node.id)


_FRONT_MATTER_RE = re.compile(
    r"^(brief contents|table of contents|contents|preface|foreword|acknowledg|dedication|copyright"
    r"|about (this|the) (book|author)|front matter|bibliographic)",
    re.IGNORECASE,
)


# Start at the section's own heading: a section that begins mid-page otherwise gets
# summarized from the previous section's tail (observed: Case Study #6 summarized as #5).
def _leaf_body(title: str, text: str, limit: int) -> str:
    flat = " ".join(text.split())
    key = " ".join(title.split())[:40].lower()
    at = flat.lower().find(key) if key else -1
    return flat[max(at, 0):][:limit]


def build_document_tree(pdf_path: Path, index: BaseIndex,
                         progress: Optional[Callable[[str, float, str], None]] = None) -> tuple[PITree, dict]:
    doc_name = pdf_path.name
    doc_slug = _slug(doc_name)
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count

    entries = _toc_entries(pdf_path)
    used_toc = entries is not None
    if not used_toc:
        entries = _heuristic_entries(pdf_path)

    tree = _build_hierarchy(doc_name, doc_slug, entries, page_count)
    _assign_chunks(tree, index, doc_name)

    cache_dir = PAGEINDEX_DIR / "summaries" / _slug(SETTINGS.pageindex_model)
    total = len(tree.nodes)
    done = 0

    def node_done(_node_id: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress("pageindex", 0.0, f"{doc_name}: summarized {done}/{total} nodes")

    _summarize_tree(tree, index, doc_name, cache_dir, node_done)

    meta = {"doc": doc_name, "pages": page_count, "nodes": total, "source": "toc" if used_toc else "heuristic"}
    return tree, meta


def build_pageindex_trees(index: BaseIndex, progress: Optional[Callable[[str, float, str], None]] = None) -> dict:
    progress = progress or (lambda s, f, m: print(f"[pageindex {f * 100:.0f}%] {m}"))
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    TREES_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pdfs = list_pdfs()
    doc_summaries: dict[str, dict] = {}
    docs_toc = docs_heuristic = total_nodes = 0

    for i, pdf in enumerate(pdfs):
        progress("pageindex", i / max(len(pdfs), 1), f"Building tree for {pdf.name}…")
        try:
            tree, meta = build_document_tree(pdf, index, progress=progress)
        except Exception as e:  # keep going on a bad PDF, matching ingest.build_chunks()
            print(f"[pageindex] failed on {pdf.name}: {e}")
            continue
        (TREES_DIR / f"{_slug(pdf.name)}.json").write_text(json.dumps(tree.to_dict(), indent=2))
        root = tree.nodes[tree.root_id]
        sections = [tree.nodes[c].title for c in root.children if not _FRONT_MATTER_RE.match(tree.nodes[c].title)]
        doc_summaries[pdf.name] = {"title": root.title, "summary": root.summary, "sections": sections}
        total_nodes += len(tree.nodes)
        docs_toc += meta["source"] == "toc"
        docs_heuristic += meta["source"] == "heuristic"

    DOC_SUMMARY_FILE.write_text(json.dumps(doc_summaries, indent=2))
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "docs_processed": len(pdfs),
        "docs_with_real_toc": docs_toc,
        "docs_with_heuristic_fallback": docs_heuristic,
        "total_nodes": total_nodes,
        "build_seconds": round(time.time() - t0, 1),
        "model": SETTINGS.pageindex_model,
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    progress("pageindex", 1.0,
              f"PageIndex: {len(pdfs)} docs, {total_nodes} nodes ({docs_toc} TOC, {docs_heuristic} heuristic).")
    return meta


def pageindex_exists() -> bool:
    return TREES_DIR.exists() and any(TREES_DIR.glob("*.json"))


def get_pageindex_meta() -> Optional[dict]:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return None


def load_tree(doc_name: str) -> Optional[PITree]:
    f = TREES_DIR / f"{_slug(doc_name)}.json"
    if not f.exists():
        return None
    return PITree.from_dict(json.loads(f.read_text()))


def load_doc_summaries() -> dict:
    if DOC_SUMMARY_FILE.exists():
        return json.loads(DOC_SUMMARY_FILE.read_text())
    return {}

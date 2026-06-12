"""PDF ingestion: extract text page-by-page with PyMuPDF, then build
paragraph-aware dynamic chunks that respect structural boundaries (headings,
paragraph breaks, line breaks) rather than a fixed character window."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF

from .config import SETTINGS, DOCUMENTS_DIR


@dataclass
class Chunk:
    id: str
    doc: str          # file name
    text: str
    page_start: int   # 1-based
    page_end: int
    char_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> str:
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        return f"{self.doc}, {pages}"


_WS = re.compile(r"[ \t ]+")
_MULTINL = re.compile(r"\n{3,}")

# Structural boundary patterns for dynamic chunking
_PARA_SPLIT = re.compile(r"\n{2,}")
# Headings: numbered sections ("1. Intro") or lines that are ALL CAPS (≥5 chars)
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[\s.]+[A-Z]|[A-Z][A-Z\s]{4,})$")

# Hard max chars per chunk; oversized paragraphs are sentence-split
_MAX_CHUNK = 1600
# Skip near-empty paragraph fragments
_MIN_PARA = 30


def _clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    # de-hyphenate words broken across line ends: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number_1based, cleaned_text), ...]."""
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            txt = _clean(page.get_text("text"))
            if txt:
                pages.append((i + 1, txt))
    return pages


def _split_at_sentences(text: str) -> list[str]:
    """Recursively split text at sentence boundaries to stay under _MAX_CHUNK."""
    if len(text) <= _MAX_CHUNK:
        return [text]
    window = text[:_MAX_CHUNK]
    pos = max(window.rfind(". "), window.rfind(".\n"),
              window.rfind("! "), window.rfind("? "))
    if pos <= 0:
        pos = window.rfind(" ") or _MAX_CHUNK
    head = text[:pos + 1].strip()
    tail = text[pos + 1:].strip()
    result = [head] if head else []
    if tail:
        result.extend(_split_at_sentences(tail))
    return result or [text[:_MAX_CHUNK].strip()]


def _chunk_doc(doc_name: str, pages: list[tuple[int, str]]) -> list[Chunk]:
    """Dynamic chunking: split on paragraph/heading boundaries, greedily merge
    short paragraphs into chunks up to _MAX_CHUNK, sentence-split oversized ones."""
    min_chars = SETTINGS.min_chunk_chars

    # Collect (page_num, paragraph_text) pairs; insert boundary sentinels at headings
    para_list: list[tuple[int, str]] = []
    for pno, txt in pages:
        for raw in _PARA_SPLIT.split(txt):
            para = raw.strip()
            if len(para) < _MIN_PARA:
                continue
            first_line = para.split("\n")[0]
            if _HEADING.match(first_line):
                para_list.append((pno, "\x00"))  # flush boundary before heading
            para_list.append((pno, para))

    chunks: list[Chunk] = []
    idx = 0
    buf_parts: list[str] = []
    buf_pages: list[int] = []

    def _emit(text: str, p_start: int, p_end: int) -> None:
        nonlocal idx
        text = text.strip()
        if len(text) < min_chars:
            return
        cid = hashlib.sha1(f"{doc_name}:{idx}:{text[:64]}".encode()).hexdigest()[:16]
        chunks.append(Chunk(id=cid, doc=doc_name, text=text,
                            page_start=p_start, page_end=p_end, char_count=len(text)))
        idx += 1

    def _flush() -> None:
        if buf_parts:
            _emit("\n\n".join(buf_parts), buf_pages[0], buf_pages[-1])
            buf_parts.clear()
            buf_pages.clear()

    for pno, para in para_list:
        if para == "\x00":          # heading sentinel → start a new chunk
            _flush()
            continue
        if len(para) > _MAX_CHUNK:  # oversized paragraph → flush then sentence-split
            _flush()
            for sub in _split_at_sentences(para):
                _emit(sub, pno, pno)
            continue
        current_len = sum(len(p) for p in buf_parts) + 2 * max(0, len(buf_parts) - 1)
        if buf_parts and current_len + 2 + len(para) > _MAX_CHUNK:
            _flush()
        buf_parts.append(para)
        buf_pages.append(pno)

    _flush()
    return chunks


def list_pdfs(documents_dir: Path | None = None) -> list[Path]:
    documents_dir = documents_dir or DOCUMENTS_DIR
    return sorted(p for p in documents_dir.glob("*.pdf") if p.is_file())


def build_chunks(documents_dir: Path | None = None, progress=None) -> list[Chunk]:
    pdfs = list_pdfs(documents_dir)
    all_chunks: list[Chunk] = []
    for i, pdf in enumerate(pdfs):
        if progress:
            progress(i, len(pdfs), pdf.name)
        try:
            pages = extract_pages(pdf)
            all_chunks.extend(_chunk_doc(pdf.name, pages))
        except Exception as e:  # keep going on a bad PDF
            print(f"[ingest] failed on {pdf.name}: {e}")
    if progress:
        progress(len(pdfs), len(pdfs), "done")
    return all_chunks

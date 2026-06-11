"""PDF ingestion: extract text page-by-page with PyMuPDF, then build overlapping
character-window chunks that remember their source document and page span."""
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


_WS = re.compile(r"[ \t ]+")
_MULTINL = re.compile(r"\n{3,}")


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


def _chunk_doc(doc_name: str, pages: list[tuple[int, str]]) -> list[Chunk]:
    """Concatenate pages with markers, slide a character window over the whole
    document, and map each window back to the page span it covers."""
    size = SETTINGS.chunk_size
    overlap = SETTINGS.chunk_overlap
    # Build a flat string and a per-character page map.
    buf: list[str] = []
    page_map: list[int] = []
    for pno, txt in pages:
        if buf:
            buf.append("\n\n")
            page_map.extend([pno] * 2)
        buf.append(txt)
        page_map.extend([pno] * len(txt))
    flat = "".join(buf)
    n = len(flat)
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + size, n)
        # try to end on a sentence/paragraph boundary within the last 200 chars
        if end < n:
            window = flat[end - 200 : end]
            m = max(window.rfind(". "), window.rfind("\n"), window.rfind("! "), window.rfind("? "))
            if m != -1:
                end = end - 200 + m + 1
        seg = flat[start:end]
        if start > 0:  # drop a leading partial word created by the overlap
            sp = seg.find(" ")
            if 0 <= sp <= 40:
                seg = seg[sp + 1 :]
        text = seg.strip()
        if len(text) >= SETTINGS.min_chunk_chars:
            p_start = page_map[start] if start < len(page_map) else pages[-1][0]
            p_end = page_map[min(end, len(page_map)) - 1] if page_map else p_start
            cid = hashlib.sha1(f"{doc_name}:{idx}:{text[:64]}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=cid,
                    doc=doc_name,
                    text=text,
                    page_start=p_start,
                    page_end=max(p_start, p_end),
                    char_count=len(text),
                )
            )
            idx += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
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

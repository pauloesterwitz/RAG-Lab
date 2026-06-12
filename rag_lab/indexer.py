"""Orchestrates the (re-)embedding of the Documents folder into the shared
BaseIndex, and tracks a manifest the UI can display."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import INDEX_DIR, SETTINGS, DOCUMENTS_DIR
from .ingest import build_chunks, list_pdfs
from .ollama_client import embed_many
from .store import BaseIndex

BASE_DIR = INDEX_DIR / "base"
MANIFEST = INDEX_DIR / "manifest.json"

ProgressCB = Callable[[str, float, str], None]  # (stage, fraction 0..1, message)


def _noop(stage: str, frac: float, msg: str) -> None:
    print(f"[{stage} {frac*100:5.1f}%] {msg}")


def build_index(progress: Optional[ProgressCB] = None) -> dict:
    progress = progress or _noop
    t0 = time.time()

    progress("chunk", 0.0, "Reading PDFs…")
    chunks = build_chunks(
        progress=lambda i, n, name: progress("chunk", (i / max(n, 1)) * 0.15, f"Parsing {name}")
    )
    if not chunks:
        raise RuntimeError(f"No chunks produced — are there PDFs in {DOCUMENTS_DIR}?")

    progress("embed", 0.15, f"Embedding {len(chunks)} chunks with {SETTINGS.embed_model}…")

    def emb_progress(done: int, total: int) -> None:
        frac = 0.15 + (done / total) * 0.8
        progress("embed", frac, f"Embedded {done}/{total} chunks")

    vecs = embed_many([c.text for c in chunks], role="document", progress=emb_progress)
    emb = np.asarray(vecs, dtype=np.float32)

    progress("persist", 0.95, "Saving index…")
    index = BaseIndex(chunks, emb)
    index.save(BASE_DIR)

    per_doc = Counter(c.doc for c in chunks)
    manifest = {
        "embedded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embed_model": SETTINGS.embed_model,
        "embed_dim": int(emb.shape[1]),
        "num_docs": len(per_doc),
        "num_chunks": len(chunks),
        "chunk_size": SETTINGS.chunk_size,
        "chunk_overlap": SETTINGS.chunk_overlap,
        "per_doc": dict(sorted(per_doc.items())),
        "documents": [p.name for p in list_pdfs()],
        "build_seconds": round(time.time() - t0, 1),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    progress("done", 1.0, f"Indexed {len(chunks)} chunks from {len(per_doc)} docs.")
    return manifest


_BASE_CACHE: Optional[BaseIndex] = None


def load_base_index(refresh: bool = False) -> BaseIndex:
    global _BASE_CACHE
    if _BASE_CACHE is not None and not refresh:
        return _BASE_CACHE
    if not BaseIndex.exists(BASE_DIR):
        raise RuntimeError("No base index yet. Re-embed the Documents folder first.")
    idx = BaseIndex.load(BASE_DIR)
    stored_dim = idx.embeddings.shape[1]
    if stored_dim != SETTINGS.embed_dim:
        raise RuntimeError(
            f"Index dimension mismatch: stored={stored_dim}d but current embed_model "
            f"'{SETTINGS.embed_model}' produces {SETTINGS.embed_dim}d vectors. "
            "Re-embed the Documents folder first (CLI: python -m rag_lab.cli embed)."
        )
    _BASE_CACHE = idx
    return _BASE_CACHE


def get_manifest() -> Optional[dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return None


def invalidate_cache() -> None:
    global _BASE_CACHE
    _BASE_CACHE = None

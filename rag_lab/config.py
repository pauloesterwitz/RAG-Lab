"""Central configuration. Everything is overridable via environment variables so
the Vue UI / CLI / eval harness all read the same source of truth."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Project layout -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = Path(os.environ.get("RAG_DOCUMENTS_DIR", ROOT / "Documents"))
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", ROOT / "data"))
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = DATA_DIR / "eval"
CACHE_DIR = DATA_DIR / "cache"
for _d in (DATA_DIR, INDEX_DIR, EVAL_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Keep HuggingFace + DeepEval caches inside the project (the user's ~/.cache is
# root-owned on this box). Must be set before transformers/huggingface_hub import.
_HF_HOME = CACHE_DIR / "hf"
_HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", str(CACHE_DIR / "deepeval"))


@dataclass
class Settings:
    # --- Ollama ---
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    embed_model: str = os.environ.get("RAG_EMBED_MODEL", "embeddinggemma:latest")
    embed_dim: int = int(os.environ.get("RAG_EMBED_DIM", "768"))
    # gemma4:e4b is the one local model here that does grammar-constrained JSON
    # cleanly (gpt-oss returns empty under format=schema; gemma4:26b/31b crash on
    # load). It is also fast (~60 tok/s). Override via env for higher-quality judging.
    gen_model: str = os.environ.get("RAG_GEN_MODEL", "gemma4:e4b")
    judge_model: str = os.environ.get("RAG_JUDGE_MODEL", "gemma4:e4b")

    # --- Chunking ---
    chunk_size: int = int(os.environ.get("RAG_CHUNK_SIZE", "1100"))      # chars
    chunk_overlap: int = int(os.environ.get("RAG_CHUNK_OVERLAP", "180"))  # chars
    min_chunk_chars: int = int(os.environ.get("RAG_MIN_CHUNK_CHARS", "120"))

    # --- Retrieval ---
    top_k: int = int(os.environ.get("RAG_TOP_K", "5"))           # chunks fed to LLM
    candidate_k: int = int(os.environ.get("RAG_CANDIDATE_K", "20"))  # pre-rerank pool
    bm25_weight: float = float(os.environ.get("RAG_BM25_WEIGHT", "0.35"))  # hybrid

    # --- Reranker ---
    reranker_model: str = os.environ.get(
        "RAG_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
    )
    reranker_fallback_model: str = os.environ.get(
        "RAG_RERANKER_FALLBACK", "BAAI/bge-reranker-v2-m3"
    )

    # --- GraphRAG ---
    graph_max_chunks: int = int(os.environ.get("RAG_GRAPH_MAX_CHUNKS", "240"))
    graph_concurrency: int = int(os.environ.get("RAG_GRAPH_CONCURRENCY", "6"))

    # --- Concurrency ---
    embed_concurrency: int = int(os.environ.get("RAG_EMBED_CONCURRENCY", "6"))
    gen_concurrency: int = int(os.environ.get("RAG_GEN_CONCURRENCY", "4"))

    # --- Generation ---
    gen_temperature: float = float(os.environ.get("RAG_GEN_TEMPERATURE", "0.1"))
    gen_num_ctx: int = int(os.environ.get("RAG_GEN_NUM_CTX", "8192"))
    gen_num_predict: int = int(os.environ.get("RAG_GEN_NUM_PREDICT", "768"))

    # --- Eval ---
    eval_num_goldens: int = int(os.environ.get("RAG_EVAL_NUM_GOLDENS", "24"))

    def to_dict(self) -> dict:
        return asdict(self)


SETTINGS = Settings()

# Approach registry ----------------------------------------------------------
APPROACHES = {
    "plain": {
        "label": "Plain RAG",
        "description": "Hybrid dense (embeddinggemma) + BM25 retrieval, top-k to LLM.",
    },
    "rerank": {
        "label": "RAG + Reranker",
        "description": "Plain retrieval over a wider pool, re-ranked by a Jina cross-encoder.",
    },
    "hyde": {
        "label": "HyDE",
        "description": "Generate a hypothetical answer, embed it, retrieve on that vector.",
    },
    "corrective": {
        "label": "Corrective RAG (CRAG)",
        "description": "Grade retrieved docs; if weak, rewrite query & decompose, then re-retrieve.",
    },
    "agentic": {
        "label": "Agentic RAG",
        "description": "An LLM agent plans sub-queries, retrieves iteratively, and reflects.",
    },
    "graph": {
        "label": "GraphRAG",
        "description": "LLM entity/relation graph + community summaries; entity-anchored retrieval.",
    },
}

APPROACH_ORDER = ["plain", "rerank", "hyde", "corrective", "agentic", "graph"]

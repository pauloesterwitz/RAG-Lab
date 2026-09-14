"""Central configuration. Everything is overridable via environment variables so
the Vue UI / CLI / eval harness all read the same source of truth.

A .env file in the project root is loaded automatically if it exists, so you
can put ANTHROPIC_API_KEY there without polluting your shell profile."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Load .env before reading os.environ below
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

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
    # --- Provider: "claude" (default) or "llamaswap" --- controls GENERATION only.
    # Claude: Anthropic API (ANTHROPIC_API_KEY required).
    # llamaswap: local models served via llama-swap, e.g.
    #   RAG_PROVIDER=llamaswap RAG_GEN_MODEL=qwen38fn-sglang-tp2-starfleet
    # Embeddings are provider-independent — see below.
    provider: str = os.environ.get("RAG_PROVIDER", "claude")

    # --- Local provider (only used when provider=llamaswap) ---
    # There is no Ollama daemon anymore (retired on this box); every local model,
    # including the reasoning ones, is served via llama-swap (:28080), which only
    # routes OpenAI/Anthropic-compatible /v1/* paths.
    llamaswap_host: str = os.environ.get("LLAMASWAP_HOST", "http://localhost:28080")
    # Generation wire format for the local provider:
    #   "openai"    -> POST /v1/chat/completions  (most llama-swap members)
    #   "anthropic" -> POST /v1/messages          (ds4 / deepseek-v4-flash only)
    local_api_style: str = os.environ.get("RAG_LOCAL_API", "openai")

    # --- Embeddings ---
    # ALWAYS sentence-transformers (CPU, no server) regardless of RAG_PROVIDER — see
    # llamaswap_client.py's embed_one/embed_many. llama-swap's GPU-hosted embedding
    # members (embeddinggemma, nomic-embed-text) compete with pinned Starfleet models
    # for GPU memory and can CUDA-OOM under normal pool pressure; sentence-transformers
    # has no such contention. Default: intfloat/multilingual-e5-small (384-d).
    embed_model: str = os.environ.get("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
    embed_dim: int = int(os.environ.get("RAG_EMBED_DIM", "384"))

    # --- Generation / judging ---
    # Per user: generation, the DeepEval judge, and GraphRAG extraction all run on
    # Sonnet 4.6. A single shared org-wide rate limiter (claude_org_rpm) keeps every
    # Claude call (any model) under the org's ~5 RPM cap so the eval never 429s.
    gen_model: str = os.environ.get("RAG_GEN_MODEL", "claude-sonnet-4-6")
    judge_model: str = os.environ.get("RAG_JUDGE_MODEL", "claude-sonnet-4-6")
    graph_model: str = os.environ.get("RAG_GRAPH_MODEL", "claude-sonnet-4-6")
    claude_org_rpm: float = float(os.environ.get("RAG_CLAUDE_ORG_RPM", "4.0"))

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
    graph_max_chunks: int = int(os.environ.get("RAG_GRAPH_MAX_CHUNKS", "0"))  # 0 = all chunks (full coverage)
    graph_concurrency: int = int(os.environ.get("RAG_GRAPH_CONCURRENCY", "6"))

    # --- PageIndex ---
    pageindex_model: str = os.environ.get("RAG_PAGEINDEX_MODEL", "claude-sonnet-4-6")
    pageindex_max_depth: int = int(os.environ.get("RAG_PAGEINDEX_MAX_DEPTH", "4"))
    pageindex_max_breadth: int = int(os.environ.get("RAG_PAGEINDEX_MAX_BREADTH", "2"))  # children descended per hop
    pageindex_max_docs: int = int(os.environ.get("RAG_PAGEINDEX_MAX_DOCS", "3"))        # root-level fan-out cap
    pageindex_summary_chars: int = int(os.environ.get("RAG_PAGEINDEX_SUMMARY_CHARS", "1200"))

    # --- Concurrency ---
    embed_concurrency: int = int(os.environ.get("RAG_EMBED_CONCURRENCY", "6"))
    gen_concurrency: int = int(os.environ.get("RAG_GEN_CONCURRENCY", "4"))

    # --- Generation ---
    gen_temperature: float = float(os.environ.get("RAG_GEN_TEMPERATURE", "0.1"))
    gen_num_ctx: int = int(os.environ.get("RAG_GEN_NUM_CTX", "8192"))
    gen_num_predict: int = int(os.environ.get("RAG_GEN_NUM_PREDICT", "768"))

    # --- Eval ---
    eval_num_goldens: int = int(os.environ.get("RAG_EVAL_NUM_GOLDENS", "100"))

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
    "pageindex": {
        "label": "PageIndex",
        "description": "Hierarchical tree index (ToC/heading-derived); LLM navigates the tree structurally instead of vector similarity.",
    },
}

APPROACH_ORDER = ["plain", "rerank", "hyde", "corrective", "agentic", "graph", "pageindex"]

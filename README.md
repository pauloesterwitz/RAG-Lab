# RAG Lab — comparing RAG approaches with DeepEval

A self-contained lab that builds **six RAG pipelines** over the PDFs in
`Documents/`, evaluates them with **DeepEval** (judged by a local Ollama model),
and ships a **Vue** web app to re-embed, chat with any approach (with the
retrieved chunks always quoted), and compare the metrics on a dashboard.

Everything runs **locally** on Ollama — no cloud keys required.

## Approaches

| key | name | idea |
|-----|------|------|
| `plain` | Plain RAG | hybrid dense (`embeddinggemma`) + BM25, top-k → LLM |
| `rerank` | RAG + Reranker | wide candidate pool re-ordered by a **Jina reranker v2** cross-encoder (BGE fallback) |
| `hyde` | HyDE | draft a hypothetical answer, embed it, retrieve on that vector |
| `corrective` | Corrective RAG (CRAG) | grade retrieved docs; if weak, rewrite/decompose & re-retrieve |
| `agentic` | Agentic RAG | an LLM agent plans sub-queries, retrieves iteratively, reflects |
| `graph` | GraphRAG | LLM entity/relation graph + community summaries; entity-anchored retrieval |

## Metrics (DeepEval)

Per approach, over synthesized **goldens** (gold chunks → question + reference answer):

- Answer Relevancy
- Faithfulness
- Contextual Relevancy
- Contextual Precision
- Contextual Recall
- **G-Eval** "Correctness" (custom criteria)
- Composite = mean of the above

Plus retrieval-level metrics we add on top: **gold-chunk hit rate**,
**gold-doc hit rate**, average latency, average #chunks / context size.

## Stack

- **Embeddings:** Ollama `embeddinggemma:latest` (768-d, query/doc prompt templates)
- **Generation / judging:** a local Ollama model (configurable, see `rag_lab/config.py`)
- **Reranker:** `jinaai/jina-reranker-v2-base-multilingual` (HuggingFace), BGE + LLM fallbacks
- **Vector store:** in-process NumPy cosine + `rank-bm25` hybrid
- **Graph:** `networkx` (Louvain communities)
- **Eval:** `deepeval` with a custom Ollama judge + embedder
- **Backend:** FastAPI · **Frontend:** Vue 3 + Vite + Chart.js

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # + einops for the Jina reranker
cd web && npm install && npm run build && cd ..
```

(The venv already exists in this checkout, with the project on its path.)

## Run

```bash
# 1) start the API (serves the built web app at http://localhost:8000)
.venv/bin/python -m uvicorn rag_lab.api.server:app --host 0.0.0.0 --port 8000

# 2) (optional) hot-reload frontend dev server on :5173, proxying /api -> :8000
cd web && npm run dev
```

Open **http://localhost:8000** (or `:5173` in dev).

### CLI (same orchestration as the UI)

```bash
.venv/bin/python -m rag_lab.cli embed          # (re)embed Documents -> base index
.venv/bin/python -m rag_lab.cli graph          # build the GraphRAG graph
.venv/bin/python -m rag_lab.cli synth -n 16     # synthesize goldens
.venv/bin/python -m rag_lab.cli eval            # DeepEval across all approaches
.venv/bin/python -m rag_lab.cli all -n 16       # everything
.venv/bin/python -m rag_lab.cli status
```

## Web app

- **Index & Eval** — re-embed the Documents folder (with live progress), build the
  graph, synthesize goldens, run DeepEval.
- **Chat / Explore** — pick an approach up front, ask questions; the answer is shown
  with the **retrieved chunks quoted verbatim** + their citation, score and the
  retrieval trace (sub-queries, grading, graph anchors…).
- **DeepEval Dashboard** — per-approach metric table (best per column highlighted),
  radar + bar charts, retrieval-level metrics, and a per-question drill-down with
  metric reasons.
- **Goldens** — the synthesized evaluation set with its gold chunks.

## Configuration

All knobs live in `rag_lab/config.py` and are environment-overridable, e.g.:

```bash
export RAG_GEN_MODEL=qwen3.6:35b-a3b-q8_0
export RAG_JUDGE_MODEL=qwen3.6:35b-a3b-q8_0
export RAG_EVAL_NUM_GOLDENS=16
export RAG_TOP_K=5
```

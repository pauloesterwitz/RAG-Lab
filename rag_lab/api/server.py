"""FastAPI backend for the RAG lab.

Endpoints
  GET  /api/health
  GET  /api/approaches          list approaches
  GET  /api/status              what's built (index, graph, goldens, results)
  POST /api/reembed             (re)embed Documents (+ optional graph) -> job
  POST /api/synth               synthesize goldens -> job
  POST /api/eval                run DeepEval over approaches -> job
  POST /api/full                embed+graph+synth+eval -> job
  GET  /api/job                 current job status (progress + log)
  POST /api/chat                run one approach on a query (quoted chunks back)
  GET  /api/metrics             DeepEval results for the dashboard
  GET  /api/goldens             synthesized goldens
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import SETTINGS, APPROACHES, APPROACH_ORDER, DOCUMENTS_DIR, ROOT
from ..indexer import build_index, load_base_index, get_manifest, invalidate_cache
from ..graph_build import build_graph, get_graph_meta, graph_exists
from ..ingest import list_pdfs
from ..ollama_client import list_models
from ..reranker import backend_name
from ..approaches import get_approach
from ..eval.synthesize import synthesize_goldens, load_goldens
from ..eval.run_eval import run_full_eval, load_results
from .jobs import MANAGER, Job

app = FastAPI(title="RAG Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- models ---------------------------------------------------------------
class ReembedReq(BaseModel):
    rebuild_graph: bool = True


class SynthReq(BaseModel):
    num: Optional[int] = None


class EvalReq(BaseModel):
    approaches: Optional[list[str]] = None


class ChatReq(BaseModel):
    approach: str
    query: str
    generate_answer: bool = True


# --- meta -----------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/approaches")
def approaches():
    return {
        "order": APPROACH_ORDER,
        "approaches": {k: APPROACHES[k] for k in APPROACH_ORDER},
    }


@app.get("/api/status")
def status():
    manifest = get_manifest()
    return {
        "documents_dir": str(DOCUMENTS_DIR),
        "documents": [p.name for p in list_pdfs()],
        "index": manifest,
        "graph": get_graph_meta(),
        "graph_built": graph_exists(),
        "goldens": (load_goldens() or {}).get("num_goldens", 0),
        "has_results": load_results() is not None,
        "models": {
            "embed": SETTINGS.embed_model,
            "gen": SETTINGS.gen_model,
            "judge": SETTINGS.judge_model,
            "reranker": backend_name(),
            "available": list_models(),
        },
        "settings": SETTINGS.to_dict(),
        "job": MANAGER.status(),
    }


# --- jobs -----------------------------------------------------------------
def _frac_progress(job: Job):
    def cb(stage: str, frac: float, msg: str):
        job.stage = stage
        job.progress = frac
        job.log.append(f"[{stage} {frac*100:.0f}%] {msg}")
    return cb


def _msg_progress(job: Job):
    def cb(msg: str):
        job.log.append(msg)
    return cb


@app.post("/api/reembed")
def reembed(req: ReembedReq):
    def task(job: Job):
        manifest = build_index(progress=_frac_progress(job))
        invalidate_cache()
        out = {"manifest": manifest}
        if req.rebuild_graph:
            job.log.append("Building knowledge graph for GraphRAG…")
            idx = load_base_index(refresh=True)
            out["graph"] = build_graph(idx, progress=_frac_progress(job))
        return out

    try:
        job = MANAGER.start("reembed", task)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return job.to_dict()


@app.post("/api/synth")
def synth(req: SynthReq):
    def task(job: Job):
        job.progress = -1.0
        return synthesize_goldens(num=req.num, progress=_msg_progress(job))

    try:
        job = MANAGER.start("synth", task)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return job.to_dict()


@app.post("/api/eval")
def run_eval(req: EvalReq):
    def task(job: Job):
        job.progress = -1.0
        return run_full_eval(approaches=req.approaches, progress=_msg_progress(job))

    try:
        job = MANAGER.start("eval", task)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return job.to_dict()


@app.post("/api/full")
def full(req: SynthReq):
    def task(job: Job):
        build_index(progress=_frac_progress(job))
        invalidate_cache()
        idx = load_base_index(refresh=True)
        build_graph(idx, progress=_frac_progress(job))
        job.progress = -1.0
        synthesize_goldens(num=req.num, progress=_msg_progress(job))
        return run_full_eval(progress=_msg_progress(job))

    try:
        job = MANAGER.start("full", task)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return job.to_dict()


@app.get("/api/job")
def job():
    s = MANAGER.status()
    if not s:
        return {"status": "idle"}
    return s


# --- chat -----------------------------------------------------------------
@app.post("/api/chat")
def chat(req: ChatReq):
    if req.approach not in APPROACHES:
        raise HTTPException(400, f"Unknown approach '{req.approach}'.")
    try:
        index = load_base_index()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    approach = get_approach(req.approach, index)
    result = approach.run(req.query, generate_answer=req.generate_answer)
    return result.to_dict()


# --- metrics --------------------------------------------------------------
@app.get("/api/metrics")
def metrics():
    r = load_results()
    if not r:
        return {"has_results": False}
    return {"has_results": True, **r}


@app.get("/api/goldens")
def goldens():
    return load_goldens() or {"num_goldens": 0, "goldens": []}


# --- serve built frontend (production) ------------------------------------
_DIST = ROOT / "web" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")

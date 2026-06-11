"""Thin Ollama HTTP client: embeddings + generation, with a thread-pool helper
for concurrent calls. EmbeddingGemma works best with task-specific prompt
prefixes, so we apply them based on whether we embed a query or a document."""
from __future__ import annotations

import concurrent.futures as cf
import time
from typing import Iterable, Optional

import httpx

from .config import SETTINGS

_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
_RETRIES = 3


def _post_with_retry(path: str, payload: dict) -> dict:
    """POST to Ollama with retries — this box is shared, so calls can time out
    or 5xx under memory pressure from other workloads."""
    last = None
    for attempt in range(_RETRIES):
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                r = client.post(f"{SETTINGS.ollama_host}{path}", json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as e:  # noqa
            last = e
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Ollama call to {path} failed after {_RETRIES} attempts: {last}")


# --- EmbeddingGemma prompt templates (improve retrieval quality) ------------
def _embed_prompt(text: str, role: str) -> str:
    text = text.replace("\n", " ").strip()
    if "embeddinggemma" in SETTINGS.embed_model:
        if role == "query":
            return f"task: search result | query: {text}"
        return f"title: none | text: {text}"
    return text


def embed_one(text: str, role: str = "document", *, model: Optional[str] = None) -> list[float]:
    model = model or SETTINGS.embed_model
    data = _post_with_retry("/api/embed", {"model": model, "input": _embed_prompt(text, role)})
    return data["embeddings"][0]


def embed_many(
    texts: list[str],
    role: str = "document",
    *,
    model: Optional[str] = None,
    concurrency: Optional[int] = None,
    progress=None,
) -> list[list[float]]:
    """Embed many texts concurrently. `progress(done, total)` is called as work completes."""
    model = model or SETTINGS.embed_model
    concurrency = concurrency or SETTINGS.embed_concurrency
    results: list[Optional[list[float]]] = [None] * len(texts)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(embed_one, t, role, model=model): i for i, t in enumerate(texts)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if progress and (done % 5 == 0 or done == len(texts)):
                progress(done, len(texts))
    return [r for r in results]  # type: ignore[return-value]


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
    fmt: Optional[dict | str] = None,
    think: Optional[bool] = None,
) -> str:
    """Single-turn generation. `fmt` may be 'json' or a JSON schema dict."""
    model = model or SETTINGS.gen_model
    options = {
        "temperature": SETTINGS.gen_temperature if temperature is None else temperature,
        "num_predict": num_predict or SETTINGS.gen_num_predict,
        "num_ctx": num_ctx or SETTINGS.gen_num_ctx,
    }
    payload: dict = {"model": model, "prompt": prompt, "stream": False, "options": options}
    if system:
        payload["system"] = system
    if fmt is not None:
        payload["format"] = fmt
    if think is not None:
        payload["think"] = think
    return _post_with_retry("/api/generate", payload).get("response", "")


def generate_many(prompts: Iterable[str], *, concurrency: Optional[int] = None, **kw) -> list[str]:
    prompts = list(prompts)
    concurrency = concurrency or SETTINGS.gen_concurrency
    out: list[Optional[str]] = [None] * len(prompts)
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(generate, p, **kw): i for i, p in enumerate(prompts)}
        for fut in cf.as_completed(futs):
            out[futs[fut]] = fut.result()
    return [o or "" for o in out]


def list_models() -> list[str]:
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            r = client.get(f"{SETTINGS.ollama_host}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []

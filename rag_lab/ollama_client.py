"""Thin Ollama HTTP client: embeddings + generation, with a thread-pool helper
for concurrent calls. EmbeddingGemma works best with task-specific prompt
prefixes, so we apply them based on whether we embed a query or a document.

All requests target SETTINGS.ollama_host (default: llama-swap at :28080).
llama-swap transparently proxies Ollama-native API paths (/api/generate,
/api/embed) to the Ollama daemon and also serves deepseek-v4-flash via the
OpenAI-compatible /v1/chat/completions path in an exclusive swap group."""
from __future__ import annotations

import concurrent.futures as cf
import time
from typing import Iterable, Optional

import httpx

from .config import SETTINGS

_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
_RETRIES = 3

# Models that require the OpenAI-compatible /v1/chat/completions path.
# These are served by ds4 (DwarfStar) via llama-swap, not the Ollama daemon.
_OPENAI_COMPAT_MODELS = frozenset({"deepseek-v4-flash"})


def _post_with_retry(path: str, payload: dict) -> dict:
    """POST to llama-swap/Ollama with retries — this box is shared, so calls
    can time out or 5xx under memory pressure from other workloads."""
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


def _generate_openai_compat(
    prompt: str,
    *,
    model: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    think: Optional[bool] = None,
) -> str:
    """OpenAI-compatible generation for ds4-served models (e.g. deepseek-v4-flash).
    Routes to /v1/chat/completions via llama-swap, which swaps ds4 in and Ollama out."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    # DeepSeek reasoning modes: /no_think prefix suppresses chain-of-thought tokens.
    user_content = ("/no_think\n" + prompt) if think is False else prompt
    messages.append({"role": "user", "content": user_content})
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": SETTINGS.gen_temperature if temperature is None else temperature,
        "max_tokens": num_predict or SETTINGS.gen_num_predict,
        "stream": False,
    }
    data = _post_with_retry("/v1/chat/completions", payload)
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


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

    # deepseek-v4-flash and other ds4-served models use the OpenAI-compat path.
    if model in _OPENAI_COMPAT_MODELS:
        return _generate_openai_compat(
            prompt, model=model, system=system, temperature=temperature,
            num_predict=num_predict, think=think,
        )

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
    # Reasoning models (qwen3, gpt-oss) must have thinking OFF for grammar-
    # constrained JSON to emit (and to avoid burning tokens on reasoning).
    if think is None and ("qwen3" in model or "gpt-oss" in model):
        think = False
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
    models: list[str] = []
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            # Ollama models via llama-swap proxy
            r = client.get(f"{SETTINGS.ollama_host}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            # Also surface ds4 models (deepseek-v4-flash) if llama-swap exposes /v1/models
            try:
                r2 = client.get(f"{SETTINGS.ollama_host}/v1/models")
                if r2.status_code == 200:
                    ds4 = [m["id"] for m in r2.json().get("data", [])
                           if m["id"] not in models]
                    models = models + ds4
            except Exception:
                pass
    except Exception:
        pass
    return models


# ---------------------------------------------------------------------------
# Provider dispatch: when RAG_PROVIDER=claude, replace this module's public
# symbols with the Claude implementations so no import sites need changing.
# ---------------------------------------------------------------------------
if SETTINGS.provider == "claude":
    from .claude_client import (  # noqa: F401, E402
        embed_one,
        embed_many,
        generate,
        generate_many,
        list_models,
    )

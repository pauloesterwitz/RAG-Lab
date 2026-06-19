"""Local-provider HTTP client: embeddings + generation against an OpenAI/Anthropic-
compatible endpoint, with a thread-pool helper for concurrent calls.

Targets SETTINGS.ollama_host (default: llama-swap at :28080). llama-swap only
routes /v1/* paths — it 404s on Ollama-native /api/* — so this client speaks:

  * embeddings  -> POST /v1/embeddings              (OpenAI-style; always)
  * generation  -> POST /v1/chat/completions        (SETTINGS.local_api_style="openai")
                or POST /v1/messages                 (SETTINGS.local_api_style="anthropic")

The /v1/* paths also work against a direct Ollama daemon (:11434), so nothing
here is llama-swap-specific. EmbeddingGemma works best with task-specific prompt
prefixes, so we apply them based on whether we embed a query or a document."""
from __future__ import annotations

import concurrent.futures as cf
import re
import time
from typing import Iterable, Optional

import httpx

from .config import SETTINGS

_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
_RETRIES = 3

# Over the /v1 wire, neither response_format nor Ollama's native `format` field
# reliably constrains output — models wrap JSON in ``` fences or prose. Callers
# do raw json.loads(), so we clean the reply here.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Best-effort: pull a JSON object/array out of a fenced or prose-wrapped reply."""
    if not text:
        return text
    s = text.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    if not (s.startswith("{") or s.startswith("[")):
        starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
        end = max(s.rfind("}"), s.rfind("]"))
        if starts and end > min(starts):
            s = s[min(starts):end + 1]
    return s.strip()


def _post_with_retry(path: str, payload: dict) -> dict:
    """POST to the local endpoint with retries — this box is shared, so calls can
    time out or 5xx under memory pressure (or while llama-swap cold-loads a model)."""
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
    raise RuntimeError(f"Local call to {path} failed after {_RETRIES} attempts: {last}")


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
    data = _post_with_retry("/v1/embeddings", {"model": model, "input": _embed_prompt(text, role)})
    return data["data"][0]["embedding"]


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


# ---------------------------------------------------------------------------
# Generation — two wire formats, selected by SETTINGS.local_api_style
# ---------------------------------------------------------------------------
def _generate_openai(
    prompt: str, *, model: str, system: Optional[str], temperature: float,
    max_tokens: int, fmt: Optional[dict | str], no_think: bool,
) -> str:
    """POST /v1/chat/completions. Spoken by the Ollama daemon AND ds4."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": False,
    }
    # Disable chain-of-thought on reasoning models (qwen3, gpt-oss): otherwise
    # the reasoning trace lands in message.reasoning and eats the whole token
    # budget, leaving message.content empty. Ollama maps reasoning_effort="none"
    # to think=off on its /v1 endpoint (a literal /no_think or think:false are
    # both ignored there).
    if no_think:
        payload["reasoning_effort"] = "none"
    if fmt is not None:  # json_object is the only hint Ollama recognises on /v1
        payload["response_format"] = {"type": "json_object"}
    data = _post_with_retry("/v1/chat/completions", payload)
    choices = data.get("choices") or [{}]
    msg = choices[0].get("message", {}) or {}
    # Fall back to the reasoning channel if a model emitted only that.
    out = msg.get("content") or msg.get("reasoning") or ""
    return _extract_json(out) if fmt is not None else out


def _generate_anthropic(
    prompt: str, *, model: str, system: Optional[str], temperature: float,
    max_tokens: int, fmt: Optional[dict | str], no_think: bool,
) -> str:
    """POST /v1/messages. Spoken by ds4 (deepseek-v4-flash); NOT the Ollama daemon."""
    sys_parts = [system] if system else []
    if fmt is not None:  # no response_format in the Messages API — steer via system
        sys_parts.append("Respond with valid JSON only, with no prose or code fences.")
    payload: dict = {
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if sys_parts:
        payload["system"] = "\n\n".join(sys_parts)
    # Messages API: extended thinking is off unless explicitly enabled, so
    # no_think needs no extra field. Enable it when reasoning is wanted.
    if not no_think:
        payload["thinking"] = {"type": "enabled", "budget_tokens": 2048}
    data = _post_with_retry("/v1/messages", payload)
    blocks = data.get("content", []) or []
    out = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return _extract_json(out) if fmt is not None else out


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,  # accepted for call-site compat; not exposed over /v1/*
    fmt: Optional[dict | str] = None,
    think: Optional[bool] = None,
) -> str:
    """Single-turn generation. `fmt` may be 'json' or a JSON schema dict.

    Routes to /v1/chat/completions (local_api_style="openai", default) or
    /v1/messages (local_api_style="anthropic")."""
    model = model or SETTINGS.gen_model
    temp = SETTINGS.gen_temperature if temperature is None else temperature
    max_tokens = num_predict or SETTINGS.gen_num_predict
    # Reasoning models (qwen3, gpt-oss) must have thinking OFF for constrained
    # JSON to emit (and to avoid burning tokens on reasoning).
    if think is None and ("qwen3" in model or "gpt-oss" in model):
        think = False
    no_think = think is False

    impl = _generate_anthropic if SETTINGS.local_api_style == "anthropic" else _generate_openai
    return impl(prompt, model=model, system=system, temperature=temp,
                max_tokens=max_tokens, fmt=fmt, no_think=no_think)


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
    """Models exposed by the endpoint. Over llama-swap this is the group list
    (e.g. deepseek-v4-flash, ollama); the individual Ollama aliases are usable
    by name even though /v1/models doesn't enumerate them."""
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            r = client.get(f"{SETTINGS.ollama_host}/v1/models")
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Provider dispatch: when RAG_PROVIDER=claude, replace this module's public
# symbols with the Claude implementations (Anthropic SDK -> api.anthropic.com)
# so no import sites need changing.
# ---------------------------------------------------------------------------
if SETTINGS.provider == "claude":
    from .claude_client import (  # noqa: F401, E402
        embed_one,
        embed_many,
        generate,
        generate_many,
        list_models,
    )

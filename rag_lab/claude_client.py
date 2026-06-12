"""Claude API client: generation via Anthropic SDK + embeddings via sentence-transformers.

Drop-in replacement for ollama_client when RAG_PROVIDER=claude (the default).
The generate() signature accepts all Ollama-compat kwargs (num_ctx, think, etc.)
and silently ignores the ones that don't map to Claude.

Per-model token-bucket rate limiters keep us under the org's RPM ceilings
(Sonnet 4.6 = 5 RPM on this org; Haiku = much higher).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import time
from typing import Iterable, Optional

import anthropic

from .config import SETTINGS

_client: Optional[anthropic.Anthropic] = None
_st_model = None  # sentence-transformers SentenceTransformer, lazy-loaded

# ---------------------------------------------------------------------------
# Per-model token-bucket rate limiter
# ---------------------------------------------------------------------------
# Conservative limits: Sonnet = 4 RPM (hard limit is 5), Haiku = 50 RPM.
# Org-wide limit is ~5 RPM across all models; use 4 RPM to stay safe.
_MODEL_RPM: dict[str, float] = {
    "claude-sonnet-4-6": 4.0,
    "claude-haiku-4-5-20251001": 4.0,
}
_DEFAULT_RPM = 4.0


class _TokenBucket:
    """Thread-safe token bucket. One token = one API call."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm  # seconds between tokens
        self._tokens = min(rpm, 5.0)  # start with a small burst
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._tokens + (now - self._last) / self._interval,
                5.0,  # cap burst
            )
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) * self._interval
                time.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


_limiters: dict[str, _TokenBucket] = {}
_limiters_lock = threading.Lock()


def _get_limiter(model: str) -> _TokenBucket:
    with _limiters_lock:
        if model not in _limiters:
            rpm = _MODEL_RPM.get(model, _DEFAULT_RPM)
            _limiters[model] = _TokenBucket(rpm)
        return _limiters[model]


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        import os
        # Support both API key (console.anthropic.com) and OAuth bearer token
        # (Claude Code session: export ANTHROPIC_AUTH_TOKEN=<token>)
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if auth_token:
            _client = anthropic.Anthropic(auth_token=auth_token)
        else:
            _client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY
    return _client


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(SETTINGS.embed_model)
    return _st_model


def _claude_model(model: Optional[str]) -> str:
    """Return a valid Claude model ID. If a non-Claude name is passed (e.g. an
    Ollama model name left over from a call site), fall back to the configured one."""
    m = model or SETTINGS.gen_model
    return m if m.startswith("claude-") else SETTINGS.gen_model


# ---------------------------------------------------------------------------
# Embeddings (sentence-transformers, no server required)
# ---------------------------------------------------------------------------

def embed_one(text: str, role: str = "document", *, model: Optional[str] = None) -> list[float]:
    m = _get_st_model()
    mod = model or SETTINGS.embed_model
    # multilingual-e5 and similar models benefit from task prefixes
    if "e5" in mod:
        text = ("query: " if role == "query" else "passage: ") + text
    vec = m.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_many(
    texts: list[str],
    role: str = "document",
    *,
    model: Optional[str] = None,
    concurrency: Optional[int] = None,
    progress=None,
) -> list[list[float]]:
    m = _get_st_model()
    mod = model or SETTINGS.embed_model
    if "e5" in mod:
        prefix = "query: " if role == "query" else "passage: "
        texts = [prefix + t for t in texts]
    vecs = m.encode(list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    if progress:
        progress(len(texts), len(texts))
    return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
# Generation (Anthropic API)
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    fmt: Optional[dict | str] = None,
    # Ollama-compat kwargs — accepted but unused by Claude
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
    think: Optional[bool] = None,
) -> str:
    """Single-turn generation.

    `fmt` may be a JSON schema dict or the string 'json'.  When a schema dict
    is given, tool_use forces Claude to emit schema-conformant JSON; callers
    can safely do json.loads() on the returned string.
    """
    mdl = _claude_model(model)
    temp = SETTINGS.gen_temperature if temperature is None else temperature
    tok = max_tokens or num_predict or SETTINGS.gen_num_predict

    _get_limiter(mdl).acquire()  # respect per-model RPM ceiling

    if fmt is not None and isinstance(fmt, dict):
        # Use tool_use to get guaranteed schema-conformant JSON
        tool = {
            "name": "output",
            "description": "Return the structured result",
            "input_schema": fmt,
        }
        resp = _get_client().messages.create(
            model=mdl,
            max_tokens=max(tok, 512),
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
            **({"system": system} if system else {}),
            tools=[tool],
            tool_choice={"type": "tool", "name": "output"},
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "output":
                return json.dumps(block.input)
        return "{}"

    if fmt == "json":
        sys_parts = [system] if system else []
        sys_parts.append("Respond with valid JSON only.")
        resp = _get_client().messages.create(
            model=mdl,
            max_tokens=max(tok, 512),
            temperature=temp,
            system="\n\n".join(sys_parts),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else "{}"

    # plain text
    kw: dict = dict(
        model=mdl,
        max_tokens=tok,
        temperature=temp,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kw["system"] = system
    resp = _get_client().messages.create(**kw)
    return resp.content[0].text if resp.content else ""


def generate_many(
    prompts: Iterable[str], *, concurrency: Optional[int] = None, **kw
) -> list[str]:
    prompts = list(prompts)
    concurrency = concurrency or SETTINGS.gen_concurrency
    out: list[Optional[str]] = [None] * len(prompts)
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(generate, p, **kw): i for i, p in enumerate(prompts)}
        for fut in cf.as_completed(futs):
            out[futs[fut]] = fut.result()
    return [o or "" for o in out]


def list_models() -> list[str]:
    return [SETTINGS.gen_model]

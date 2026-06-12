"""DeepEval judge + embedder implementations for both the Claude API and Ollama.

Use get_judge() / get_embedder() to get the right implementation for the
current RAG_PROVIDER setting instead of instantiating the classes directly."""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import re
from typing import Optional, Union

from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel

from ..config import SETTINGS
from ..ollama_client import generate as ollama_generate, embed_one, embed_many

_JSON_RE = re.compile(r"\{.*\}", re.S)

# Single shared, bounded executor: caps how many Ollama calls hit the local
# server at once no matter how many async eval tasks DeepEval fans out.
_EXEC = cf.ThreadPoolExecutor(max_workers=max(2, SETTINGS.gen_concurrency))


async def _run(fn, *a):
    return await asyncio.get_event_loop().run_in_executor(_EXEC, fn, *a)


def _coerce(raw: str, schema: type[BaseModel]) -> BaseModel:
    try:
        return schema.model_validate_json(raw)
    except Exception:
        m = _JSON_RE.search(raw)
        if m:
            return schema.model_validate_json(m.group(0))
        raise


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model: Optional[str] = None, num_ctx: Optional[int] = None):
        self.model_name = model or SETTINGS.judge_model
        # Match the generation num_ctx exactly: a single context size means Ollama
        # keeps ONE runner instance for the model instead of reload-thrashing
        # between sizes (critical for the 40GB qwen3.6).
        self.num_ctx = num_ctx or SETTINGS.gen_num_ctx

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return f"ollama:{self.model_name}"

    def generate(
        self, prompt: str, schema: Optional[type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        # NOTE: custom DeepEvalBaseLLM models must return the BARE result
        # (schema instance or str) — the (result, cost) tuple is only for
        # native models. Returning a tuple breaks metrics + the synthesizer.
        fmt = schema.model_json_schema() if schema is not None else None
        raw = ollama_generate(
            prompt,
            model=self.model_name,
            fmt=fmt,
            temperature=0.0,
            num_ctx=self.num_ctx,
            num_predict=3072,
        )
        if schema is not None:
            return _coerce(raw, schema)
        return raw

    async def a_generate(
        self, prompt: str, schema: Optional[type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        return await _run(self.generate, prompt, schema)


class OllamaEmbedder(DeepEvalBaseEmbeddingModel):
    def __init__(self, model: Optional[str] = None):
        self.model_name = model or SETTINGS.embed_model

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return f"ollama:{self.model_name}"

    def embed_text(self, text: str) -> list[float]:
        return embed_one(text, role="document", model=self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return embed_many(texts, role="document", model=self.model_name)

    async def a_embed_text(self, text: str) -> list[float]:
        return await _run(self.embed_text, text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await _run(self.embed_texts, texts)


# ---------------------------------------------------------------------------
# Claude judge + embedder
# ---------------------------------------------------------------------------

class ClaudeJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by the Anthropic API (claude-sonnet-4-6).

    Uses tool_use for structured output so the model is forced to emit
    schema-conformant JSON — no regex extraction needed."""

    def __init__(self, model: Optional[str] = None):
        self.model_name = model or SETTINGS.judge_model
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            import anthropic, os
            auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
            if auth_token:
                self._client = anthropic.Anthropic(auth_token=auth_token)
            else:
                self._client = anthropic.Anthropic()
        return self._client

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return f"claude:{self.model_name}"

    def generate(
        self, prompt: str, schema: Optional[type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        # NOTE: must return the BARE result (not a tuple) — see OllamaJudge note above.
        if schema is not None:
            json_schema = schema.model_json_schema()
            tool = {
                "name": "output",
                "description": "Return the evaluation result",
                "input_schema": json_schema,
            }
            from ..claude_client import _get_limiter
            _get_limiter(self.model_name).acquire()
            resp = self._get_client().messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                tool_choice={"type": "tool", "name": "output"},
            )
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "output":
                    return schema.model_validate(block.input)
            return schema.model_validate({})
        from ..claude_client import _get_limiter
        _get_limiter(self.model_name).acquire()
        resp = self._get_client().messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    async def a_generate(
        self, prompt: str, schema: Optional[type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        return await _run(self.generate, prompt, schema)


class STEmbedder(DeepEvalBaseEmbeddingModel):
    """sentence-transformers embedder for DeepEval (provider=claude)."""

    def __init__(self, model: Optional[str] = None):
        self.model_name = model or SETTINGS.embed_model

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return f"st:{self.model_name}"

    def embed_text(self, text: str) -> list[float]:
        return embed_one(text, role="document")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return embed_many(texts, role="document")

    async def a_embed_text(self, text: str) -> list[float]:
        return await _run(self.embed_text, text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await _run(self.embed_texts, texts)


# ---------------------------------------------------------------------------
# Factories — always use these instead of instantiating classes directly
# ---------------------------------------------------------------------------

def get_judge() -> DeepEvalBaseLLM:
    if SETTINGS.provider == "claude":
        return ClaudeJudge()
    return OllamaJudge()


def get_embedder() -> DeepEvalBaseEmbeddingModel:
    if SETTINGS.provider == "claude":
        return STEmbedder()
    return OllamaEmbedder()

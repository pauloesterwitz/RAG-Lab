"""Local Ollama-backed model + embedder for DeepEval, so all judging runs
offline on this box (no OpenAI key). We control num_ctx / num_predict because
Faithfulness / Contextual* prompts get long."""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import re
from typing import Optional, Tuple, Union

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
    def __init__(self, model: Optional[str] = None, num_ctx: int = 10240):
        self.model_name = model or SETTINGS.judge_model
        self.num_ctx = num_ctx

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

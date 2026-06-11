"""Cross-encoder reranker. Primary: Jina reranker v2 (pulled from HuggingFace).
Falls back to BGE reranker, then to an LLM-based pointwise reranker, so the
pipeline always works even if a model download or arch fails on this box."""
from __future__ import annotations

import json
from typing import Optional

from .config import SETTINGS
from .ollama_client import generate

_BACKEND = None  # cached singleton
_BACKEND_NAME = "uninitialized"


def backend_name() -> str:
    return _BACKEND_NAME


def _shim_xlm_roberta(torch):
    """Jina reranker v2's remote code imports a helper transformers 5.x removed.
    Re-inject it so the custom modeling file imports cleanly."""
    try:
        import transformers.models.xlm_roberta.modeling_xlm_roberta as xlmr

        if not hasattr(xlmr, "create_position_ids_from_input_ids"):
            def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
                mask = input_ids.ne(padding_idx).int()
                inc = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
                return inc.long() + padding_idx

            xlmr.create_position_ids_from_input_ids = create_position_ids_from_input_ids
    except Exception:
        pass


class _JinaReranker:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification

        self.torch = torch
        _shim_xlm_roberta(torch)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, docs: list[str]) -> list[float]:
        pairs = [[query, d] for d in docs]
        with self.torch.no_grad():
            scores = self.model.compute_score(pairs, max_length=1024)
        if not isinstance(scores, list):
            scores = [float(scores)]
        return [float(s) for s in scores]


class _CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=1024, trust_remote_code=True)

    def score(self, query: str, docs: list[str]) -> list[float]:
        pairs = [[query, d] for d in docs]
        return [float(s) for s in self.model.predict(pairs)]


class _LLMReranker:
    """Pointwise LLM reranker: ask the judge to score relevance 0-10."""

    def score(self, query: str, docs: list[str]) -> list[float]:
        scores = []
        schema = {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]}
        for d in docs:
            prompt = (
                f"Question: {query}\n\nPassage:\n{d[:1500]}\n\n"
                "On a scale of 0 to 10, how relevant is this passage to answering the "
                'question? Reply as JSON: {"score": <number>}.'
            )
            try:
                raw = generate(prompt, fmt=schema, num_predict=64, temperature=0.0)
                scores.append(float(json.loads(raw).get("score", 0)))
            except Exception:
                scores.append(0.0)
        return scores


def _init_backend():
    global _BACKEND, _BACKEND_NAME
    if _BACKEND is not None:
        return _BACKEND
    # 1) Jina (requested)
    try:
        _BACKEND = _JinaReranker(SETTINGS.reranker_model)
        _BACKEND_NAME = SETTINGS.reranker_model
        return _BACKEND
    except Exception as e:  # noqa
        print(f"[reranker] Jina load failed ({e}); trying fallback cross-encoder…")
    # 2) BGE / generic cross-encoder
    try:
        _BACKEND = _CrossEncoderReranker(SETTINGS.reranker_fallback_model)
        _BACKEND_NAME = SETTINGS.reranker_fallback_model
        return _BACKEND
    except Exception as e:  # noqa
        print(f"[reranker] Cross-encoder fallback failed ({e}); using LLM reranker.")
    # 3) LLM pointwise
    _BACKEND = _LLMReranker()
    _BACKEND_NAME = f"llm:{SETTINGS.judge_model}"
    return _BACKEND


def rerank(query: str, docs: list[str]) -> list[float]:
    """Return a relevance score per doc (higher is better)."""
    if not docs:
        return []
    return _init_backend().score(query, docs)


def warmup() -> str:
    _init_backend()
    return _BACKEND_NAME

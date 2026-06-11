"""Run DeepEval across every approach over the synthesized goldens.

Metrics: Answer Relevancy, Faithfulness, Contextual Relevancy, Contextual
Precision, Contextual Recall, plus a G-Eval Correctness metric. We also compute
retrieval-level metrics (gold-chunk hit rate, gold-doc hit rate, latency,
context size) that DeepEval doesn't cover, to round out the comparison."""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
from statistics import mean
from typing import Optional

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from ..config import SETTINGS, EVAL_DIR, APPROACH_ORDER, APPROACHES
from ..indexer import load_base_index
from ..approaches import get_approach
from ..reranker import backend_name
from .deepeval_models import OllamaJudge
from .synthesize import load_goldens

RESULTS_FILE = EVAL_DIR / "results.json"

METRIC_ORDER = [
    "Answer Relevancy",
    "Faithfulness",
    "Contextual Relevancy",
    "Contextual Precision",
    "Contextual Recall",
    "Correctness (G-Eval)",
]


def _build_metrics(judge: OllamaJudge) -> list:
    common = dict(model=judge, async_mode=True, include_reason=True)
    return [
        AnswerRelevancyMetric(**common),
        FaithfulnessMetric(**common),
        ContextualRelevancyMetric(**common),
        ContextualPrecisionMetric(**common),
        ContextualRecallMetric(**common),
        GEval(
            name="Correctness (G-Eval)",
            criteria=(
                "Determine whether the actual output is factually correct and complete "
                "with respect to the expected output, given the input question. Penalize "
                "contradictions and omissions of key facts."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            async_mode=True,
            _include_g_eval_suffix=False,  # keep __name__ == "Correctness (G-Eval)"
        ),
    ]


def _run_approach_outputs(approach_name: str, goldens: list[dict], index, progress=None):
    """Run the approach over every golden, returning test cases + per-case retrieval info."""
    approach = get_approach(approach_name, index)
    n = len(goldens)
    cases: list[Optional[LLMTestCase]] = [None] * n
    info: list[Optional[dict]] = [None] * n

    def work(i: int):
        g = goldens[i]
        res = approach.run(g["input"], generate_answer=True)
        retrieved_ids = [c.chunk.id for c in res.contexts]
        retrieved_docs = {c.chunk.doc for c in res.contexts}
        gold_ids = set(g.get("gold_chunk_ids", []))
        case = LLMTestCase(
            input=g["input"],
            actual_output=res.answer,
            expected_output=g.get("expected_output") or "",
            retrieval_context=res.context_texts(),
            context=list(g.get("context") or []),
        )
        meta = {
            "latency_s": res.latency_s,
            "num_contexts": len(res.contexts),
            "context_chars": sum(len(t) for t in res.context_texts()),
            "gold_chunk_hit": bool(gold_ids & set(retrieved_ids)),
            "gold_doc_hit": g.get("source_file") in retrieved_docs,
            "retrieved": [c.to_dict() for c in res.contexts],
            "answer": res.answer,
            "trace": [t.to_dict() for t in res.trace],
        }
        return i, case, meta

    with cf.ThreadPoolExecutor(max_workers=max(2, SETTINGS.gen_concurrency)) as ex:
        done = 0
        for fut in cf.as_completed([ex.submit(work, i) for i in range(n)]):
            i, case, meta = fut.result()
            cases[i], info[i] = case, meta
            done += 1
            if progress:
                progress(f"{approach_name}: generated {done}/{n} answers")
    return [c for c in cases], [m for m in info]


def _aggregate(test_results, cases_info: list[dict]) -> dict:
    # collect per-metric scores aligned to test cases
    per_metric: dict[str, list[float]] = {m: [] for m in METRIC_ORDER}
    per_case_scores: list[dict] = []
    for tr in test_results:
        md = {m.name: m for m in (tr.metrics_data or [])}

        def _find(name):
            if name in md:
                return md[name]
            base = name.split(" (")[0]  # tolerate suffix drift, e.g. "(GEval)"
            for k, v in md.items():
                if k.startswith(base):
                    return v
            return None

        row = {}
        for name in METRIC_ORDER:
            m = _find(name)
            if m is not None and m.score is not None:
                per_metric[name].append(float(m.score))
                row[name] = {"score": round(float(m.score), 3), "reason": (m.reason or "")[:400]}
            else:
                row[name] = {"score": None, "reason": (getattr(m, "error", "") or "") if m else "n/a"}
        per_case_scores.append(row)

    metrics_mean = {
        name: round(mean(vals), 3) if vals else None for name, vals in per_metric.items()
    }
    extra = {
        "avg_latency_s": round(mean([c["latency_s"] for c in cases_info]), 2),
        "gold_chunk_hit_rate": round(mean([1.0 if c["gold_chunk_hit"] else 0.0 for c in cases_info]), 3),
        "gold_doc_hit_rate": round(mean([1.0 if c["gold_doc_hit"] else 0.0 for c in cases_info]), 3),
        "avg_context_chars": int(mean([c["context_chars"] for c in cases_info])),
        "avg_num_contexts": round(mean([c["num_contexts"] for c in cases_info]), 1),
    }
    # composite DeepEval score = mean of available metric means
    avail = [v for v in metrics_mean.values() if v is not None]
    composite = round(mean(avail), 3) if avail else None
    return {"metrics": metrics_mean, "composite": composite, "extra": extra, "cases": per_case_scores}


def evaluate_approach(approach_name: str, goldens: list[dict], index, judge, progress=None) -> dict:
    if progress:
        progress(f"Running {approach_name}: generating answers…")
    cases, info = _run_approach_outputs(approach_name, goldens, index, progress)
    if progress:
        progress(f"Running {approach_name}: scoring {len(cases)} cases with DeepEval…")
    metrics = _build_metrics(judge)
    result = evaluate(
        test_cases=cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=True, max_concurrent=max(2, SETTINGS.gen_concurrency), throttle_value=0),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
    )
    agg = _aggregate(result.test_results, info)
    # attach the per-case retrieval detail next to the scores
    for row, meta in zip(agg["cases"], info):
        row["latency_s"] = round(meta["latency_s"], 2)
        row["gold_chunk_hit"] = meta["gold_chunk_hit"]
        row["gold_doc_hit"] = meta["gold_doc_hit"]
        row["answer"] = meta["answer"]
        row["retrieved"] = meta["retrieved"]
        row["trace"] = meta["trace"]
    agg["label"] = APPROACHES[approach_name]["label"]
    return agg


def run_full_eval(approaches: Optional[list[str]] = None, progress=None) -> dict:
    data = load_goldens()
    if not data or not data.get("goldens"):
        raise RuntimeError("No goldens found. Synthesize goldens first.")
    goldens = data["goldens"]
    index = load_base_index(refresh=True)
    judge = OllamaJudge()
    approaches = approaches or APPROACH_ORDER

    # include each approach's golden-set questions for transparency
    out: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": SETTINGS.judge_model,
        "gen_model": SETTINGS.gen_model,
        "embed_model": SETTINGS.embed_model,
        "reranker": backend_name(),
        "num_goldens": len(goldens),
        "metric_order": METRIC_ORDER,
        "goldens": [
            {"input": g["input"], "expected_output": g.get("expected_output"), "source_file": g.get("source_file")}
            for g in goldens
        ],
        "approaches": {},
    }
    for name in approaches:
        t0 = time.time()
        agg = evaluate_approach(name, goldens, index, judge, progress)
        agg["eval_seconds"] = round(time.time() - t0, 1)
        out["approaches"][name] = agg
        # persist incrementally so the dashboard updates as approaches finish
        RESULTS_FILE.write_text(json.dumps(out, indent=2))
        if progress:
            progress(f"✓ {name}: composite {agg['composite']} ({agg['eval_seconds']}s)")
    return out


def load_results() -> Optional[dict]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return None

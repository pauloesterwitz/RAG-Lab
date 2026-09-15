"""Retrieval-only eval: run approaches with generate_answer=False over the goldens
and report gold-chunk / gold-doc hits per hop. No answer generation and no DeepEval
scoring, so a pass takes minutes instead of hours. Use it to A/B one change at a
time, then confirm the winners with the full DeepEval run.

  .venv/bin/python scripts/retrieval_eval.py --tag baseline
  .venv/bin/python scripts/retrieval_eval.py --tag step1 -a pageindex --compare baseline
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_lab.approaches import get_approach  # noqa: E402
from rag_lab.config import APPROACH_ORDER, EVAL_DIR, SETTINGS  # noqa: E402
from rag_lab.eval.run_eval import gold_sets  # noqa: E402
from rag_lab.eval.synthesize import load_goldens  # noqa: E402
from rag_lab.indexer import load_base_index  # noqa: E402


def out_path(tag: str, approach: str) -> Path:
    return EVAL_DIR / f"retrieval_{tag}_{approach}.json"


def run_case(approach, g: dict) -> dict:
    res = approach.run(g["input"], generate_answer=False)
    gold_ids, gold_docs = gold_sets(g)
    ids = {c.chunk.id for c in res.contexts}
    docs = [c.chunk.doc for c in res.contexts]
    labels = [t.label for t in res.trace]
    return {
        "input": g["input"],
        "hop": g.get("hop", "single"),
        "chunk_hit": bool(gold_ids & ids),
        "chunk_recall": len(gold_ids & ids) / len(gold_ids) if gold_ids else 0.0,
        "doc_hit": bool(gold_docs & set(docs)),
        "all_docs": gold_docs <= set(docs),
        "gold_doc_share": sum(d in gold_docs for d in docs) / len(docs) if docs else 0.0,
        "fallback": any(c.stage == "fallback" for c in res.contexts),
        # PageIndex trace labels: one root-selection call plus one call per descend/stop decision.
        "llm_calls": (sum(l == "Root selection" or "Descend " in l or "Stopped at" in l for l in labels)
                      if approach.name.startswith("pageindex") else None),
        # Calls that still failed after the client's retries; PageIndex degrades silently on them.
        "failed_calls": sum(t.detail == "parse failed" or "Descend failed" in t.label for t in res.trace),
        "latency_s": round(res.latency_s, 2),
        "retrieved": [{k: v for k, v in c.to_dict().items() if k != "text"} for c in res.contexts],
        "trace": [t.to_dict() for t in res.trace],
    }


def summarize(cases: list[dict]) -> dict:
    out = {}
    for hop in ("all", "single", "multi"):
        cs = [c for c in cases if hop == "all" or c["hop"] == hop]
        if not cs:
            continue
        s = {
            "n": len(cs),
            "chunk_hits": sum(c["chunk_hit"] for c in cs),
            "chunk_recall": round(mean(c["chunk_recall"] for c in cs), 3),
            "doc_hits": sum(c["doc_hit"] for c in cs),
            "gold_doc_share": round(mean(c["gold_doc_share"] for c in cs), 3),
            "fallbacks": sum(c["fallback"] for c in cs),
            "failed_calls": sum(c["failed_calls"] for c in cs),
            "latency_s": round(mean(c["latency_s"] for c in cs), 1),
        }
        if hop == "multi":
            s["all_docs"] = sum(c["all_docs"] for c in cs)
        calls = [c["llm_calls"] for c in cs if c["llm_calls"] is not None]
        if calls:
            s["llm_calls"] = round(mean(calls), 2)
        out[hop] = s
    return out


def compare(cases: list[dict], ref_file: Path) -> str:
    ref = {c["input"]: c for c in json.loads(ref_file.read_text())["cases"]}
    parts = []
    for hop in ("single", "multi"):
        cs = [c for c in cases if c["hop"] == hop and c["input"] in ref]
        gained = sum(c["chunk_hit"] and not ref[c["input"]]["chunk_hit"] for c in cs)
        lost = sum(ref[c["input"]]["chunk_hit"] and not c["chunk_hit"] for c in cs)
        parts.append(f"{hop} +{gained}/-{lost}")
    net = sum(c["chunk_hit"] - ref[c["input"]]["chunk_hit"] for c in cases if c["input"] in ref)
    all_now = sum(c["all_docs"] for c in cases if c["hop"] == "multi")
    all_ref = sum(c["all_docs"] for c in ref.values() if c["hop"] == "multi")
    return f"vs {ref_file.name}: {', '.join(parts)}, net {net:+d}; multi all-docs {all_ref} -> {all_now}"


def use_llm_cache(name: str) -> None:
    """Replay identical model calls from disk. The pinned model isn't deterministic under
    concurrent load (two identical PageIndex runs flipped 11 cases), which swamps a paired A/B."""
    import atexit
    import hashlib
    import threading
    import rag_lab.llamaswap_client as client

    path = EVAL_DIR / f"llm_cache_{name}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    lock, stats, live = threading.Lock(), {"hits": 0, "new": 0}, client.generate

    def generate(prompt, *args, **kw):
        key = hashlib.sha1(json.dumps([prompt, args, kw], sort_keys=True, default=str).encode()).hexdigest()
        with lock:
            if key in cache:
                stats["hits"] += 1
                return cache[key]
        out = live(prompt, *args, **kw)
        with lock:
            cache[key] = out
            stats["new"] += 1
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(path)
        return out

    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("rag_lab.") and getattr(mod, "generate", None) is live:
            mod.generate = generate
    atexit.register(lambda: print(f"llm cache {name}: {stats['hits']} replayed, {stats['new']} new calls", flush=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run label, used in the output filename")
    ap.add_argument("-a", "--approaches", nargs="+", default=APPROACH_ORDER)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compare", metavar="TAG", help="print paired hit flips against an earlier run")
    ap.add_argument("--llm-cache", metavar="NAME", help="replay identical model calls via data/eval/llm_cache_NAME.json")
    args = ap.parse_args()

    goldens = load_goldens()["goldens"]
    index = load_base_index(refresh=True)
    if args.llm_cache:
        use_llm_cache(args.llm_cache)
    for name in args.approaches:
        approach = get_approach(name, index)
        t0 = time.time()
        cases: list = [None] * len(goldens)
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_case, approach, g): i for i, g in enumerate(goldens)}
            for done, fut in enumerate(cf.as_completed(futs), 1):
                cases[futs[fut]] = fut.result()
                if done % 20 == 0:
                    print(f"  {name}: {done}/{len(goldens)}", flush=True)
        summary = summarize(cases)
        out_path(args.tag, name).write_text(json.dumps({
            "approach": name, "tag": args.tag, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embed_model": SETTINGS.embed_model, "gen_model": SETTINGS.gen_model,
            "seconds": round(time.time() - t0, 1), "summary": summary, "cases": cases,
        }, indent=2))
        s, m = summary["single"], summary["multi"]
        print(f"{name} [{time.time() - t0:.0f}s] chunk hit {summary['all']['chunk_hits']}/100 "
              f"(single {s['chunk_hits']}, multi {m['chunk_hits']}) | recall {s['chunk_recall']}/{m['chunk_recall']} "
              f"| multi all-docs {m['all_docs']} | gold-doc share {summary['all']['gold_doc_share']} "
              f"| fallbacks {summary['all']['fallbacks']} | failed calls {summary['all']['failed_calls']}"
              + (f" | llm calls {summary['all']['llm_calls']}" if "llm_calls" in summary["all"] else ""),
              flush=True)
        if args.compare and out_path(args.compare, name).exists():
            print("  " + compare(cases, out_path(args.compare, name)), flush=True)


if __name__ == "__main__":
    main()

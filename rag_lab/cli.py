"""Command-line orchestration: embed, graph, synth, eval. The FastAPI server
reuses the same underlying functions, so CLI and UI stay in lock-step.

Usage:
    python -m rag_lab.cli embed        # (re)embed Documents -> base index
    python -m rag_lab.cli graph        # build the GraphRAG entity graph
    python -m rag_lab.cli synth [-n N]  # synthesize goldens
    python -m rag_lab.cli eval [-a ...] # run DeepEval over approaches
    python -m rag_lab.cli all [-n N]    # embed + graph + synth + eval
    python -m rag_lab.cli status        # show what's built
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import SETTINGS, APPROACH_ORDER
from .indexer import build_index, load_base_index, get_manifest
from .graph_build import build_graph, get_graph_meta
from .eval.synthesize import synthesize_goldens, load_goldens
from .eval.run_eval import run_full_eval, load_results


def _p(msg):
    print(msg, flush=True)


def cmd_embed(args):
    m = build_index(progress=lambda s, f, msg: _p(f"[embed {f*100:5.1f}%] {msg}"))
    _p(json.dumps(m, indent=2))


def cmd_graph(args):
    idx = load_base_index(refresh=True)
    m = build_graph(idx, progress=lambda s, f, msg: _p(f"[graph {f*100:5.1f}%] {msg}"))
    _p(json.dumps(m, indent=2))


def cmd_synth(args):
    payload = synthesize_goldens(num=args.num, progress=_p)
    _p(f"Synthesized {payload['num_goldens']} goldens.")


def cmd_eval(args):
    approaches = args.approaches or APPROACH_ORDER
    out = run_full_eval(approaches=approaches, progress=_p)
    _p("=== Summary (composite DeepEval score) ===")
    for name, agg in out["approaches"].items():
        _p(f"  {name:12s} composite={agg['composite']}  metrics={agg['metrics']}")


def cmd_all(args):
    cmd_embed(args)
    cmd_graph(args)
    cmd_synth(args)
    cmd_eval(args)


def cmd_status(args):
    _p("== Base index ==");  _p(json.dumps(get_manifest(), indent=2))
    _p("== Graph ==");       _p(json.dumps(get_graph_meta(), indent=2))
    g = load_goldens();      _p(f"== Goldens == {g['num_goldens'] if g else 0}")
    r = load_results()
    if r:
        _p("== Results ==")
        for name, agg in r.get("approaches", {}).items():
            _p(f"  {name:12s} composite={agg['composite']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag_lab.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed").set_defaults(func=cmd_embed)
    sub.add_parser("graph").set_defaults(func=cmd_graph)
    sp = sub.add_parser("synth"); sp.add_argument("-n", "--num", type=int, default=None); sp.set_defaults(func=cmd_synth)
    ep = sub.add_parser("eval"); ep.add_argument("-a", "--approaches", nargs="*", default=None); ep.set_defaults(func=cmd_eval)
    al = sub.add_parser("all"); al.add_argument("-n", "--num", type=int, default=None); al.add_argument("-a", "--approaches", nargs="*", default=None); al.set_defaults(func=cmd_all)
    sub.add_parser("status").set_defaults(func=cmd_status)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

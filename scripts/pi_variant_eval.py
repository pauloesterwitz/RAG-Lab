"""Full DeepEval run for one PageIndex variant, written to its own results file.

  .venv/bin/python scripts/pi_variant_eval.py route o1_route

The variant runs under the existing "pageindex" name (so DeepEval finds its label) and
the output goes to data/eval/results_<tag>.json, leaving the lab's results.json alone.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_lab.approaches import REGISTRY  # noqa: E402
from rag_lab.config import EVAL_DIR, SETTINGS  # noqa: E402
from rag_lab.eval.deepeval_models import get_judge  # noqa: E402
from rag_lab.eval.run_eval import METRIC_ORDER, evaluate_approach  # noqa: E402
from rag_lab.eval.synthesize import load_goldens  # noqa: E402
from rag_lab.indexer import load_base_index  # noqa: E402
from rag_lab.reranker import backend_name  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi_variants_run import VARIANTS  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: pi_variant_eval.py <{'|'.join(VARIANTS)}> <tag>")
    variant, tag = sys.argv[1], sys.argv[2]
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}")

    REGISTRY["pageindex"] = VARIANTS[variant]
    goldens = load_goldens()["goldens"]
    index = load_base_index(refresh=True)
    judge = get_judge()

    t0 = time.time()
    agg = evaluate_approach("pageindex", goldens, index, judge, progress=lambda m: print(m, flush=True))
    agg["eval_seconds"] = round(time.time() - t0, 1)

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variant": f"{variant} ({VARIANTS[variant].__name__})",
        "judge_model": SETTINGS.judge_model,
        "gen_model": SETTINGS.gen_model,
        "embed_model": SETTINGS.embed_model,
        "reranker": backend_name(),
        "num_goldens": len(goldens),
        "metric_order": METRIC_ORDER,
        "approaches": {"pageindex": agg},
    }
    path = EVAL_DIR / f"results_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{tag}: composite {agg['composite']} | gold-chunk hit {agg['extra']['gold_chunk_hit_rate']} "
          f"| {agg['eval_seconds'] / 60:.0f} min -> {path}", flush=True)


if __name__ == "__main__":
    main()

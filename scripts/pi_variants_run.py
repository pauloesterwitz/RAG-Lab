"""Run one round-3 PageIndex navigation variant through the retrieval harness.

  .venv/bin/python scripts/pi_variants_run.py --variant whole --tag v1_whole \
      -a pageindex --workers 4 --llm-cache pi_r3 --compare r3_pi

The variant is registered under the existing "pageindex" name so --compare can pair
it against the baseline runs; every other flag goes to scripts/retrieval_eval.py.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from rag_lab.approaches import REGISTRY  # noqa: E402
from rag_lab.approaches import pageindex_variants as variants  # noqa: E402
from rag_lab.approaches.pageindex import PageIndexRAG  # noqa: E402

VARIANTS = {
    "base": PageIndexRAG,  # parity gate: must reproduce the baseline with no new model calls
    "whole": variants.WholeTreePageIndex,
    "prior": variants.TreePriorPageIndex,
    "beam": variants.BeamPageIndex,
    "suff": variants.SufficiencyReentryPageIndex,
    "route": variants.ScoreRoutedPageIndex,
    "route2": variants.ScoreRoutedTop2PageIndex,
    "routeflat": variants.ScoreRoutedNoFloorPageIndex,
    "evidence": variants.EvidenceNavPageIndex,
    "fine": variants.FineGrainPageIndex,
}


def main() -> None:
    argv = sys.argv[1:]
    if "--variant" not in argv:
        raise SystemExit(f"--variant is required, one of: {', '.join(VARIANTS)}")
    i = argv.index("--variant")
    name = argv[i + 1]
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}, expected one of: {', '.join(VARIANTS)}")
    del argv[i:i + 2]

    REGISTRY["pageindex"] = VARIANTS[name]
    import retrieval_eval

    sys.argv = ["retrieval_eval.py", *argv]
    print(f"variant: {name} ({VARIANTS[name].__name__})", flush=True)
    retrieval_eval.main()


if __name__ == "__main__":
    main()

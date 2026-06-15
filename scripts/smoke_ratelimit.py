import json, time
from rag_lab.indexer import load_base_index
from rag_lab.eval.synthesize import load_goldens
from rag_lab.eval.run_eval import evaluate_approach
from rag_lab.eval.deepeval_models import get_judge

g = load_goldens()
goldens = g["goldens"][:1]   # ONE golden — just prove no 429/None
idx = load_base_index(refresh=True)
judge = get_judge()
t0=time.time()
print("Running plain over 1 golden with Sonnet judge (retry+shared-limiter)…", flush=True)
agg = evaluate_approach("plain", goldens, idx, judge, progress=lambda m: print("  ", m, flush=True))
print("METRICS:", json.dumps(agg["metrics"]), flush=True)
print("metrics_used:", agg.get("metrics_used"), "/", agg.get("metrics_total"), flush=True)
none_ct = sum(1 for v in agg["metrics"].values() if v is None)
print("NONE metrics:", none_ct, "| composite:", agg["composite"], "| %.0fs"%(time.time()-t0), flush=True)
print("SMOKE_OK" if none_ct==0 else "SMOKE_HAS_NONE", flush=True)

"""Full pipeline: build GraphRAG graph -> synthesize goldens -> DeepEval all
approaches. Results persist incrementally (data/eval/results.json) so the
dashboard updates after each approach completes."""
import time
from rag_lab.graph_build import build_graph, graph_exists, get_graph_meta
from rag_lab.indexer import load_base_index
from rag_lab.eval.synthesize import synthesize_goldens
from rag_lab.eval.run_eval import run_full_eval

N_GOLDENS = 16
t0 = time.time()

idx = load_base_index(refresh=True)

print("=== STAGE 1/3: GraphRAG graph ===", flush=True)
meta = build_graph(idx, progress=lambda s, f, m: print(f"  [graph {f*100:3.0f}%] {m}", flush=True))
print("  graph:", meta, flush=True)

print("=== STAGE 2/3: synthesize goldens ===", flush=True)
payload = synthesize_goldens(num=N_GOLDENS, progress=lambda m: print("  synth:", m, flush=True))
print(f"  synth: {payload['num_goldens']} goldens", flush=True)

print("=== STAGE 3/3: DeepEval all approaches ===", flush=True)
out = run_full_eval(progress=lambda m: print("  eval:", m, flush=True))
print("=== SUMMARY ===", flush=True)
for name, agg in out["approaches"].items():
    print(f"  {name:12s} composite={agg['composite']}  {agg['metrics']}", flush=True)
print(f"FULL_PIPELINE_DONE in {(time.time()-t0)/60:.1f} min", flush=True)

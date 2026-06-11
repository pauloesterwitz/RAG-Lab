import json, time
from rag_lab.eval.synthesize import synthesize_goldens
from rag_lab.eval.run_eval import run_full_eval, RESULTS_FILE

t0=time.time()
print("=== SYNTH (n=2) ===", flush=True)
payload = synthesize_goldens(num=2, progress=lambda m: print("  synth:", m, flush=True))
print("goldens:", payload["num_goldens"], flush=True)
for g in payload["goldens"]:
    print("  Q:", (g["input"] or "")[:90], "| gold_ids:", g["gold_chunk_ids"], flush=True)

print("=== EVAL (plain) ===", flush=True)
out = run_full_eval(approaches=["plain"], progress=lambda m: print("  eval:", m, flush=True))
agg = out["approaches"]["plain"]
print("METRICS:", json.dumps(agg["metrics"]), flush=True)
print("COMPOSITE:", agg["composite"], "| EXTRA:", json.dumps(agg["extra"]), flush=True)
print("total_time_s=%.0f"%(time.time()-t0), flush=True)
print("VALIDATE_OK", flush=True)

"""Generate one knowledge-graph diagram per DeepEval metric.

Calls the plot-mcp-server engine/inspector directly (no MCP protocol overhead).
Outputs PNG + SVG to docs/deepeval/.  Runs automated VQA after each render.
"""
import json
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
PLOT_SRV = Path("/home/pauloesterwitz/.local/share/opencode/plot-mcp-server")
sys.path.insert(0, str(PLOT_SRV))

import engine      # noqa: E402
import inspector   # noqa: E402
import validator   # noqa: E402

OUT = Path("/home/pauloesterwitz/Bosch/RAG PoCs/docs/deepeval")
OUT.mkdir(parents=True, exist_ok=True)


def kg(title, nodes, edges, name, layout="spring"):
    out_path = str(OUT / name)
    val = validator.validate_knowledge_graph(nodes, edges)
    if not val["ok"]:
        print(f"  [WARN] validation: {val}")
    gen = engine.generate_knowledge_graph(
        title, nodes, edges, out_path,
        layout=layout, figure_size="graph", style="business",
    )
    qa = inspector.inspect_outputs(gen, plot_type="knowledge_graph")
    ok = gen.get("ok", False)
    png = gen.get("outputs", {}).get("png", "—")
    qa_ok = (qa or {}).get("clean", "?")
    qa_issues = (qa or {}).get("issues", [])
    print(f"  ok={ok}  png={png}")
    print(f"  qa_clean={qa_ok}  issues={qa_issues}")
    return gen, qa


# ════════════════════════════════════════════════════════════════════════════
# 1. Answer Relevancy
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Answer Relevancy")
kg(
    title="Answer Relevancy — How DeepEval Scores It",
    nodes=[
        {"id": "Q",   "label": "User Question",          "group": "Input"},
        {"id": "A",   "label": "Generated Answer",        "group": "Input"},
        {"id": "GEN", "label": "LLM: Generate\nHypothetical Questions", "group": "Process"},
        {"id": "HQ1", "label": "Hyp. Question 1",         "group": "Candidate"},
        {"id": "HQ2", "label": "Hyp. Question 2",         "group": "Candidate"},
        {"id": "HQ3", "label": "Hyp. Question 3",         "group": "Candidate"},
        {"id": "COS", "label": "Cosine Similarity\n(embeddings)", "group": "Process"},
        {"id": "SC",  "label": "Score = mean(sim)",       "group": "Score"},
    ],
    edges=[
        {"source": "A",   "target": "GEN", "label": "input"},
        {"source": "GEN", "target": "HQ1", "label": "generates"},
        {"source": "GEN", "target": "HQ2", "label": "generates"},
        {"source": "GEN", "target": "HQ3", "label": "generates"},
        {"source": "Q",   "target": "COS", "label": "original"},
        {"source": "HQ1", "target": "COS", "label": "compare"},
        {"source": "HQ2", "target": "COS", "label": "compare"},
        {"source": "HQ3", "target": "COS", "label": "compare"},
        {"source": "COS", "target": "SC",  "label": "aggregates"},
    ],
    name="answer_relevancy",
    layout="spring",
)

# ════════════════════════════════════════════════════════════════════════════
# 2. Faithfulness
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Faithfulness")
kg(
    title="Faithfulness — How DeepEval Scores It",
    nodes=[
        {"id": "A",   "label": "Generated Answer",     "group": "Input"},
        {"id": "CTX", "label": "Retrieved Context\n(chunks)", "group": "Input"},
        {"id": "EXT", "label": "LLM: Extract\nClaims",        "group": "Process"},
        {"id": "C1",  "label": "Claim 1",              "group": "Claim"},
        {"id": "C2",  "label": "Claim 2",              "group": "Claim"},
        {"id": "C3",  "label": "Claim N",              "group": "Claim"},
        {"id": "VER", "label": "LLM: Verify\neach Claim vs Context", "group": "Process"},
        {"id": "SC",  "label": "Score =\n#supported / #total", "group": "Score"},
    ],
    edges=[
        {"source": "A",   "target": "EXT", "label": "input"},
        {"source": "EXT", "target": "C1",  "label": "extracts"},
        {"source": "EXT", "target": "C2",  "label": "extracts"},
        {"source": "EXT", "target": "C3",  "label": "extracts"},
        {"source": "CTX", "target": "VER", "label": "evidence"},
        {"source": "C1",  "target": "VER", "label": "verify"},
        {"source": "C2",  "target": "VER", "label": "verify"},
        {"source": "C3",  "target": "VER", "label": "verify"},
        {"source": "VER", "target": "SC",  "label": "→"},
    ],
    name="faithfulness",
    layout="spring",
)

# ════════════════════════════════════════════════════════════════════════════
# 3. Contextual Relevancy
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Contextual Relevancy")
kg(
    title="Contextual Relevancy — How DeepEval Scores It",
    nodes=[
        {"id": "Q",   "label": "User Question",            "group": "Input"},
        {"id": "CH1", "label": "Chunk 1",                  "group": "Context"},
        {"id": "CH2", "label": "Chunk 2",                  "group": "Context"},
        {"id": "CHN", "label": "Chunk N",                  "group": "Context"},
        {"id": "JDG", "label": "LLM: Judge\nRelevance",    "group": "Process"},
        {"id": "RS1", "label": "Relevant\nStatements 1",   "group": "Relevant"},
        {"id": "RS2", "label": "Relevant\nStatements 2",   "group": "Relevant"},
        {"id": "SC",  "label": "Score =\n#relevant / #total", "group": "Score"},
    ],
    edges=[
        {"source": "Q",   "target": "JDG", "label": "question"},
        {"source": "CH1", "target": "JDG", "label": "evaluate"},
        {"source": "CH2", "target": "JDG", "label": "evaluate"},
        {"source": "CHN", "target": "JDG", "label": "evaluate"},
        {"source": "JDG", "target": "RS1", "label": "extracts"},
        {"source": "JDG", "target": "RS2", "label": "extracts"},
        {"source": "RS1", "target": "SC",  "label": "→"},
        {"source": "RS2", "target": "SC",  "label": "→"},
    ],
    name="contextual_relevancy",
    layout="spring",
)

# ════════════════════════════════════════════════════════════════════════════
# 4. Contextual Precision
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Contextual Precision")
kg(
    title="Contextual Precision — How DeepEval Scores It",
    nodes=[
        {"id": "Q",    "label": "User Question",          "group": "Input"},
        {"id": "EXP",  "label": "Expected Answer",        "group": "Input"},
        {"id": "RANK", "label": "Ranked Context\n(top-k chunks)", "group": "Input"},
        {"id": "JDG",  "label": "LLM: Judge\nRelevance per Rank", "group": "Process"},
        {"id": "R1",   "label": "Rank 1: relevant?",     "group": "Judgment"},
        {"id": "R2",   "label": "Rank 2: relevant?",     "group": "Judgment"},
        {"id": "RN",   "label": "Rank N: relevant?",     "group": "Judgment"},
        {"id": "SC",   "label": "Score = weighted\nprecision@k", "group": "Score"},
    ],
    edges=[
        {"source": "Q",    "target": "JDG",  "label": "context"},
        {"source": "EXP",  "target": "JDG",  "label": "reference"},
        {"source": "RANK", "target": "R1",   "label": "pos 1"},
        {"source": "RANK", "target": "R2",   "label": "pos 2"},
        {"source": "RANK", "target": "RN",   "label": "pos N"},
        {"source": "JDG",  "target": "R1",   "label": "judges"},
        {"source": "JDG",  "target": "R2",   "label": "judges"},
        {"source": "JDG",  "target": "RN",   "label": "judges"},
        {"source": "R1",   "target": "SC",   "label": "↑ weight"},
        {"source": "R2",   "target": "SC",   "label": "↑ weight"},
        {"source": "RN",   "target": "SC",   "label": "↓ weight"},
    ],
    name="contextual_precision",
    layout="spring",
)

# ════════════════════════════════════════════════════════════════════════════
# 5. Contextual Recall
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Contextual Recall")
kg(
    title="Contextual Recall — How DeepEval Scores It",
    nodes=[
        {"id": "EXP",  "label": "Expected Answer",       "group": "Input"},
        {"id": "CTX",  "label": "Retrieved Context",     "group": "Input"},
        {"id": "EXT",  "label": "LLM: Extract\nStatements", "group": "Process"},
        {"id": "S1",   "label": "Statement 1",           "group": "Statement"},
        {"id": "S2",   "label": "Statement 2",           "group": "Statement"},
        {"id": "SN",   "label": "Statement N",           "group": "Statement"},
        {"id": "CHK",  "label": "LLM: Check\nContext Support", "group": "Process"},
        {"id": "SC",   "label": "Score =\n#covered / #total", "group": "Score"},
    ],
    edges=[
        {"source": "EXP",  "target": "EXT", "label": "decompose"},
        {"source": "EXT",  "target": "S1",  "label": "extracts"},
        {"source": "EXT",  "target": "S2",  "label": "extracts"},
        {"source": "EXT",  "target": "SN",  "label": "extracts"},
        {"source": "CTX",  "target": "CHK", "label": "evidence"},
        {"source": "S1",   "target": "CHK", "label": "check"},
        {"source": "S2",   "target": "CHK", "label": "check"},
        {"source": "SN",   "target": "CHK", "label": "check"},
        {"source": "CHK",  "target": "SC",  "label": "→"},
    ],
    name="contextual_recall",
    layout="spring",
)

# ════════════════════════════════════════════════════════════════════════════
# 6. Correctness (G-Eval)
# ════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Correctness (G-Eval)")
kg(
    title="Correctness (G-Eval) — How DeepEval Scores It",
    nodes=[
        {"id": "A",   "label": "Generated Answer",       "group": "Input"},
        {"id": "REF", "label": "Reference Answer\n(golden)", "group": "Input"},
        {"id": "CRT", "label": "Evaluation\nCriteria",   "group": "Process"},
        {"id": "COT", "label": "LLM: Chain-of-Thought\nReasoning", "group": "Process"},
        {"id": "CR1", "label": "Correctness\ncriterion", "group": "Criterion"},
        {"id": "CR2", "label": "Completeness\ncriterion","group": "Criterion"},
        {"id": "CR3", "label": "Consistency\ncriterion", "group": "Criterion"},
        {"id": "SC",  "label": "Weighted\nG-Eval Score", "group": "Score"},
    ],
    edges=[
        {"source": "A",   "target": "COT", "label": "compare"},
        {"source": "REF", "target": "COT", "label": "compare"},
        {"source": "CRT", "target": "CR1", "label": "defines"},
        {"source": "CRT", "target": "CR2", "label": "defines"},
        {"source": "CRT", "target": "CR3", "label": "defines"},
        {"source": "COT", "target": "CR1", "label": "scores"},
        {"source": "COT", "target": "CR2", "label": "scores"},
        {"source": "COT", "target": "CR3", "label": "scores"},
        {"source": "CR1", "target": "SC",  "label": "→"},
        {"source": "CR2", "target": "SC",  "label": "→"},
        {"source": "CR3", "target": "SC",  "label": "→"},
    ],
    name="correctness_geval",
    layout="spring",
)

print("\nAll 6 plots generated.")
print("PNGs:", sorted(p.name for p in OUT.glob("*.png")))

# ── Brand logo overlay (bottom-right) ────────────────────────────
from PIL import Image as _PIL

_LOGO_PATH = Path("/home/pauloesterwitz/PPTX MCP/AI & SAP Consulting Paul Oesterwitz/assets/logo-trans.png")
if _LOGO_PATH.exists():
    _logo_src = _PIL.open(_LOGO_PATH).convert("RGBA")
    _logo_h = 44
    _logo = _logo_src.resize(
        (int(_logo_src.width * _logo_h / _logo_src.height), _logo_h), _PIL.LANCZOS
    )
    for _p in sorted(OUT.glob("*.png")):
        _img = _PIL.open(_p).convert("RGBA")
        _W, _H = _img.size
        _img.paste(_logo, (_W - _logo.width - 18, _H - _logo.height - 18), _logo)
        _img.convert("RGB").save(_p, "PNG")
    print("Brand logo added to all PNGs.")
else:
    print("[WARN] Brand logo not found — skipping overlay.")

"""
make_diagrams.py – regenerate all 6 RAG pipeline schema diagrams.

Uses matplotlib.patches (Circle, FancyArrowPatch) so we control every pixel.
No networkx draw helpers. Saves PNG + SVG to docs/schemas/.
"""

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.image import imread

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "schemas"
LOGO_PATH = Path("/home/pauloesterwitz/AI & SAP Consulting/website/dist/assets/logo-DB8lDVaM.png")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------
COLORS = {
    "input":        "#D97706",
    "retrieval":    "#1D4ED8",
    "intermediate": "#7C3AED",
    "generation":   "#059669",
    "output":       "#B91C1C",
    "decision":     "#B91C1C",
    "correction":   "#7C3AED",
    "agent":        "#1D4ED8",
}
EDGE_COLOR = "#9aa3b2"
NODE_RADIUS = 0.35
FIG_SIZE = (9, 10.5)
DPI = 150
DATA_XLIM = (0, 10)
DATA_YLIM = (0, 9)
FOOTER = "AI & SAP Consulting Paul Oesterwitz  |  oesterwitz-consulting.de"

LEGEND_TYPES = [
    ("input",        "Input"),
    ("retrieval",    "Retrieval"),
    ("generation",   "Generation / LLM"),
    ("intermediate", "Intermediate"),
    ("output",       "Output / Decision"),
]

LABEL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.88,
                  boxstyle="round,pad=0.3")
EDGE_LABEL_STYLE = dict(fontsize=7, color="#555", fontstyle="italic",
                        ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.82, boxstyle="round,pad=0.2"))

# ---------------------------------------------------------------------------
# Helper – draw one diagram
# ---------------------------------------------------------------------------

def draw_diagram(name: str, title: str, nodes: list[dict], edges: list[dict]) -> None:
    """
    nodes: [{"id": str, "pos": (x, y), "type": str}]
    edges: [{"src": str, "dst": str, "label": str}]
    """
    node_map = {n["id"]: n for n in nodes}

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(*DATA_XLIM)
    ax.set_ylim(*DATA_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- edges (draw before nodes so nodes sit on top) ---
    for e in edges:
        src = node_map[e["src"]]
        dst = node_map[e["dst"]]
        x0, y0 = src["pos"]
        x1, y1 = dst["pos"]

        # shorten endpoints so arrow doesn't overlap circle fill
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        pad = NODE_RADIUS + 0.08
        sx, sy = x0 + ux * pad, y0 + uy * pad
        ex, ey = x1 - ux * pad, y1 - uy * pad

        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey),
            arrowstyle="-|>",
            mutation_scale=12,
            color=EDGE_COLOR,
            linewidth=1.4,
            zorder=2,
        )
        ax.add_patch(arrow)

        label = e.get("label", "")
        if label:
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            # slight perpendicular nudge so label doesn't sit on the line
            perp_x, perp_y = -uy * 0.18, ux * 0.18
            ax.text(mx + perp_x, my + perp_y, label, zorder=5,
                    **EDGE_LABEL_STYLE)

    # --- nodes ---
    for n in nodes:
        x, y = n["pos"]
        color = COLORS.get(n["type"], "#888888")
        circle = Circle((x, y), NODE_RADIUS,
                         facecolor=color, edgecolor="white",
                         linewidth=2, zorder=3)
        ax.add_patch(circle)

        # label position: prefer below node; if near bottom edge use above
        label_y = y - NODE_RADIUS - 0.28
        va = "top"
        if label_y < DATA_YLIM[0] + 0.3:
            label_y = y + NODE_RADIUS + 0.22
            va = "bottom"

        ax.text(x, label_y, n["id"],
                ha="center", va=va,
                fontsize=7.5, fontweight="bold", color="#1a1a2e",
                zorder=6, bbox=LABEL_BBOX,
                wrap=False)

    # --- logo inset (top-left, outside data area) ---
    if LOGO_PATH.exists():
        logo_ax = fig.add_axes([0.02, 0.925, 0.22, 0.055])
        logo_ax.set_axis_off()
        logo_img = imread(str(LOGO_PATH))
        logo_ax.imshow(logo_img, aspect="auto")

    # --- title below logo area ---
    fig.text(0.04, 0.91, title,
             ha="left", va="top",
             fontsize=11, fontweight="bold", color="#1a1a2e",
             transform=fig.transFigure)

    # --- legend box ---
    legend_handles = []
    for ltype, lname in LEGEND_TYPES:
        legend_handles.append(
            mpatches.Patch(facecolor=COLORS[ltype], edgecolor="white",
                           label=lname)
        )
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.87),
        bbox_transform=fig.transFigure,
        fontsize=7,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=1.2,
        handleheight=0.9,
        borderpad=0.5,
        labelspacing=0.35,
        ncol=2,
    )
    legend.get_frame().set_linewidth(0.6)

    # --- footer ---
    fig.text(0.5, 0.005, FOOTER,
             ha="center", va="bottom",
             fontsize=7, color="#777", style="italic",
             transform=fig.transFigure)

    # --- save ---
    plt.tight_layout(rect=[0, 0.015, 1, 0.89])
    for ext in ("png", "svg"):
        out_path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                    facecolor="white")
        print(f"  wrote {out_path}")
    plt.close(fig)


# ===========================================================================
# Diagram definitions
# ===========================================================================

def plain_rag():
    nodes = [
        {"id": "Query",                    "pos": (8.0, 8.0), "type": "input"},
        {"id": "BM25 Score",               "pos": (3.5, 6.5), "type": "retrieval"},
        {"id": "Embed Query",              "pos": (8.0, 6.0), "type": "retrieval"},
        {"id": "Score Fusion",             "pos": (5.5, 4.5), "type": "retrieval"},
        {"id": "Top-K Chunks",             "pos": (8.0, 3.0), "type": "retrieval"},
        {"id": "LLM\n(claude-sonnet-4-6)", "pos": (4.5, 1.5), "type": "generation"},
        {"id": "Answer (cited)",           "pos": (1.5, 1.5), "type": "output"},
    ]
    edges = [
        {"src": "Query",                    "dst": "BM25 Score",               "label": "keywords"},
        {"src": "Query",                    "dst": "Embed Query",               "label": "embed"},
        {"src": "BM25 Score",               "dst": "Score Fusion",             "label": "sparse scores"},
        {"src": "Embed Query",              "dst": "Score Fusion",             "label": "dense scores"},
        {"src": "Score Fusion",             "dst": "Top-K Chunks",             "label": ""},
        {"src": "Top-K Chunks",             "dst": "LLM\n(claude-sonnet-4-6)", "label": "context"},
        {"src": "LLM\n(claude-sonnet-4-6)", "dst": "Answer (cited)",           "label": ""},
    ]
    draw_diagram("plain_rag", "Plain RAG (Hybrid Retrieval)", nodes, edges)


def rerank_rag():
    nodes = [
        {"id": "Query",                       "pos": (8.0, 8.0), "type": "input"},
        {"id": "BM25 Score",                  "pos": (3.5, 6.5), "type": "retrieval"},
        {"id": "Embed Query",                 "pos": (8.0, 6.0), "type": "retrieval"},
        {"id": "Score Fusion",                "pos": (5.8, 4.8), "type": "retrieval"},
        {"id": "Candidate Pool",              "pos": (8.5, 3.2), "type": "retrieval"},
        {"id": "Reranker\n(Jina v2)",         "pos": (5.5, 1.8), "type": "intermediate"},
        {"id": "Top-K Chunks",                "pos": (2.8, 1.8), "type": "retrieval"},
        {"id": "LLM\n(claude-sonnet-4-6)",    "pos": (1.2, 0.5), "type": "generation"},
        # output moved to far left, below
    ]
    # Answer (cited) needs space – put at separate position
    nodes.append({"id": "Answer (cited)", "pos": (4.5, 0.5), "type": "output"})

    edges = [
        {"src": "Query",                    "dst": "BM25 Score",              "label": "keywords"},
        {"src": "Query",                    "dst": "Embed Query",              "label": "embed"},
        {"src": "BM25 Score",               "dst": "Score Fusion",            "label": "sparse scores"},
        {"src": "Embed Query",              "dst": "Score Fusion",            "label": "dense scores"},
        {"src": "Score Fusion",             "dst": "Candidate Pool",          "label": "top-50"},
        {"src": "Candidate Pool",           "dst": "Reranker\n(Jina v2)",     "label": "candidates"},
        {"src": "Reranker\n(Jina v2)",      "dst": "Top-K Chunks",            "label": "reranked"},
        {"src": "Top-K Chunks",             "dst": "LLM\n(claude-sonnet-4-6)","label": "context"},
        {"src": "LLM\n(claude-sonnet-4-6)", "dst": "Answer (cited)",          "label": ""},
    ]
    draw_diagram("rerank_rag", "Rerank RAG (Cross-Encoder Reranking)", nodes, edges)


def hyde_rag():
    nodes = [
        {"id": "Query",                      "pos": (5.0, 8.0), "type": "input"},
        {"id": "Hypothetical Answer\n(LLM)", "pos": (5.0, 6.2), "type": "generation"},
        {"id": "Embed Hypothesis",           "pos": (5.0, 4.4), "type": "retrieval"},
        {"id": "Top-K Chunks",               "pos": (5.0, 2.8), "type": "retrieval"},
        {"id": "LLM\n(claude-sonnet-4-6)",   "pos": (5.0, 1.3), "type": "generation"},
        {"id": "Answer (cited)",             "pos": (1.5, 1.3), "type": "output"},
    ]
    edges = [
        {"src": "Query",                      "dst": "Hypothetical Answer\n(LLM)", "label": "generate hypothesis"},
        {"src": "Hypothetical Answer\n(LLM)", "dst": "Embed Hypothesis",           "label": "embed"},
        {"src": "Embed Hypothesis",           "dst": "Top-K Chunks",               "label": "dense search"},
        {"src": "Top-K Chunks",               "dst": "LLM\n(claude-sonnet-4-6)",   "label": "context"},
        {"src": "LLM\n(claude-sonnet-4-6)",   "dst": "Answer (cited)",             "label": ""},
    ]
    draw_diagram("hyde_rag", "HyDE RAG (Hypothetical Document Embeddings)", nodes, edges)


def corrective_rag():
    nodes = [
        {"id": "Query",                          "pos": (5.0, 8.3),  "type": "input"},
        {"id": "Initial Retrieval\n(hybrid k=20)","pos": (5.0, 6.8), "type": "retrieval"},
        {"id": "LLM: Grade\nRelevance (0–1)",    "pos": (5.0, 5.3),  "type": "generation"},
        {"id": "Sufficient?\n(≥3 chunks>0.5)",   "pos": (5.0, 3.9),  "type": "decision"},
        {"id": "Knowledge Filter\n(keep ≥0.5)",  "pos": (8.0, 2.5),  "type": "correction"},
        {"id": "LLM: Rewrite into\n3 Sub-queries","pos": (2.0, 2.5), "type": "generation"},
        {"id": "Re-retrieve\n(per sub-query)",   "pos": (2.0, 1.2),  "type": "retrieval"},
        {"id": "LLM\n(claude-sonnet-4-6)",       "pos": (6.5, 0.6),  "type": "generation"},
        {"id": "Answer (cited)",                 "pos": (8.8, 0.6),  "type": "output"},
    ]
    edges = [
        {"src": "Query",                           "dst": "Initial Retrieval\n(hybrid k=20)", "label": ""},
        {"src": "Initial Retrieval\n(hybrid k=20)","dst": "LLM: Grade\nRelevance (0–1)",     "label": "top-8 chunks"},
        {"src": "LLM: Grade\nRelevance (0–1)",     "dst": "Sufficient?\n(≥3 chunks>0.5)",    "label": "scores"},
        {"src": "Sufficient?\n(≥3 chunks>0.5)",    "dst": "Knowledge Filter\n(keep ≥0.5)",   "label": "yes"},
        {"src": "Sufficient?\n(≥3 chunks>0.5)",    "dst": "LLM: Rewrite into\n3 Sub-queries","label": "no (weak)"},
        {"src": "LLM: Rewrite into\n3 Sub-queries","dst": "Re-retrieve\n(per sub-query)",    "label": "sub-queries"},
        {"src": "Re-retrieve\n(per sub-query)",    "dst": "LLM\n(claude-sonnet-4-6)",        "label": "new candidates"},
        {"src": "Knowledge Filter\n(keep ≥0.5)",   "dst": "LLM\n(claude-sonnet-4-6)",        "label": "context"},
        {"src": "LLM\n(claude-sonnet-4-6)",        "dst": "Answer (cited)",                  "label": ""},
    ]
    draw_diagram("corrective_rag", "Corrective RAG (Graded Retrieval + Sub-query)", nodes, edges)


def agentic_rag():
    nodes = [
        {"id": "Query",                            "pos": (5.0, 8.3),  "type": "input"},
        {"id": "LLM: Plan\n2-3 Sub-queries",       "pos": (5.0, 6.9),  "type": "agent"},
        {"id": "Hybrid Retrieval\n(per sub-query)", "pos": (5.0, 5.5), "type": "retrieval"},
        {"id": "Merge & Deduplicate\nChunks",       "pos": (5.0, 4.1), "type": "intermediate"},
        {"id": "LLM: Reflect\n(sufficient?)",       "pos": (5.0, 2.8), "type": "agent"},
        {"id": "Evidence\nSufficient?",             "pos": (5.0, 1.5), "type": "decision"},
        {"id": "Follow-up\nQueries (≤2)",           "pos": (8.2, 2.8), "type": "agent"},
        {"id": "Top-K Chunks\n(ranked by score)",   "pos": (2.2, 0.5), "type": "retrieval"},
        {"id": "LLM\n(claude-sonnet-4-6)",          "pos": (5.5, 0.5), "type": "generation"},
        {"id": "Answer (cited)",                    "pos": (8.5, 0.5), "type": "output"},
    ]
    edges = [
        {"src": "Query",                             "dst": "LLM: Plan\n2-3 Sub-queries",       "label": ""},
        {"src": "LLM: Plan\n2-3 Sub-queries",        "dst": "Hybrid Retrieval\n(per sub-query)", "label": "sub-queries"},
        {"src": "Hybrid Retrieval\n(per sub-query)", "dst": "Merge & Deduplicate\nChunks",       "label": "chunks"},
        {"src": "Merge & Deduplicate\nChunks",       "dst": "LLM: Reflect\n(sufficient?)",       "label": ""},
        {"src": "LLM: Reflect\n(sufficient?)",       "dst": "Evidence\nSufficient?",             "label": "assessment"},
        {"src": "Evidence\nSufficient?",             "dst": "Follow-up\nQueries (≤2)",           "label": "no"},
        {"src": "Follow-up\nQueries (≤2)",           "dst": "Hybrid Retrieval\n(per sub-query)", "label": "retry"},
        {"src": "Evidence\nSufficient?",             "dst": "Top-K Chunks\n(ranked by score)",   "label": "yes"},
        {"src": "Top-K Chunks\n(ranked by score)",   "dst": "LLM\n(claude-sonnet-4-6)",          "label": "context"},
        {"src": "LLM\n(claude-sonnet-4-6)",          "dst": "Answer (cited)",                    "label": ""},
    ]
    draw_diagram("agentic_rag", "Agentic RAG (Multi-step Reflection Loop)", nodes, edges)


def graph_rag():
    nodes = [
        {"id": "Query",                        "pos": (5.0, 8.2),  "type": "input"},
        {"id": "BM25 Score",                   "pos": (2.5, 6.7),  "type": "retrieval"},
        {"id": "Embed Query",                  "pos": (7.5, 6.7),  "type": "retrieval"},
        {"id": "Hybrid Fusion",                "pos": (5.0, 5.3),  "type": "retrieval"},
        {"id": "Community\nSummaries (top-k)", "pos": (2.5, 3.8),  "type": "retrieval"},
        {"id": "Entity Subgraph\n(neighbourhood)","pos": (7.5, 3.8),"type": "retrieval"},
        {"id": "Merge Context",                "pos": (5.0, 2.4),  "type": "intermediate"},
        {"id": "LLM\n(claude-sonnet-4-6)",     "pos": (5.0, 1.0),  "type": "generation"},
        {"id": "Answer (cited)",               "pos": (1.5, 1.0),  "type": "output"},
    ]
    edges = [
        {"src": "Query",                          "dst": "BM25 Score",                    "label": "keywords"},
        {"src": "Query",                          "dst": "Embed Query",                   "label": "embed"},
        {"src": "BM25 Score",                     "dst": "Hybrid Fusion",                 "label": "sparse scores"},
        {"src": "Embed Query",                    "dst": "Hybrid Fusion",                 "label": "dense scores"},
        {"src": "Hybrid Fusion",                  "dst": "Community\nSummaries (top-k)",  "label": "community search"},
        {"src": "Hybrid Fusion",                  "dst": "Entity Subgraph\n(neighbourhood)","label": "entity expansion"},
        {"src": "Community\nSummaries (top-k)",   "dst": "Merge Context",                 "label": ""},
        {"src": "Entity Subgraph\n(neighbourhood)","dst": "Merge Context",                "label": ""},
        {"src": "Merge Context",                  "dst": "LLM\n(claude-sonnet-4-6)",      "label": "context"},
        {"src": "LLM\n(claude-sonnet-4-6)",       "dst": "Answer (cited)",                "label": ""},
    ]
    draw_diagram("graph_rag", "Graph RAG (Community + Entity Subgraph)", nodes, edges)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Generating RAG pipeline diagrams…")
    for fn in (plain_rag, rerank_rag, hyde_rag, corrective_rag, agentic_rag, graph_rag):
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nDone.")

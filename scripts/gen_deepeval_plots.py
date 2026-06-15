"""Generate one clean pipeline diagram per DeepEval metric.

Pure-PIL renderer (no plot engine / no force-directed layout). Each metric is
drawn as a numbered step pipeline on a left rail, wrapped in the AI & SAP
Consulting brand frame: logo top-left header + the LinkedIn-carousel footer.

The layout is fully deterministic — cards are full-width rows, sub-items are
chips *inside* a card, and the only connectors are the vertical rail arrows in
the gaps between cards. Nothing is force-placed, so no label can overlap a line,
box, arrow, or another label.

Outputs PNG to docs/deepeval/.  Run with the plot-venv python (has Pillow):
    /home/pauloesterwitz/.local/share/plot-venv/bin/python scripts/gen_deepeval_plots.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("/home/pauloesterwitz/Bosch/RAG PoCs/docs/deepeval")
OUT.mkdir(parents=True, exist_ok=True)
LOGO_PATH = Path("/home/pauloesterwitz/AI & SAP Consulting/website/src/assets/logo.png")

# ── Brand palette (ai-sap-consulting-paul-oesterwitz) ───────────────
DARK        = (34, 42, 58)      # #222A3A  inputs
MAGENTA     = (181, 23, 158)    # #B5179E  LLM steps / accent
TEAL        = (8, 145, 178)     # #0891B2  intermediate
GREEN       = (5, 150, 105)     # #059669  score
AMBER       = (217, 119, 6)     # #D97706  weighting
MUTED       = (107, 114, 128)
RULE        = (220, 218, 235)
WHITE       = (255, 255, 255)
CARD_BG     = (248, 246, 251)
CARD_BORDER = (228, 224, 238)
RAIL        = (210, 206, 226)

W      = 1600
MARGIN = 56
CARD_X0 = 156           # cards start right of the rail
CARD_X1 = W - MARGIN
PAD     = 30            # inner card padding
TEXT_X0 = CARD_X0 + PAD
TEXTW   = CARD_X1 - PAD - TEXT_X0
SPINE_X = 92
BADGE_R = 36

# ── Fonts ───────────────────────────────────────────────────────────
FONT_BOLD = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
FONT_REG  = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]


def font(size, bold=False):
    for fp in (FONT_BOLD if bold else FONT_REG):
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(50, bold=True)
F_SUB   = font(31)
F_TAG   = font(34, bold=True)
F_KICK  = font(26, bold=True)
F_STEP  = font(37, bold=True)
F_BODY  = font(30)
F_CHIP  = font(25, bold=True)
F_BADGE = font(40, bold=True)
F_NAME  = font(38, bold=True)
F_URL   = font(30)


def measure(draw, text, fnt):
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw, text, fnt, maxw):
    lines, cur = [], []
    for word in text.split():
        if measure(draw, " ".join(cur + [word]), fnt)[0] <= maxw:
            cur.append(word)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def chip_rows(draw, chips, maxw):
    """Pack chips into rows that fit maxw. Returns list of rows; each row is a
    list of (text, chip_width)."""
    rows, cur, cur_w = [], [], 0
    for c in chips:
        cw = measure(draw, c, F_CHIP)[0] + 32
        if cur and cur_w + 14 + cw > maxw:
            rows.append(cur)
            cur, cur_w = [], 0
        cur.append((c, cw))
        cur_w += (14 if cur_w else 0) + cw
    if cur:
        rows.append(cur)
    return rows


CHIP_H, CHIP_GAP, CHIP_ROW_GAP = 52, 14, 12


def card_height(draw, step):
    h = PAD + 32                                   # kicker line
    h += len(wrap(draw, step["title"], F_STEP, TEXTW)) * 45
    if step.get("body"):
        h += 8 + len(wrap(draw, step["body"], F_BODY, TEXTW)) * 38
    if step.get("chips"):
        rows = chip_rows(draw, step["chips"], TEXTW)
        h += 16 + len(rows) * CHIP_H + (len(rows) - 1) * CHIP_ROW_GAP
    h += PAD
    return max(h, 152)


def rr(draw, box, radius, fill, outline=None, ow=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=ow)


def render(metric):
    steps = metric["steps"]
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    heights = [card_height(probe, s) for s in steps]

    header_h = 150
    title_y  = header_h + 34
    sub_y    = title_y + 66
    cards_top = sub_y + 58
    gap = 40
    footer_h = 156

    body_h = sum(heights) + gap * (len(steps) - 1)
    H = cards_top + body_h + 44 + footer_h

    img = ImageDraw.Draw(Image.new("RGB", (1, 1)))  # placeholder
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    # ── card geometry (centers for the rail) ────────────────────────
    centers = []
    y = cards_top
    for h in heights:
        centers.append(y + h // 2)
        y += h + gap

    # ── rail (drawn first, behind badges) ───────────────────────────
    draw.line([(SPINE_X, centers[0]), (SPINE_X, centers[-1])], fill=RAIL, width=4)
    for a, b in zip(centers, centers[1:]):
        my = (a + b) // 2                          # arrowhead in the gap
        draw.polygon([(SPINE_X - 9, my - 4), (SPINE_X + 9, my - 4), (SPINE_X, my + 9)],
                     fill=RAIL)

    # ── cards ───────────────────────────────────────────────────────
    y = cards_top
    for idx, (step, h) in enumerate(zip(steps, heights), start=1):
        col = step["color"]
        rr(draw, [CARD_X0, y, CARD_X1, y + h], 18, CARD_BG, outline=CARD_BORDER, ow=2)
        # colored accent strip at the card's left inner edge
        rr(draw, [CARD_X0, y + 16, CARD_X0 + 8, y + h - 16], 4, col)

        ty = y + PAD
        draw.text((TEXT_X0, ty), step["kicker"], font=F_KICK, fill=col)
        ty += 32
        for ln in wrap(draw, step["title"], F_STEP, TEXTW):
            draw.text((TEXT_X0, ty), ln, font=F_STEP, fill=DARK)
            ty += 45
        if step.get("body"):
            ty += 8
            for ln in wrap(draw, step["body"], F_BODY, TEXTW):
                draw.text((TEXT_X0, ty), ln, font=F_BODY, fill=MUTED)
                ty += 38
        if step.get("chips"):
            ty += 16
            for row in chip_rows(draw, step["chips"], TEXTW):
                cx = TEXT_X0
                for text, cw in row:
                    rr(draw, [cx, ty, cx + cw, ty + CHIP_H], CHIP_H // 2, WHITE,
                       outline=col, ow=2)
                    draw.text((cx + cw // 2, ty + CHIP_H // 2), text,
                              font=F_CHIP, fill=col, anchor="mm")
                    cx += cw + CHIP_GAP
                ty += CHIP_H + CHIP_ROW_GAP

        # numbered badge on the rail
        cy = y + h // 2
        draw.ellipse([SPINE_X - BADGE_R, cy - BADGE_R, SPINE_X + BADGE_R, cy + BADGE_R],
                     fill=col)
        draw.text((SPINE_X, cy - 2), str(idx), font=F_BADGE, fill=WHITE, anchor="mm")
        y += h + gap

    # ── header: logo top-left + magenta kicker + rule ───────────────
    if LOGO_PATH.exists():
        logo_h = 84
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo = logo.resize((int(logo.width * logo_h / logo.height), logo_h), Image.LANCZOS)
        canvas.paste(logo, (MARGIN, (header_h - logo_h) // 2), logo)
        draw.text((MARGIN + logo.width + 28, header_h // 2), "DEEPEVAL METRICS",
                  font=F_TAG, fill=MAGENTA, anchor="lm")
    draw.rectangle([(MARGIN, header_h - 2), (W - MARGIN, header_h - 1)], fill=RULE)

    # ── title + subtitle ────────────────────────────────────────────
    draw.text((MARGIN, title_y), f"{metric['name']} — how it's scored",
              font=F_TITLE, fill=DARK)
    draw.text((MARGIN, sub_y), metric["question"], font=F_SUB, fill=MAGENTA)

    # ── footer (identical layout to the LinkedIn carousel) ──────────
    fy = H - footer_h
    draw.rectangle([(MARGIN, fy + 34), (W - MARGIN, fy + 35)], fill=RULE)
    draw.text((MARGIN, fy + 54), "AI & SAP Consulting Paul Oesterwitz",
              font=F_NAME, fill=DARK)
    draw.text((MARGIN, fy + 102), "oesterwitz-consulting.de", font=F_URL, fill=MUTED)

    path = OUT / f"{metric['key']}.png"
    canvas.save(path, "PNG")
    print(f"  saved {path.name}  ({W}x{H})")


# ════════════════════════════════════════════════════════════════════
# Metric specifications
# ════════════════════════════════════════════════════════════════════
METRICS = [
    dict(
        key="answer_relevancy", name="Answer Relevancy",
        question="Does the answer actually address the question?",
        steps=[
            dict(kicker="INPUT", color=DARK,
                 title="Take the generated answer",
                 body="Together with the original user question."),
            dict(kicker="GENERATE", color=MAGENTA,
                 title="LLM drafts hypothetical questions",
                 body="Questions this answer would be a good response to.",
                 chips=["Hyp. question 1", "Hyp. question 2", "Hyp. question 3"]),
            dict(kicker="COMPARE", color=TEAL,
                 title="Cosine similarity vs the real question",
                 body="Embed each hypothetical question and compare it to the user question."),
            dict(kicker="SCORE", color=GREEN,
                 title="Mean similarity  →  0–1",
                 body="Average similarity across the hypothetical questions."),
        ],
    ),
    dict(
        key="faithfulness", name="Faithfulness",
        question="Is every claim in the answer grounded in the retrieved context?",
        steps=[
            dict(kicker="INPUT", color=DARK,
                 title="Take the answer + retrieved context",
                 body="The chunks that were passed to the model."),
            dict(kicker="EXTRACT", color=MAGENTA,
                 title="LLM splits the answer into atomic claims",
                 chips=["Claim 1", "Claim 2", "Claim N"]),
            dict(kicker="VERIFY", color=TEAL,
                 title="Check each claim against the context",
                 body="Is the claim supported by the retrieved chunks?",
                 chips=["supported", "contradicted", "not found"]),
            dict(kicker="SCORE", color=GREEN,
                 title="# supported  ÷  # total claims  →  0–1"),
        ],
    ),
    dict(
        key="contextual_relevancy", name="Contextual Relevancy",
        question="How much of the retrieved context is actually relevant?",
        steps=[
            dict(kicker="INPUT", color=DARK,
                 title="Take the question + retrieved chunks",
                 chips=["Chunk 1", "Chunk 2", "Chunk N"]),
            dict(kicker="JUDGE", color=MAGENTA,
                 title="LLM marks the relevant statements",
                 body="Which statements in the chunks help answer the question?"),
            dict(kicker="SCORE", color=GREEN,
                 title="# relevant  ÷  # total statements  →  0–1"),
        ],
    ),
    dict(
        key="contextual_precision", name="Contextual Precision",
        question="Are the relevant chunks ranked above the irrelevant ones?",
        steps=[
            dict(kicker="INPUT", color=DARK,
                 title="Take the question, expected answer & ranked context",
                 body="The top-k retrieved chunks, in retrieval order."),
            dict(kicker="JUDGE", color=MAGENTA,
                 title="LLM marks each rank relevant or not",
                 chips=["Rank 1 ✓", "Rank 2 ✗", "Rank N ✓"]),
            dict(kicker="WEIGHT", color=AMBER,
                 title="Relevant chunks higher up count more",
                 body="Earlier ranks carry more weight than later ones."),
            dict(kicker="SCORE", color=GREEN,
                 title="Weighted precision@k  →  0–1"),
        ],
    ),
    dict(
        key="contextual_recall", name="Contextual Recall",
        question="Did retrieval capture everything the answer needs?",
        steps=[
            dict(kicker="INPUT", color=DARK,
                 title="Take the expected answer + retrieved context",
                 body="The golden answer plus the chunks retrieved for it."),
            dict(kicker="EXTRACT", color=MAGENTA,
                 title="Split the expected answer into statements",
                 chips=["Statement 1", "Statement 2", "Statement N"]),
            dict(kicker="CHECK", color=TEAL,
                 title="Is each statement attributable to the context?"),
            dict(kicker="SCORE", color=GREEN,
                 title="# covered  ÷  # total statements  →  0–1"),
        ],
    ),
    dict(
        key="correctness_geval", name="Correctness (G-Eval)",
        question="Is the answer factually correct versus the golden answer?",
        steps=[
            dict(kicker="INPUT", color=DARK,
                 title="Take the generated answer + reference answer",
                 body="The golden answer is the ground truth."),
            dict(kicker="REASON", color=MAGENTA,
                 title="LLM evaluates with chain-of-thought",
                 body="Scored against explicit, named criteria.",
                 chips=["Correctness", "Completeness", "Consistency"]),
            dict(kicker="SCORE", color=GREEN,
                 title="Weighted G-Eval score  →  0–1",
                 body="Criteria are rated, then normalised."),
        ],
    ),
]


if __name__ == "__main__":
    print("Rendering DeepEval metric pipelines:")
    for m in METRICS:
        render(m)
    print("Done — 6 branded PNGs in", OUT)

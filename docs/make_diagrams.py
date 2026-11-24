"""Draws docs/architecture.png and docs/split_timeline.png.  Run: python docs/make_diagrams.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
BLUE, TEAL, AMBER, GREY, RED = "#e9ebee", "#dcdfe3", "#f4f4f4", "#ffffff", "#ece4e4"


def box(ax, x, y, w, h, text, fc, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec="#333333", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, text=None, rad=0.0, fs=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.1,
                                 color="#333333", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.16, text, ha="center", fontsize=fs, color="#333333")


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 22); ax.set_ylim(0, 13); ax.axis("off")

    # lane titles
    for y, t in ((11.9, "OFFLINE: training pipeline  (python -m src.train)"),
                 (7.55, "SERVING: online API and batch CLI share one Predictor"),
                 (2.3, "MONITORING and RELIABILITY")):
        ax.text(0.2, y, t, fontsize=9, fontweight="bold", color="#222222")
    ax.plot([0.2, 21.8], [11.75, 11.75], color="#999999", lw=0.8)
    ax.plot([0.2, 21.8], [7.4, 7.4], color="#999999", lw=0.8)
    ax.plot([0.2, 21.8], [2.15, 2.15], color="#999999", lw=0.8)

    # offline lane
    box(ax, 0.3, 8.6, 3.2, 2.5, "paysim.csv\n6.36M rows\n(raw, not shipped)", GREY)
    box(ax, 4.2, 8.6, 4.3, 2.5, "data_pipeline.py\nquality report, scope filter\n(TRANSFER, CASH_OUT),\ntemporal split, leakage audit", BLUE)
    box(ax, 9.2, 8.6, 4.2, 2.5, "features.py\nstateless features +\nColumnTransformer\n(fit on TRAIN only)", BLUE)
    box(ax, 14.1, 8.6, 3.9, 2.5, "train.py\nlogreg baseline,\nLightGBM search,\ncost-based threshold\non VAL", BLUE, fs=7.5)
    box(ax, 18.6, 8.6, 3.1, 2.5, "artifacts/\nmodel_pipeline.joblib\nthreshold.json\nreference_stats.json", AMBER, fs=7.5)
    arrow(ax, (3.5, 9.85), (4.2, 9.85)); arrow(ax, (8.5, 9.85), (9.2, 9.85))
    arrow(ax, (13.4, 9.85), (14.1, 9.85)); arrow(ax, (18.0, 9.85), (18.6, 9.85))

    # serving lane
    box(ax, 0.3, 4.6, 3.0, 2.0, "Client / payment\nswitch\nPOST /predict", GREY)
    box(ax, 4.0, 4.6, 4.2, 2.0, "FastAPI  app.py\nschemas.py (Pydantic)\nrejects bad input: HTTP 422", BLUE)
    box(ax, 4.0, 2.85, 4.2, 1.3, "Batch CLI  predict_batch.py\nchunked CSV, rejects file,\nexit codes 0/1/2", BLUE, fs=7.5)
    box(ax, 9.0, 3.9, 4.6, 2.9, "Predictor  (predictor.py)\nloads artifacts once\nadd_features -> preprocessor\n-> LightGBM -> p(fraud)\nscore >= threshold -> flag", TEAL, bold=False)
    box(ax, 14.4, 5.55, 3.7, 1.25, "mode = model", TEAL)
    box(ax, 14.4, 4.2, 3.7, 1.25, "mode = fallback_rule\n(exact-drain rule)", RED, fs=7.5)
    box(ax, 18.8, 4.9, 2.9, 1.9, "JSON decision\nprobability, is_fraud,\nmode, warnings,\nmodel_version", GREY, fs=7.5)
    arrow(ax, (3.3, 5.6), (4.0, 5.6)); arrow(ax, (8.2, 5.6), (9.0, 5.9))
    arrow(ax, (8.2, 3.4), (9.0, 4.4))
    arrow(ax, (13.6, 6.0), (14.4, 6.15)); arrow(ax, (13.6, 4.6), (14.4, 4.8), "model\nfails", fs=6.5)
    arrow(ax, (18.1, 6.15), (18.8, 5.9)); arrow(ax, (18.1, 4.8), (18.8, 5.2))
    ax.annotate("", xy=(11.3, 6.8), xytext=(20.15, 8.6),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1, connectionstyle="arc3,rad=0.25"))
    ax.text(16.6, 8.05, "loaded at start-up", fontsize=7, color="#333333")

    # monitoring lane
    box(ax, 9.0, 0.2, 4.6, 1.6, "GET /stats and /health\nlatency p50/p95/p99,\n4xx/5xx, fallback count,\nalert rate", BLUE, fs=7.5)
    box(ax, 14.4, 0.2, 3.7, 1.6, "monitoring.py\nPSI / KS vs\nreference_stats", BLUE, fs=7.5)
    box(ax, 18.8, 0.2, 2.9, 1.6, "alert or\nretrain trigger", AMBER, fs=8)
    box(ax, 0.3, 0.2, 8.0, 1.6, "benchmark.py: p50/p95/p99, records/sec\nrobustness.py: edge-case and failure checks\ntests/: schemas, features, split, API, batch, drift", GREY, fs=7.5)
    arrow(ax, (11.3, 3.9), (11.3, 1.8)); arrow(ax, (13.6, 1.0), (14.4, 1.0)); arrow(ax, (18.1, 1.0), (18.8, 1.0))
    fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def timeline():
    fig, ax = plt.subplots(figsize=(9, 2.1))
    ax.set_xlim(0, 743); ax.set_ylim(0, 2); ax.axis("off")
    segs = [(0, 450, "TRAIN  steps 1-450\n2,626,155 rows | 5,057 fraud | 0.19%", BLUE),
            (450, 600, "VAL  451-600\n100,704 | 1,556 | 1.55%", AMBER),
            (600, 743, "TEST  601-743\n43,550 | 1,600 | 3.67%", TEAL)]
    for a, b, t, c in segs:
        ax.add_patch(plt.Rectangle((a, 0.7), b - a, 0.9, fc=c, ec="#333333"))
        ax.text((a + b) / 2, 1.15, t, ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(743, 0.45), xytext=(0, 0.45), arrowprops=dict(arrowstyle="-|>", color="#333333"))
    ax.text(371, 0.12, "time (hourly steps).  No partition sees anything from a later one; fit on train, apply to val/test.",
            ha="center", fontsize=7.5, color="#333333")
    fig.savefig(OUT / "split_timeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    timeline()

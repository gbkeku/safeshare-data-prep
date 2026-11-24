"""Report charts. Each detector keeps the same color in every notebook (color follows the detector, not its rank).
Every chart is shown next to its numbers table, because two of these colors are low-contrast on white."""
from __future__ import annotations

import numpy as np

from .data_prep import LABELS

DETECTOR_COLORS = {"rules_only": "#2a78d6", "original_checkpoint": "#eb6834",
                   "fine_tuned": "#1baf7a", "fine_tuned_rules": "#eda100"}
INK, MUTED, GRID = "#1d2939", "#667085", "#e4e7ec"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def recall_by_category(results, path, metric="recall", title="Recall by category, validation (95% interval)"):
    """Horizontal grouped bars: one group per category, one bar per detector, with interval whiskers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    detectors = [d for d in DETECTOR_COLORS if d in results]
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(LABELS) * max(1, len(detectors)) / 1.6 + 1.4))
    height = 0.8 / max(1, len(detectors))
    for k, name in enumerate(detectors):
        m = [results[name]["metrics"][f"{metric}:{label}"] for label in LABELS]
        y = np.arange(len(LABELS)) + (k - (len(detectors) - 1) / 2) * height
        values = np.nan_to_num([v["value"] for v in m])
        err = np.array([[max(0, v["value"] - v["low"]) for v in m], [max(0, v["high"] - v["value"]) for v in m]])
        ax.barh(y, values, height=height - 0.04, color=DETECTOR_COLORS[name], label=name.replace("_", " "),
                edgecolor="white", linewidth=1)
        ax.errorbar(values, y, xerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for yy, v in zip(y, values):
            if v == 0:                       # a zero bar is invisible, so say it
                ax.text(0.015, yy, "0", va="center", ha="left", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel(metric.replace("_", " "), color=MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.06 * 9 / fig.get_figheight()),
              ncol=len(detectors))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def pr_curves(curves, path, detector="original_checkpoint", chosen=None,
              title="Precision-recall by category (word level, validation)"):
    """Small multiples, one panel per category. curves: {label: word_pr_curve output}; chosen: {label: threshold}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 4
    rows = int(np.ceil(len(LABELS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    for ax, label in zip(axes.flat, LABELS):
        pts = [p for p in curves.get(label, []) if not np.isnan(p["precision"]) and not np.isnan(p["recall"])]
        ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts], color=DETECTOR_COLORS[detector], linewidth=2)
        if chosen and label in chosen:
            c = chosen[label]
            if isinstance(c, dict):                 # exact operating point from the threshold table
                p, t = {"recall": c["recall"], "precision": c["precision"]}, c["threshold"]
            else:                                   # only a threshold: nearest point on the stored curve
                p, t = min(pts, key=lambda q: abs(q["threshold"] - c), default=None), c
            if p:
                ax.plot(p["recall"], p["precision"], "o", markersize=8, color=INK)
                ax.annotate(f"t={t:.2f}", (p["recall"], p["precision"]), textcoords="offset points",
                            xytext=(6, -12), fontsize=8, color=INK)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        values = [v for q in pts for v in (q["recall"], q["precision"])]
        low = max(0.0, np.floor((min(values, default=0) - 0.01) * 50) / 50)       # each panel zooms to its own data
        ax.set_xlim(low, 1.002)
        ax.set_ylim(low, 1.002)
        ax.grid(True, color=GRID, linewidth=0.8)
        _style(ax)
    for ax in list(axes.flat)[len(LABELS):]:
        ax.axis("off")
    fig.supxlabel("recall", color=MUTED)
    fig.supylabel("precision", color=MUTED)
    fig.suptitle(title + " (each panel zoomed to its own range)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)

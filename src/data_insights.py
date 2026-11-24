"""Data Prep evidence for the BERT model, read from the frozen data run (no data is changed or re-created).

Notebook 1 cleaned, grouped and split the data and rendered the screenshots. This module reads what it saved
(audit.json, docs/<split>.jsonl, the screenshot records and summary) and produces the cleanup funnel, the BERT
feature-engineering statistics and the insight charts. Descriptive statistics only: no model is scored, and the
test split is only counted (from the audit), never opened.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data_prep import LABELS
from .plots import GRID, INK, MUTED, _style

SPLIT_COLORS = {"train": "#2a78d6", "validation": "#eda100", "test": "#7d8a9c"}
CATEGORY_NAMES = {"PERSON": "Person", "EMAIL": "Email", "PHONE": "Phone", "ACCOUNT_NUMBER": "Account no.",
                  "PAYMENT_CARD": "Card / CVV", "SSN": "SSN", "PASSWORD": "Password / PIN / key"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    _plt().close(fig)
    return str(path)


# ------------------------------------------------------------------ cleanup

def cleanup_funnel(audit):
    """Row counts from the source to the final splits, with what was removed at each step and why."""
    src, sel, prep = audit["source_splits"], audit["selection"], audit["prepared_splits"]
    rows = [
        ("Source rows (official train + test)", src["train"]["loaded_rows"] + src["test"]["loaded_rows"], None),
        ("Rejected: invalid records", -(src["train"]["rejected_rows"] + src["test"]["rejected_rows"]),
         "out-of-bounds or mismatched spans, empty text, malformed rows"),
        ("Removed: training rows in the same group as a test row", -sel["training_rows_removed_for_test_overlap"],
         "prevents near-copies of test records in training"),
        ("Removed: exact duplicates", -sum(sel["exact_duplicates_removed_by_source"].values()), None),
        ("Removed: conflicting duplicates (same text, different labels)",
         -sum(sel["duplicate_conflict_rows_removed_by_source"].values()), None),
        ("Official test rows not sampled", -sel.get("unused_official_test_rows", 0),
         f"test fixed at {sel['requested_test_documents']:,} documents"),
    ]
    final = sum(prep[s]["documents"] for s in prep)
    gap = final - sum(r[1] for r in rows)
    if gap:                                   # shown, never hidden, if the audit's steps do not add up
        rows.append(("Other adjustments (steps do not add up exactly)", gap, "check audit.json"))
    rows.append(("Final: train + validation + test", final, None))
    table = pd.DataFrame(rows, columns=["step", "rows", "note"])
    reasons = Counter()
    for split in ("train", "test"):
        reasons.update(src[split].get("rejection_reasons", {}))
    rejected = pd.DataFrame(sorted(reasons.items(), key=lambda kv: -kv[1]), columns=["reason", "rows"])
    splits = pd.DataFrame([{"split": s, "documents": prep[s]["documents"], "groups": prep[s]["groups"],
                            "documents_without_sensitive_text": prep[s]["negative_documents"],
                            "share_without_sensitive_text": round(prep[s]["negative_fraction"], 4)}
                           for s in ("train", "validation", "test")])
    leakage = pd.DataFrame([{"key": k, **v} for k, v in sel["cross_split_overlap_counts"].items()])
    return table, rejected, splits, leakage


def funnel_chart(table, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _style(ax)
    steps = table.step.tolist()[::-1]
    values = table.rows.tolist()[::-1]
    colors = ["#1baf7a" if v > 0 and i in (0, len(values) - 1) else "#d0573a" for i, v in enumerate(values)]
    ax.barh(range(len(values)), [abs(v) for v in values], color=colors, height=0.62, edgecolor="white", linewidth=2)
    ax.set_yticks(range(len(values)), steps, fontsize=9)
    for i, v in enumerate(values):
        ax.annotate(f"{v:+,}" if v < 0 else f"{v:,}", (abs(v), i), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("rows", color=MUTED)
    ax.set_xlim(0, max(abs(v) for v in values) * 1.18)
    ax.set_title("Cleanup funnel: source rows to final splits (green = kept, red = removed)", loc="left",
                 fontsize=11, color=INK)
    return _save(fig, path)


# ------------------------------------------------------------------ insights

def entity_table(audit):
    prep = audit["prepared_splits"]
    rows = []
    for s in ("train", "validation", "test"):
        counts, docs = prep[s]["entity_counts"], prep[s]["documents"]
        for label in LABELS:
            rows.append({"split": s, "category": label, "entities": counts.get(label, 0),
                         "per_1000_documents": round(1000 * counts.get(label, 0) / docs, 1)})
    return pd.DataFrame(rows)


def entity_chart(table, path):
    """Entities per 1,000 documents by category and split: the splits have the same mix, which is what a grouped
    random split should give."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _style(ax)
    x = np.arange(len(LABELS))
    for k, s in enumerate(("train", "validation", "test")):
        v = table[table.split == s].set_index("category").loc[list(LABELS), "per_1000_documents"]
        ax.bar(x + (k - 1) * 0.27, v, width=0.27, color=SPLIT_COLORS[s], label=s, edgecolor="white", linewidth=1)
    ax.set_xticks(x, [CATEGORY_NAMES[l] for l in LABELS], fontsize=9)
    ax.set_ylabel("entities per 1,000 documents", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    ax.set_title("Sensitive entities by category and split (same mix in every split)", loc="left", fontsize=11, color=INK)
    return _save(fig, path)


def mix_tables(audit):
    prep = audit["prepared_splits"]
    locale = pd.DataFrame([{"split": s, "locale": k, "documents": v} for s in ("train", "validation", "test")
                           for k, v in prep[s]["locales"].items()])
    domains = pd.DataFrame([{"split": s, "domain": k, "documents": v} for s in ("train", "validation")
                            for k, v in prep[s]["domains"].items()])
    return locale, domains


def domain_chart(domains, path, top=20):
    plt = _plt()
    t = domains[domains.split == "train"].sort_values("documents", ascending=False)
    head = t.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    _style(ax)
    ax.barh(head.domain, head.documents, color="#2a78d6", height=0.65)
    ax.set_xlabel("training documents", color=MUTED)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(f"Training documents by domain: top {top} of {len(t)}", loc="left", fontsize=11, color=INK)
    return _save(fig, path)


# ------------------------------------------------------------------ BERT feature engineering

def bert_features(documents, tokenizer, max_lengths=(256, 512), stride=64):
    """Per document: words, BERT sub-word tokens, and the number of overlapping windows the training code makes
    at each max_length (the same tokenizer call as src/train.windows). Also the word-level BIO label counts:
    each word's first sub-token gets B-/I-/O, later sub-tokens and special tokens are ignored (-100)."""
    from .train import bio_word_labels
    per_doc, bio = [], Counter()
    for doc in documents:
        words = [w["text"] for w in doc["words"]]
        if not words:
            continue
        enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        row = {"words": len(words), "tokens": len(enc["input_ids"])}
        for L in max_lengths:
            s = min(stride, L // 4)
            w = tokenizer(words, is_split_into_words=True, truncation=True, max_length=L, stride=s,
                          return_overflowing_tokens=True)
            row[f"windows_{L}"] = len(w["input_ids"])
        per_doc.append(row)
        bio.update(bio_word_labels(doc))
    df = pd.DataFrame(per_doc)
    labels = pd.DataFrame([{"tag": t, "words": n} for t, n in sorted(bio.items(), key=lambda kv: -kv[1])])
    labels["share"] = labels.words / labels.words.sum()
    return df, labels


def feature_summary(df, max_lengths=(256, 512)):
    rows = [{"measure": "words per document", **_describe(df.words)},
            {"measure": "BERT tokens per document", **_describe(df.tokens)},
            {"measure": "BERT tokens per word", **_describe(df.tokens / df.words)}]
    for L in max_lengths:
        rows.append({"measure": f"windows per document at {L} tokens", **_describe(df[f"windows_{L}"]),
                     "share_needing_more_than_one": round(float((df[f"windows_{L}"] > 1).mean()), 4)})
    return pd.DataFrame(rows)


def _describe(s):
    return {"mean": round(float(s.mean()), 2), "median": float(s.median()), "p95": float(s.quantile(0.95)),
            "max": float(s.max())}


def length_chart(df, path, max_lengths=(256, 512)):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axes:
        _style(ax)
    axes[0].hist(df.tokens, bins=50, color="#2a78d6", edgecolor="white")
    for L, c in zip(max_lengths, ("#eda100", "#d0573a")):
        axes[0].axvline(L - 2, color=c, linewidth=2, linestyle="--", label=f"{L}-token window")
    axes[0].set(xlabel="BERT tokens per document", ylabel="documents")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Training documents: BERT sub-word tokens", loc="left", fontsize=10.5, color=INK)
    k = max(int(df[[f"windows_{L}" for L in max_lengths]].max().max()), 1)
    x = np.arange(1, min(k, 8) + 1)
    for j, (L, c) in enumerate(zip(max_lengths, ("#eda100", "#d0573a"))):
        counts = df[f"windows_{L}"].clip(upper=8).value_counts().reindex(x, fill_value=0) / len(df)
        axes[1].bar(x + (j - 0.5) * 0.38, counts, width=0.38, color=c, label=f"{L} tokens")
    axes[1].set_xticks(x, [str(i) if i < 8 else "8+" for i in x])
    axes[1].set(xlabel="overlapping windows per document", ylabel="share of documents")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title("Windows the training code makes per document", loc="left", fontsize=10.5, color=INK)
    return _save(fig, path)


def bio_chart(labels, path):
    plt = _plt()
    t = labels.copy()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _style(ax)
    colors = ["#9aa4b2" if tag == "O" else ("#2a78d6" if tag.startswith("B-") else "#7fb2ec") for tag in t.tag]
    ax.bar(t.tag, t.words, color=colors, edgecolor="white")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45, labelsize=8.5)
    for xi, (n, sh) in enumerate(zip(t.words, t.share)):
        ax.annotate(f"{sh:.2%}" if sh < 0.5 else f"{sh:.0%}", (xi, n), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color=MUTED)
    ax.set_ylabel("words (log scale)", color=MUTED)
    ax.set_title("Word-level BIO labels in training (grey = ordinary, dark = first word, light = continuation)",
                 loc="left", fontsize=10.5, color=INK)
    return _save(fig, path)


# ------------------------------------------------------------------ screenshots

def ocr_read_by_category(images_summary, splits=("train", "validation")):
    rows = []
    for s in splits:
        for fmt in ("png", "jpeg"):
            for label in LABELS:
                r = images_summary["splits"][s][fmt]["by_label"].get(label, {})
                n = r.get("entities", 0)
                rows.append({"split": s, "format": fmt, "category": label, "entities": n,
                             "located_rate": r.get("located", 0) / n if n else np.nan,
                             "read_exact_rate": r.get("read_exact", 0) / n if n else np.nan})
    return pd.DataFrame(rows)


def ocr_category_chart(table, path):
    plt = _plt()
    t = table[(table.split == "validation")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _style(ax)
    x = np.arange(len(LABELS))
    for k, (fmt, c) in enumerate((("png", "#2a78d6"), ("jpeg", "#eda100"))):
        v = t[t.format == fmt].set_index("category").loc[list(LABELS), "read_exact_rate"]
        ax.bar(x + (k - 0.5) * 0.38, v, width=0.38, color=c, label=fmt.upper(), edgecolor="white")
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.0%}", (xi + (k - 0.5) * 0.38, vi), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, [CATEGORY_NAMES[l] for l in LABELS], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("sensitive spans read exactly", color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("OCR on its own, validation screenshots: passwords are the hardest text to read exactly",
                 loc="left", fontsize=10.5, color=INK)
    return _save(fig, path)


CONDITIONS = ("font", "font_size", "theme", "layout", "scale", "blur")


def rendering_table(records):
    """How often each rendering setting appears, by split (PNG records; each JPEG copies its PNG)."""
    rows = Counter()
    for r in records:
        if r.get("variant", "png") != "png":
            continue
        for field in CONDITIONS:
            rows[(r["split"], field, str(r["condition"].get(field)))] += 1
    return pd.DataFrame([{"split": s, "setting": f, "value": v, "screenshots": n}
                         for (s, f, v), n in sorted(rows.items())])


def rendering_chart(table, path):
    plt = _plt()
    t = table[table.split == "train"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, field in zip(axes.flat, CONDITIONS):
        _style(ax)
        sub = t[t.setting == field]
        ax.bar(sub.value, sub.screenshots, color="#2a78d6", edgecolor="white")
        ax.set_title(field.replace("_", " "), loc="left", fontsize=10, color=INK)
        ax.tick_params(axis="x", labelsize=8.5, rotation=20)
    fig.suptitle("Screenshot rendering settings in the training sample (randomized per document)", x=0.01,
                 ha="left", fontsize=11, color=INK)
    return _save(fig, path)

"""Render selected documents as screenshots (PNG and JPEG), run Tesseract, and save OCR output next to the gold labels.

Writes to output_dir:
  figures/*.png                  OCR confidence histogram, error by font and size, sample pages with boxes drawn
  images/<split>/*.png, *.jpg     the screenshots
  records/<split>.jsonl           one row per image: OCR words, gold (reference text, word boxes, entities), error counts
  summary.json                    counts, CER/WER (overall and by visual condition), warnings, selected document ids
  manifest.json                   file hashes (what the GCS upload checks)
  ATTRIBUTION.txt

Visual conditions (font, size, layout, theme, scale, blur, JPEG quality) are chosen from a seeded hash of the document
ID, never from its labels. One font and one layout are held out of train and validation and appear only in test.
"""
from __future__ import annotations

import csv
import heapq
import io
import json
import math
import os
import shutil
import subprocess
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rapidfuzz.distance import Levenshtein

from .data_prep import LABELS, SPLITS, _hash
from .gcs_io import sha256_file, write_manifest

SCHEMA_VERSION = "safeshare-images-v2"
DEFAULTS = {
    "seed": 42, "train_size": 200, "validation_size": 50, "test_size": 50,
    "min_entities_per_label": 100,      # per split, where the pool allows; shortfalls are reported
    "negative_fraction": 0.15,          # share of documents with no sensitive text, to measure over-masking
    "image_width": 1280, "image_height": 960, "margin": 48, "line_spacing": 1.4,
    "font_sizes": [14, 18, 22, 28], "jpeg_qualities": [40, 60, 75, 90],
    "fonts": {"sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
              "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"},
    "test_only_font": "mono", "test_only_layout": "ticket",
    "layouts": ["dialog", "log", "chat", "form", "ticket"],
    "themes": ["light", "dark"], "scale_factors": [1.0, 0.85], "blur_radii": [0.0, 0.5],
    "ocr_language": "eng", "tesseract_psm": 6, "ocr_timeout_seconds": 30,
}
LAYOUT_MARGIN = {"dialog": 36, "log": 0, "chat": 28, "form": 20, "ticket": 40}   # extra margin per layout
CONDITIONS = ("font", "layout", "theme", "font_size", "scale", "blur")


def _config(config=None):
    unknown = set(config or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown image setting(s): {sorted(unknown)}")
    cfg = {**deepcopy(DEFAULTS), **(config or {})}
    if not 0 <= cfg["negative_fraction"] < 0.9:
        raise ValueError("negative_fraction must be at least 0 and below 0.9.")
    if cfg["test_only_font"] not in cfg["fonts"] or cfg["test_only_layout"] not in cfg["layouts"]:
        raise ValueError("test_only_font and test_only_layout must be listed in fonts and layouts.")
    if len(cfg["fonts"]) < 2 or len(cfg["layouts"]) < 2:
        raise ValueError("Need at least two fonts and two layouts, so train keeps one after one is held out.")
    unknown_layouts = set(cfg["layouts"]) - set(LAYOUT_MARGIN)
    if unknown_layouts:
        raise ValueError(f"Unknown layout(s): {sorted(unknown_layouts)}")
    return cfg


# ------------------------------------------------------------------ choosing documents

def _select_balanced(pool, size, seed, min_per_label, negative_fraction):
    """Pick `size` documents so each label reaches `min_per_label` entities where the pool allows.

    Greedy: take the document that closes the most unmet need, weighting each label by 1 / its supply so
    rare labels (SSN, PASSWORD) go first. A share of entity-free documents is reserved. Deterministic: ties
    break on a seeded hash. With min_per_label=0 this is a seeded random sample with the same negative share.
    """
    def key(row):
        return (_hash(f"image-sample:{seed}:{row['id']}"), row["id"])

    order = sorted(pool, key=key)
    negatives = [row for row in order if not row["entities"]]
    positives = [row for row in order if row["entities"]]
    n_neg = min(len(negatives), round(size * negative_fraction))
    budget = min(size - n_neg, len(positives))

    counts = [Counter(e["label"] for e in row["entities"]) for row in positives]
    supply = Counter()
    for c in counts:
        supply.update(c)
    need = {label: min_per_label for label in LABELS}

    def score(i):
        return sum(min(n, need[label]) / supply[label] for label, n in counts[i].items() if need[label] > 0)

    heap = [(-score(i), i) for i in range(len(positives))]     # scores only fall, so stale entries are upper bounds
    heapq.heapify(heap)
    chosen = []
    while heap and len(chosen) < budget:
        _, i = heapq.heappop(heap)
        fresh = score(i)
        if fresh <= 0:
            continue
        if heap and fresh < -heap[0][0]:
            heapq.heappush(heap, (-fresh, i))
            continue
        chosen.append(i)
        for label, n in counts[i].items():
            need[label] = max(0, need[label] - n)
    taken = set(chosen)
    chosen += [i for i in range(len(positives)) if i not in taken][:budget - len(chosen)]

    documents = sorted([positives[i] for i in chosen] + negatives[:n_neg], key=key)
    achieved = Counter(e["label"] for row in documents for e in row["entities"])
    return documents, {label: min_per_label - achieved.get(label, 0) for label in LABELS
                       if achieved.get(label, 0) < min_per_label}


def _select(prepared_dir, cfg, expected_hash):
    """Choose the documents for every split. Returns (documents by split, text manifest hash, shortfalls by split)."""
    source = Path(prepared_dir)
    manifest_hash = sha256_file(source / "manifest.json")
    if expected_hash and manifest_hash != expected_hash:
        raise ValueError("The text bundle does not match the expected manifest hash.")
    selected, shortfalls = {}, {}
    for split in SPLITS:
        lines = (source / "docs" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        pool = [json.loads(line) for line in lines if line.strip()]
        size = cfg[f"{split}_size"]
        if len(pool) < size:
            raise ValueError(f"Only {len(pool)} {split} documents exist; {size} requested.")
        selected[split], shortfalls[split] = _select_balanced(
            pool, size, cfg["seed"], cfg["min_entities_per_label"], cfg["negative_fraction"])
    return selected, manifest_hash, shortfalls


def preview_selection(prepared_dir, config=None, expected_text_manifest_sha256=None):
    """Count what would be rendered, without rendering or OCR. Use it to size the splits first."""
    selected, _, shortfalls = _select(prepared_dir, _config(config), expected_text_manifest_sha256)
    return {split: {"documents": len(docs), "negative_documents": sum(not d["entities"] for d in docs),
                    "entities": {label: sum(e["label"] == label for d in docs for e in d["entities"]) for label in LABELS},
                    "shortfalls": shortfalls[split]}
            for split, docs in selected.items()}


# ------------------------------------------------------------------ drawing pages

def _load_fonts(cfg, documents):
    """Open every (font, size) pair once. Every character in the documents must exist in every font."""
    need = {ord(c) for doc in documents for w in doc["words"] for c in w["text"]}
    fonts = {}
    for name, path in cfg["fonts"].items():
        if not Path(path).is_file():
            raise ValueError(f"Font not found: {path}")
        with TTFont(path, lazy=True) as f:
            missing = need - set(f.getBestCmap() or {})
        if missing:
            raise ValueError(f"The {name} font lacks characters: " + ", ".join(f"U+{c:04X}" for c in sorted(missing)[:20]))
        for size in cfg["font_sizes"]:
            fonts[(name, size)] = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)
    return fonts


def _condition(doc, split, cfg):
    """Seeded visual conditions, independent of entity labels. Test also gets the held-out font and layout."""
    n = int(_hash(f"condition:{cfg['seed']}:{doc['id']}"), 16)
    fonts, layouts = list(cfg["fonts"]), list(cfg["layouts"])
    if split != "test":
        fonts = [x for x in fonts if x != cfg["test_only_font"]]
        layouts = [x for x in layouts if x != cfg["test_only_layout"]]
    elif n % 4 == 0:
        fonts = [cfg["test_only_font"]]
    elif n % 4 == 1:
        layouts = [cfg["test_only_layout"]]
    elif n % 4 == 2:
        fonts, layouts = [cfg["test_only_font"]], [cfg["test_only_layout"]]

    def pick(items, shift):
        return items[(n >> shift) % len(items)]
    return {"font": pick(fonts, 4), "font_size": pick(cfg["font_sizes"], 12), "layout": pick(layouts, 20),
            "theme": pick(cfg["themes"], 28), "scale": pick(cfg["scale_factors"], 36),
            "blur": pick(cfg["blur_radii"], 44), "jpeg_quality": pick(cfg["jpeg_qualities"], 52)}


def _render(document, font, cfg, condition):
    """Lay words out left to right, wrap lines, start a new page when full.

    Returns pages: {image, fragments (word pieces with pixel boxes), reference_text}. Only the words and
    their spacing decide the layout, never the labels.
    """
    width, height, layout = cfg["image_width"], cfg["image_height"], condition["layout"]
    margin = cfg["margin"] + LAYOUT_MARGIN[layout]
    dark = condition["theme"] == "dark"
    background, foreground = ("#19212d", "#f2f5f9") if dark else ("#ffffff", "#1d2939")
    border = "#697789" if dark else "#bac5d1"
    ascent, descent = font.getmetrics()
    line_h = max(math.ceil(condition["font_size"] * cfg["line_spacing"]), ascent + descent + 2)
    space, usable = math.ceil(font.getlength(" ")), width - 2 * margin
    pages, cur = [], {}

    def start_page():
        image = Image.new("RGB", (width, height), background)
        frame = ImageDraw.Draw(image)
        cur.update(image=image, draw=frame, fragments=[], lines=[[]], x=margin, y=margin + ascent)
        if layout != "log":
            frame.rounded_rectangle([margin - 16, margin - 16, width - margin + 16, height - margin + 16],
                                    radius=14 if layout in ("dialog", "chat") else 2, outline=border, width=2)
        if layout in ("ticket", "dialog"):
            frame.line([margin, margin + 8, width - margin, margin + 8], fill=border, width=2)
            cur["y"] += 28
        if layout == "chat":
            frame.line([width // 2, margin, width // 2, height - margin], fill=border, width=1)

    def end_page():
        if cur["fragments"]:
            pages.append({"image": cur["image"], "fragments": cur["fragments"],
                          "reference_text": "\n".join(" ".join(line) for line in cur["lines"] if line)})

    def newline():
        cur["x"], cur["y"] = margin, cur["y"] + line_h
        if cur["y"] + descent > height - margin:
            end_page()
            start_page()
        else:
            cur["lines"].append([])

    def draw(word, a, b):
        part, origin = word["text"][a:b], (cur["x"], cur["y"])
        left, top, right, bottom = cur["draw"].textbbox(origin, part, font=font, anchor="ls")
        box = [math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError("A word would be clipped by the image edge. Increase the margin or image size.")
        cur["draw"].text(origin, part, font=font, anchor="ls", fill=foreground)
        cur["fragments"].append({"id": len(cur["fragments"]), "word_id": word["id"], "text": part, "bbox": box})
        cur["lines"][-1].append(part)
        cur["x"] = max(origin[0] + math.ceil(font.getlength(part)), box[2]) + space

    start_page()
    previous_end = 0
    for word in document["words"]:
        text = word["text"]
        if "\n" in document["text"][previous_end:word["start"]] and cur["lines"][-1]:
            newline()
        previous_end = word["end"]
        if font.getlength(text) <= usable:
            if cur["x"] + font.getlength(text) > width - margin:
                newline()
            draw(word, 0, len(text))
        else:                                   # wider than a page: split into pieces that fit
            if cur["lines"][-1]:
                newline()
            a = 0
            while a < len(text):
                b = a + 1
                while b < len(text) and (font.getlength(text[a:b + 1]) <= usable or unicodedata.combining(text[b])):
                    b += 1
                draw(word, a, b)
                a = b
                if a < len(text):
                    newline()
    end_page()
    return pages


def _transform(image, condition):
    """Scale and blur the rendered page. Gold boxes are scaled by the same factor (see _gold)."""
    result = image
    if condition["scale"] != 1:
        result = result.resize((round(image.width * condition["scale"]), round(image.height * condition["scale"])),
                               Image.Resampling.LANCZOS)
    if condition["blur"]:
        result = result.filter(ImageFilter.GaussianBlur(condition["blur"]))
    return result


# ------------------------------------------------------------------ OCR

def _engine_info(cfg):
    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError("Install tesseract-ocr and tesseract-ocr-eng first.")
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15)
    return {"name": "tesseract", "version": (out.stdout or out.stderr).splitlines()[0],
            "language": cfg["ocr_language"], "psm": cfg["tesseract_psm"], "oem": 1}


def _ocr(path, cfg):
    """Run Tesseract on one image. Returns {status: ok|empty|error, words: [{id, text, bbox, confidence}]}."""
    try:
        run = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", cfg["ocr_language"], "--oem", "1",
             "--psm", str(cfg["tesseract_psm"]), "tsv"],
            capture_output=True, text=True, encoding="utf-8", timeout=cfg["ocr_timeout_seconds"],
            env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        if run.returncode:
            raise RuntimeError(run.stderr.strip()[:300])
        words = []
        for row in csv.DictReader(io.StringIO(run.stdout), delimiter="\t", quoting=csv.QUOTE_NONE):
            text = (row.get("text") or "").strip()
            if int(row["level"]) == 5 and text:
                left, top, w, h = (int(row[k]) for k in ("left", "top", "width", "height"))
                words.append({"id": len(words), "text": text, "bbox": [left, top, left + w, top + h],
                              "confidence": float(row["conf"])})
    except Exception as error:      # a failed image stays in the data, counted as a failure
        return {"status": "error", "error": f"{type(error).__name__}: {error}"[:300], "words": []}
    return {"status": "ok" if words else "empty", "error": None, "words": words}


# ------------------------------------------------------------------ gold labels and scores

def _gold(document, page, scale):
    """Reference text, word-piece boxes and entity boxes for one page, scaled to the saved image."""
    def box(b):
        return [round(v * scale) for v in b]
    entities = []
    for i, e in enumerate(document["entities"]):
        parts = [f for f in page["fragments"] if e["start_word"] <= f["word_id"] <= e["end_word"]]
        if parts:       # an entity on another page has no boxes here
            entities.append({**e, "entity_id": f"{document['id']}:entity-{i:04d}",
                             "boxes": [box(f["bbox"]) for f in parts], "texts": [f["text"] for f in parts]})
    return {"reference_text": page["reference_text"], "entities": entities,
            "fragments": [{**f, "bbox": box(f["bbox"])} for f in page["fragments"]]}


def _hits(box, words, pad=3):
    """OCR words whose centre lies inside a gold box (grown by pad pixels), left to right."""
    found = [w for w in words if box[0] - pad <= (w["bbox"][0] + w["bbox"][2]) / 2 <= box[2] + pad
             and box[1] - pad <= (w["bbox"][1] + w["bbox"][3]) / 2 <= box[3] + pad]
    return sorted(found, key=lambda w: w["bbox"][0])


def _entity_read(entity, words):
    """(located, read_exactly): every word piece of the entity has an OCR word on it / their text equals the original."""
    located = exact = True
    for box, text in zip(entity["boxes"], entity["texts"]):
        hits = _hits(box, words)
        located &= bool(hits)
        exact &= "".join(w["text"] for w in hits) == text
    return located, exact


def _tally(gold, ocr):
    """Counts for one image: edit distances, word recall, and per-label sensitive text located / read exactly."""
    ref = " ".join(gold["reference_text"].split())
    hyp = " ".join(" ".join(w["text"] for w in ocr["words"]).split())
    tally = Counter({"images": 1, "ocr_failures": ocr["status"] == "error", "ocr_empty": ocr["status"] == "empty",
                     "ref_chars": len(ref), "char_edits": Levenshtein.distance(ref, hyp),
                     "ref_words": len(ref.split()), "word_edits": Levenshtein.distance(ref.split(), hyp.split()),
                     "words_found": sum((Counter(ref.split()) & Counter(hyp.split())).values())})
    for entity in gold["entities"]:
        located, exact = _entity_read(entity, ocr["words"])
        tally[f"n:{entity['label']}"] += 1
        tally[f"located:{entity['label']}"] += located
        tally[f"exact:{entity['label']}"] += exact
    return tally


def _rates(c, by_label=False):
    total = sum(c[f"n:{label}"] for label in LABELS)
    rates = {"images": c["images"], "ocr_failures": c["ocr_failures"], "ocr_empty": c["ocr_empty"],
             "cer": c["char_edits"] / c["ref_chars"] if c["ref_chars"] else None,
             "wer": c["word_edits"] / c["ref_words"] if c["ref_words"] else None,
             "word_recall": c["words_found"] / c["ref_words"] if c["ref_words"] else None,
             "sensitive_entities": total,
             "sensitive_located_rate": sum(c[f"located:{l}"] for l in LABELS) / total if total else None,
             "sensitive_read_exact_rate": sum(c[f"exact:{l}"] for l in LABELS) / total if total else None}
    if by_label:
        rates["by_label"] = {label: {"entities": c[f"n:{label}"], "located": c[f"located:{label}"],
                                     "read_exact": c[f"exact:{label}"]} for label in LABELS if c[f"n:{label}"]}
    return rates


def create_ocr_report(root, cfg):
    """OCR charts from the saved records, written to <root>/figures/: confidence histogram, CER/WER by font and
    size, and sample pages with the boxes drawn (red = sensitive text from the labels, blue = words OCR found)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(root)
    (root / "figures").mkdir(exist_ok=True)
    records = [json.loads(line) for split in SPLITS
               for line in (root / "records" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    png = [r for r in records if r["variant"] == "png"]

    sensitive, other = [], []
    for r in png:
        boxes = [b for e in r["gold"]["entities"] for b in e["boxes"]]
        for w in r["ocr"]["words"]:
            on_sensitive = any(_hits(b, [w]) for b in boxes)
            (sensitive if on_sensitive else other).append(w["confidence"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, 105, 5)
    ax.hist(other, bins=bins, density=True, alpha=.65, color="#60749a", label=f"other words ({len(other):,})")
    if sensitive:
        ax.hist(sensitive, bins=bins, density=True, alpha=.65, color="#d98c3f", label=f"sensitive words ({len(sensitive):,})")
    ax.set(title="Tesseract word confidence (PNG pages)", xlabel="Confidence (engine score, not a probability)", ylabel="Share of words")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(root / "figures" / "ocr_confidence.png", dpi=150, facecolor="white"); plt.close(fig)

    fonts, sizes = list(cfg["fonts"]), cfg["font_sizes"]
    cells = {}
    for r in png:
        cells.setdefault((r["condition"]["font"], r["condition"]["font_size"]), Counter()).update(r["errors"])
        cells[(r["condition"]["font"], r["condition"]["font_size"])]["images"] += 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (title, edits, ref) in zip(axes, (("Character error rate (%)", "char_edits", "ref_chars"),
                                              ("Word error rate (%)", "word_edits", "ref_words"))):
        grid = np.full((len(fonts), len(sizes)), np.nan)
        for i, font in enumerate(fonts):
            for j, size in enumerate(sizes):
                c = cells.get((font, size))
                if c and c[ref]:
                    grid[i, j] = 100 * c[edits] / c[ref]
        image = ax.imshow(grid, cmap="Oranges", vmin=0)
        ax.set(title=title, xlabel="Font size (px)", xticks=range(len(sizes)), xticklabels=sizes,
               yticks=range(len(fonts)), yticklabels=fonts)
        for i, font in enumerate(fonts):
            for j, size in enumerate(sizes):
                if (font, size) in cells:
                    ax.text(j, i, f"{grid[i, j]:.1f}\nn={cells[(font, size)]['images']}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
    fig.suptitle("OCR error by font and size (PNG pages, all splits)")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(root / "figures" / "ocr_error_by_font_size.png", dpi=150, facecolor="white"); plt.close(fig)

    picks = []
    for split in SPLITS:
        rows = sorted((r for r in png if r["split"] == split and r["gold"]["entities"]), key=lambda r: _hash(r["id"]))
        held = [r for r in rows if r["condition"]["font"] == cfg["test_only_font"]
                or r["condition"]["layout"] == cfg["test_only_layout"]] if split == "test" else []
        picks += (held[:1] + [r for r in rows if r not in held[:1]])[:2]
    for i, r in enumerate(picks):
        image = Image.open(root / r["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for w in r["ocr"]["words"]:
            draw.rectangle(w["bbox"], outline="#1f6feb", width=1)
        for e in r["gold"]["entities"]:
            for b in e["boxes"]:
                draw.rectangle(b, outline="#d1242f", width=3)
        c = r["condition"]
        image.save(root / "figures" / f"sample_{i}_{r['split']}_{c['font']}_{c['layout']}_{c['theme']}.png")
        image.close()


# ------------------------------------------------------------------ main

def prepare_images(prepared_dir, output_dir, config=None, *, ocr_runner=None,
                   expected_text_manifest_sha256=None, progress_every=100):
    """Create screenshots, OCR output and gold labels. Returns {output_dir, manifest, summary}.

    The PNG and JPEG copies of a page are OCR'd in parallel (two Tesseract processes). Pass ocr_runner(path, cfg)
    to replace Tesseract in tests.
    """
    cfg, root = _config(config), Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose a new empty output folder. Existing releases are not overwritten.")
    selected, source_hash, shortfalls = _select(prepared_dir, cfg, expected_text_manifest_sha256)
    source_manifest = json.loads((Path(prepared_dir) / "manifest.json").read_text(encoding="utf-8"))
    fonts = _load_fonts(cfg, [d for docs in selected.values() for d in docs])   # fails now, not an hour in
    engine = {"name": "custom"} if ocr_runner else _engine_info(cfg)
    run_ocr = ocr_runner or _ocr
    (root / "records").mkdir(parents=True)

    summary = {"schema_version": SCHEMA_VERSION, "source_text_manifest_sha256": source_hash, "splits": {},
               "warnings": [], "ocr_engine": engine,
               "fonts": {name: Path(path).name for name, path in cfg["fonts"].items()}, "font_sizes": cfg["font_sizes"],
               "evaluation_policy": {"primary_format": "png", "jpeg_role": "paired robustness",
                                     "test_only_font": cfg["test_only_font"], "test_only_layout": cfg["test_only_layout"]},
               "limitations": ["Synthetic rendered text; real-app screenshots are a separate test set.",
                               "PNG and JPEG are paired copies of the same page, not independent samples.",
                               "CER and WER are case-sensitive, whitespace-normalised edit distances.",
                               "word_recall: share of reference words found in the OCR output (case-sensitive, counted as a bag of words).",
                               "Sensitive text 'located' = every word piece of the entity has an OCR word centred on it; "
                               "'read exact' = those OCR words also spell the original text. Counted per page."]}
    totals, by_condition = {"png": Counter(), "jpeg": Counter()}, {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for split in SPLITS:
            (root / "images" / split).mkdir(parents=True)
            by_format, entity_counts, pages_count = {"png": Counter(), "jpeg": Counter()}, Counter(), 0
            with (root / "records" / f"{split}.jsonl").open("w", encoding="utf-8") as out:
                for done, doc in enumerate(selected[split], start=1):
                    entity_counts.update(e["label"] for e in doc["entities"])
                    condition = _condition(doc, split, cfg)
                    for page_no, page in enumerate(_render(doc, fonts[(condition["font"], condition["font_size"])], cfg, condition)):
                        pages_count += 1
                        pair_id = f"{_hash(doc['id'])[:24]}-p{page_no:04d}"
                        gold = _gold(doc, page, condition["scale"])
                        image = _transform(page["image"], condition)
                        paths = {"png": f"images/{split}/{pair_id}-png.png", "jpeg": f"images/{split}/{pair_id}-jpeg.jpg"}
                        image.save(root / paths["png"], format="PNG")
                        image.save(root / paths["jpeg"], format="JPEG", quality=condition["jpeg_quality"], subsampling=0)
                        results = pool.map(lambda variant: run_ocr(root / paths[variant], cfg), paths)
                        for variant, ocr in zip(paths, results):
                            tally = _tally(gold, ocr)
                            by_format[variant].update(tally)
                            for name in CONDITIONS:
                                by_condition.setdefault(f"{split}|{variant}|{name}={condition[name]}", Counter()).update(tally)
                            out.write(json.dumps({
                                "id": f"{pair_id}-{variant}", "pair_id": pair_id, "variant": variant, "split": split,
                                "document_id": doc["id"], "source_uid": doc["source_uid"], "group_id": doc["group_id"],
                                "source_split": doc["source_split"], "locale": doc["locale"], "page_index": page_no,
                                "image_path": paths[variant], "width": image.width, "height": image.height,
                                "condition": condition, "ocr": ocr, "gold": gold,
                                "errors": {k: tally[k] for k in ("ref_chars", "char_edits", "ref_words", "word_edits")}},
                                sort_keys=True, ensure_ascii=False) + "\n")
                        image.close()
                        page["image"].close()
                    if progress_every and done % progress_every == 0:
                        print(f"{split}: {done}/{len(selected[split])} documents, {time.monotonic() - started:.0f}s", flush=True)
            for variant in totals:
                totals[variant].update(by_format[variant])
            summary["splits"][split] = {
                "documents": len(selected[split]), "groups": len({d["group_id"] for d in selected[split]}),
                "pages": pages_count, "images": sum(c["images"] for c in by_format.values()),
                "negative_documents": sum(not d["entities"] for d in selected[split]),
                "entity_counts": dict(entity_counts), "shortfalls": shortfalls[split],
                "png": _rates(by_format["png"], True), "jpeg": _rates(by_format["jpeg"], True)}
            absent = [label for label in LABELS if not entity_counts[label]]
            if absent:
                summary["warnings"].append(f"{split}: no entities for {', '.join(absent)}.")
            if shortfalls[split]:
                short = ", ".join(f"{k} ({entity_counts[k]})" for k in shortfalls[split])
                summary["warnings"].append(f"{split}: fewer than {cfg['min_entities_per_label']} entities for {short}. "
                                           "Raise the split size or lower min_entities_per_label.")
    summary["overall"] = _rates(sum(totals.values(), Counter()), True)
    summary["ocr_by_condition"] = {key: _rates(value) for key, value in sorted(by_condition.items())}
    if summary["overall"]["ocr_failures"] or summary["overall"]["ocr_empty"]:
        summary["warnings"].append("Some images had OCR failures or no words. They stay in the data and count as errors.")
    summary["selected_ids"] = {s: [d["id"] for d in selected[s]] for s in SPLITS}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (root / "ATTRIBUTION.txt").write_text(
        "Derived from Nemotron-PII, NVIDIA Corporation (2025), CC BY 4.0: https://huggingface.co/datasets/nvidia/Nemotron-PII\n"
        "Changes: document subset, text rendered as PNG/JPEG pages, Tesseract OCR, pixel boxes for entities.\n"
        "Synthetic examples. Not a production privacy guarantee.\n", encoding="utf-8")
    create_ocr_report(root, cfg)
    manifest = write_manifest(root, {
        "schema_version": SCHEMA_VERSION, "config": cfg, "source_text_manifest_sha256": source_hash,
        "source_release_id": source_manifest.get("release_id"), "source_dataset": source_manifest.get("dataset"),
        "splits": {s: {k: summary["splits"][s][k] for k in ("documents", "groups", "pages", "images")} for s in SPLITS},
        "ocr_engine": engine, "preparation_code_sha256": sha256_file(Path(__file__))}, "safeshare-images-")
    return {"output_dir": str(root), "manifest": manifest, "summary": summary}

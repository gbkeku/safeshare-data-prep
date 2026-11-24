"""BERT token classification on SafeShare documents: word-level scores, decoding, and BIO training labels.

The same code runs the original gravitee checkpoint (its 25 entity types mapped onto the seven SafeShare
categories) and, later, the fine-tuned model (its labels are already SafeShare BIO labels).

Documents are already split into words. Each word is tokenized into sub-tokens; a word's score is the
probability on its first sub-token, averaged over the overlapping windows that contain it. Long documents
are covered by windows of max_length tokens that overlap by `stride` tokens.
"""
from __future__ import annotations

import numpy as np

from .data_prep import LABELS

# gravitee-io/bert-small-pii-detection entity -> SafeShare category. Only clear matches are mapped.
# FINANCIAL is left out on purpose: the model card does not define it, so mapping it would be a guess.
GRAVITEE_LABEL_MAP = {
    "PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "CREDIT_CARD": "PAYMENT_CARD",
    "US_SSN": "SSN", "PASSWORD": "PASSWORD", "US_BANK_NUMBER": "ACCOUNT_NUMBER", "IBAN_CODE": "ACCOUNT_NUMBER",
}
SAFESHARE_BIO = ["O"] + [f"{p}-{label}" for label in LABELS for p in ("B", "I")]
IDENTITY_MAP = {label: label for label in LABELS}


def load_model(model_id, revision=None, token=None, device=None):
    """Tokenizer and model in eval mode, on the GPU when there is one."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is needed to map sub-tokens back to words.")
    model = AutoModelForTokenClassification.from_pretrained(model_id, revision=revision, token=token)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model.eval().to(device)


def _split(name):
    """'B-PERSON' -> ('B', 'PERSON'); 'O' -> (None, None); a label without a prefix counts as inside."""
    if name == "O":
        return None, None
    if len(name) > 2 and name[1] == "-" and name[0] in "BIESLU":
        return name[0], name[2:]
    return "I", name


def label_matrices(id2label, label_map):
    """Two (model labels x SafeShare labels) 0/1 matrices: probability mass that starts / continues an entity."""
    begin = np.zeros((len(id2label), len(LABELS)), dtype=np.float32)
    inside = np.zeros_like(begin)
    for index, name in id2label.items():
        prefix, entity = _split(name)
        target = label_map.get(entity)
        if target:
            (begin if prefix in ("B", "S", "U") else inside)[int(index), LABELS.index(target)] = 1
    if not begin.any() and not inside.any():
        raise ValueError("No model label maps to a SafeShare category. Check the label map.")
    return begin, inside


def unmapped_labels(id2label, label_map):
    """Model entity types that are ignored (scored as ordinary text)."""
    return sorted({_split(n)[1] for n in id2label.values() if _split(n)[1] and _split(n)[1] not in label_map})


def word_scores(documents, tokenizer, model, label_map, max_length=None, stride=64, batch_size=32, docs_per_step=64):
    """Yield (document, begin, inside) per document: arrays of shape (n_words, 7) with SafeShare scores."""
    import torch
    begin_m, inside_m = label_matrices(model.config.id2label, label_map)
    begin_t, inside_t = (torch.tensor(m, device=model.device) for m in (begin_m, inside_m))
    max_length = max_length or min(int(tokenizer.model_max_length), int(model.config.max_position_embeddings), 512)
    stride = min(stride, max_length // 4)              # the overlap must be smaller than a window
    for step in range(0, len(documents), docs_per_step):
        chunk = documents[step:step + docs_per_step]
        windows = []                                    # (doc index in chunk, input ids, word id per token)
        for d, doc in enumerate(chunk):
            if not doc["words"]:
                continue
            enc = tokenizer([w["text"] for w in doc["words"]], is_split_into_words=True, truncation=True,
                            max_length=max_length, stride=stride, return_overflowing_tokens=True)
            for i in range(len(enc["input_ids"])):
                windows.append((d, enc["input_ids"][i], enc.word_ids(i)))
        sums = [np.zeros((len(doc["words"]), len(LABELS), 2), dtype=np.float32) for doc in chunk]
        seen = [np.zeros(len(doc["words"]), dtype=np.float32) for doc in chunk]
        for b in range(0, len(windows), batch_size):
            batch = windows[b:b + batch_size]
            padded = tokenizer.pad({"input_ids": [w[1] for w in batch]}, return_tensors="pt")
            with torch.no_grad():
                logits = model(input_ids=padded["input_ids"].to(model.device),
                               attention_mask=padded["attention_mask"].to(model.device)).logits
                probs = torch.softmax(logits.float(), dim=-1)
                pair = torch.stack([probs @ begin_t, probs @ inside_t], dim=-1).cpu().numpy()   # tokens x 7 x 2
            for row, (d, _, word_ids) in enumerate(batch):
                previous = None
                for t, w in enumerate(word_ids):
                    if w is not None and w != previous:          # first sub-token of each word
                        sums[d][w] += pair[row, t]
                        seen[d][w] += 1
                    previous = w
        for d, doc in enumerate(chunk):
            avg = sums[d] / np.maximum(seen[d], 1)[:, None, None]      # words with no sub-token stay at 0
            yield doc, avg[:, :, 0], avg[:, :, 1]


def decode(begin, inside, thresholds=0.5):
    """Word scores -> entities. A word is sensitive when its best category scores at least that category's
    threshold; neighbouring words of the same category join one entity unless the model scores a new start."""
    if not isinstance(thresholds, dict):
        thresholds = {label: float(thresholds) for label in LABELS}
    score = begin + inside
    entities, current = [], None
    for w in range(len(score)):
        j = int(np.argmax(score[w])) if len(score[w]) else 0
        label = LABELS[j]
        if score[w, j] < thresholds[label]:
            current = None
            continue
        if current and current["label"] == label and current["end_word"] == w - 1 and inside[w, j] >= begin[w, j]:
            current["end_word"] = w
        else:
            current = {"start_word": w, "end_word": w, "label": label}
            entities.append(current)
    return entities


def predict(documents, tokenizer, model, label_map, thresholds=0.5, keep_scores=False, **kwargs):
    """Entities for every document (dict id -> list), and optionally the per-word scores (float16) for PR curves."""
    predictions, scores = {}, []
    for doc, begin, inside in word_scores(documents, tokenizer, model, label_map, **kwargs):
        predictions[doc["id"]] = decode(begin, inside, thresholds)
        if keep_scores:
            scores.append((begin + inside).astype(np.float16))
    return predictions, scores


def bio_word_labels(document):
    """Training targets: one SafeShare BIO label per word. Overlapping gold entities keep the first one."""
    labels = ["O"] * len(document["words"])
    for e in sorted(document["entities"], key=lambda e: (e["start_word"], e["end_word"])):
        if all(labels[i] == "O" for i in range(e["start_word"], e["end_word"] + 1)):
            labels[e["start_word"]] = f"B-{e['label']}"
            for i in range(e["start_word"] + 1, e["end_word"] + 1):
                labels[i] = f"I-{e['label']}"
    return labels

"""Data prep in BERT mode: documents and windows only (no chat examples), output verified, charts drawn."""
from __future__ import annotations

import json

from synth import make_docs

SOURCE_LABEL = {"PERSON": "first_name", "EMAIL": "email", "PHONE": "phone_number", "ACCOUNT_NUMBER": "account_number",
                "PAYMENT_CARD": "credit_debit_card", "SSN": "ssn", "PASSWORD": "password"}


def fake_source(n_train=60, n_test=30):
    def rows(prefix, docs):
        out = []
        for i, d in enumerate(docs):
            text = f"{d['text']} Reference {prefix}{i}."
            spans = [{"start": s["start"], "end": s["end"], "label": SOURCE_LABEL[s["label"]],
                      "text": d["text"][s["start"]:s["end"]]} for s in d["spans"]]
            out.append({"uid": f"{prefix}{i}", "text": text, "spans": json.dumps(spans), "domain": d["domain"],
                        "locale": d["locale"], "document_format": d["document_format"]})
        return out
    docs = make_docs(n_train + n_test)
    return {"train": rows("tr", docs[:n_train]), "test": rows("te", docs[n_train:])}


def test_bert_mode(tmp_path, tiny_model_dir):
    from transformers import AutoTokenizer
    from src.data_prep import load_config, prepare_data, verify_local_output
    from src.train import windows
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/data_prep.json")
    assert cfg["chat_examples"] is False and "bert" in cfg["model_id"]
    cfg = {**cfg, "test_size": 10, "model_id": "local", "max_seq_length": 32, "window_stride": 8}
    cfg.pop("image", None)
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    out = prepare_data(fake_source(), tok, cfg, tmp_path / "prepared")
    root = tmp_path / "prepared"
    assert (root / "docs/train.jsonl").exists() and not (root / "sft").exists() and not (root / "tokenized").exists()
    assert verify_local_output(root)["verified"]
    train = [json.loads(l) for l in (root / "docs/train.jsonl").read_text().splitlines()]
    assert out["split_summary"]["train"]["examples"] == sum(1 for _ in windows(train, tok, 32, 8))
    assert "BERT" in out["audit"]["examples_are"]
    assert {p.name for p in (root / "figures").iterdir()} >= {"entity_counts.png", "length_distributions.png"}

"""Data insights: the funnel adds up, the BERT feature counts match the training code, charts render."""
from __future__ import annotations

from synth import make_docs


def fake_audit(docs):
    from src.data_prep import LABELS
    split_info = lambda n, ents: {"documents": n, "groups": n // 2, "negative_documents": 1, "negative_fraction": 1 / n,
                                  "entity_counts": {l: ents for l in LABELS}, "locales": {"us": n // 2, "intl": n - n // 2},
                                  "domains": {"finance": n // 2, "hr": n - n // 2}}
    return {"source_splits": {"train": {"loaded_rows": 120, "rejected_rows": 5, "rejection_reasons": {"bad span": 5}},
                              "test": {"loaded_rows": 60, "rejected_rows": 2, "rejection_reasons": {"bad span": 2}}},
            "selection": {"training_rows_removed_for_test_overlap": 3, "exact_duplicates_removed_by_source": {"train": 4, "test": 1},
                          "duplicate_conflict_rows_removed_by_source": {"train": 1, "test": 0}, "unused_official_test_rows": 24,
                          "requested_test_documents": 33, "cross_split_overlap_counts": {"group_id": {"train:validation": 0}}},
            "prepared_splits": {"train": split_info(80, 9), "validation": split_info(27, 3), "test": split_info(33, 4)}}


def test_funnel_balances_and_charts(tmp_path):
    from src import data_insights as di
    table, rejected, splits, leakage = di.cleanup_funnel(fake_audit(None))
    assert table.rows.iloc[-1] == 140 and "Other adjustments" not in " ".join(table.step)
    assert rejected.rows.sum() == 7 and splits.documents.sum() == 140
    di.funnel_chart(table, tmp_path / "f.png")
    ents = di.entity_table(fake_audit(None))
    di.entity_chart(ents, tmp_path / "e.png")
    di.domain_chart(di.mix_tables(fake_audit(None))[1], tmp_path / "d.png")
    assert all((tmp_path / n).stat().st_size > 1000 for n in ("f.png", "e.png", "d.png"))


def test_bert_features_match_training_windows(tiny_model_dir, tmp_path):
    from transformers import AutoTokenizer
    from src import data_insights as di
    from src.train import windows
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    docs = make_docs(12)
    df, bio = di.bert_features(docs, tok, max_lengths=(32,), stride=8)
    assert df.windows_32.sum() == sum(1 for _ in windows(docs, tok, 32, 8))
    assert bio.words.sum() == sum(len(d["words"]) for d in docs) and "O" in set(bio.tag)
    di.length_chart(df, tmp_path / "l.png", max_lengths=(32,))
    di.bio_chart(bio, tmp_path / "b.png")
    assert not di.feature_summary(df, (32,)).empty

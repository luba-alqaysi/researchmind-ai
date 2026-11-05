"""Benchmark loaders read the saved result files; missing files give empty frames."""

import json

import pandas as pd

from src.benchmarks import generation_results, retrieval_results, rewrite_results


def test_missing_files_give_empty_frames(tmp_path):
    assert retrieval_results(tmp_path).empty and generation_results(tmp_path).empty and rewrite_results(tmp_path).empty


def test_retrieval_results_use_the_deployed_cutoffs(tmp_path):
    pd.DataFrame([
        {"reranker": "none", "MRR": 0.7, "cutoff": 0.41, "short_refused": "15/15", "unanswerable_refused": "3/4", "ms_p50": 5},
        {"reranker": "ms-marco+laya", "MRR": 0.9, "cutoff": 0.5, "short_refused": "5/15", "unanswerable_refused": "4/4", "ms_p50": 780},
    ]).to_csv(tmp_path / "rerank_summary.csv", index=False)
    pd.DataFrame([
        {"reranker": "none", "threshold": 0.3, "answerable_refused": 0, "unanswerable_refused": 2, "short_refused": 12},
        {"reranker": "ms-marco+laya", "threshold": 0.35, "answerable_refused": 0, "unanswerable_refused": 4, "short_refused": 4},
    ]).to_csv(tmp_path / "rerank_thresholds.csv", index=False)
    df = retrieval_results(tmp_path).set_index("config")
    assert df.loc["none", "short_refused"] == 12 and df.loc["none", "cutoff"] == 0.30  # default, not the optimistic 0.41
    assert df.loc["ms-marco+laya", "short_refused"] == 4 and bool(df.loc["ms-marco+laya", "used"])
    assert df.loc["ms-marco+laya", "short_total"] == 15


def test_generation_results_skip_experimental_cutoffs(tmp_path):
    base = {"model": "llama3.1:8b", "citation_precision": 0.6, "min_similarity": 0.3, "reranker": "none"}
    (tmp_path / "generation_ollama_a.json").write_text(json.dumps(base))
    (tmp_path / "generation_ollama_b.json").write_text(json.dumps({**base, "min_similarity": 0.15}))
    (tmp_path / "generation_ollama_c.json").write_text(json.dumps({**base, "reranker": "ms-marco", "rerank_gate": "laya"}))
    df = generation_results(tmp_path)
    assert sorted(df.config) == ["ms-marco+laya", "none"] and df[df.used].iloc[0].config == "ms-marco+laya"


def test_rewrite_findings_are_built_from_the_summary():
    from src.benchmarks import rewrite_findings

    df = pd.DataFrame([
        {"set": "follow_up", "variant": "as written", "answerable": 15, "page_hit@5": 0.4},
        {"set": "follow_up", "variant": "rewritten", "answerable": 15, "page_hit@5": 0.8, "rewrite_s_p50": 2.5,
         "answerable_passed": 12},
        {"set": "arabic", "variant": "English original", "answerable": 12, "page_hit@5": 1.0},
        {"set": "arabic", "variant": "as written", "answerable": 12, "page_hit@5": 0.25},
        {"set": "arabic", "variant": "rewritten", "answerable": 12, "page_hit@5": 0.75},
        {"set": "arabic", "variant": "answers by llama3.1:8b", "answerable": 12, "answered": 11,
         "answered_in_arabic": 10, "answers_with_citation": 9, "citation_precision": 0.7, "unanswerable": 4,
         "unanswerable_refused": 4},
    ])
    text = " ".join(rewrite_findings(df))
    assert "<b>6 of 15</b> when searched as written" in text and "<b>12 of 15</b> after rewriting" in text
    assert "<b>3 of 12</b> when searched in Arabic" in text and "<b>9 of 12</b> after translation" in text
    assert "<b>11 of 12</b> answerable questions (1 wrongly refused), <b>10 of 11</b> in Arabic" in text
    assert "<b>12 of 15</b> rewritten questions then passed" in text and "<b>4 of 4</b> unanswerable" in text
    assert rewrite_findings(pd.DataFrame()) == []

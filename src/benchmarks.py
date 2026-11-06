"""Load the saved evaluation results for the app's Benchmarks tab.

Everything here reads files written by ``evaluation/evaluate_rag.py``; nothing
is computed from scratch or typed in by hand. Missing files give empty frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"

# Configuration names in the result files -> labels a reader understands.
LABELS = {
    "none": "Embedding similarity only",
    "ms-marco": "ms-marco reranker",
    "bge": "bge reranker",
    "laya": "Laya reranker",
    "ms-marco+laya": "ms-marco reranker + Laya check",
}
USED = "ms-marco+laya"  # the configuration in .env.example
# Evidence cutoffs as deployed: cosine default 0.30, Laya check 0.35 (RERANK_MIN_SCORE).
# Other rerankers use the optimistic cutoff saved in rerank_summary.csv.
DEPLOYED_CUTOFF = {"none": 0.30, "laya": 0.35, "ms-marco+laya": 0.35}


def _label(name: str) -> str:
    return LABELS.get(name, name)


def retrieval_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """One row per configuration: MRR, refusals at the deployed cutoff, and latency."""
    summary_path, thresholds_path = results_dir / "rerank_summary.csv", results_dir / "rerank_thresholds.csv"
    if not summary_path.exists() or not thresholds_path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(summary_path)
    thresholds = pd.read_csv(thresholds_path)
    rows = []
    for r in summary.itertuples():
        cutoff = DEPLOYED_CUTOFF.get(r.reranker, float(r.cutoff))
        t = thresholds[(thresholds.reranker == r.reranker) & ((thresholds.threshold - cutoff).abs() < 1e-6)]
        if t.empty:
            continue
        t = t.iloc[0]
        rows.append({
            "config": r.reranker, "label": _label(r.reranker), "used": r.reranker == USED,
            "mrr": float(r.MRR), "cutoff": cutoff,
            "short_refused": int(t.short_refused), "short_total": int(str(r.short_refused).split("/")[1]),
            "unanswerable_refused": int(t.unanswerable_refused),
            "unanswerable_total": int(str(r.unanswerable_refused).split("/")[1]),
            "answerable_refused": int(t.answerable_refused), "ms_p50": float(r.ms_p50),
        })
    return pd.DataFrame(rows)


def generation_results(results_dir: Path = RESULTS_DIR, model: str | None = None) -> pd.DataFrame:
    """End-to-end runs with local Ollama models, one row per (model, retrieval configuration).

    ``used`` marks the configuration the app uses, with ``model`` if it has been evaluated,
    otherwise with the most recently evaluated model. Labels name the model when several are present."""
    rows = []
    for path in sorted(results_dir.glob("generation_ollama_*.json")):
        run = json.loads(path.read_text())
        ranker, gate = run.get("reranker") or "none", run.get("rerank_gate") or "none"
        if ranker == gate == "none" and abs(run.get("min_similarity", 0.3) - 0.3) > 1e-9:
            continue  # experimental cutoffs are reported in RESULTS.md, not here
        config = ranker if gate == "none" else f"{ranker}+{gate}"
        rows.append({
            "config": config, "label": _label(config), "used": config == USED, "model": run["model"],
            "citation_precision": run.get("citation_precision"),
            "answers_citing_evidence": run.get("answers_citing_evidence_page"),
            "unanswerable_refused_rate": run.get("unanswerable_abstention_rate"),
            "false_refusal_rate": run.get("false_abstention_rate"),
            "invalid_citations": run.get("invalid_citation_ids"),
            "questions": run.get("questions"), "date": run.get("date"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if df.model.nunique() > 1:
        df["label"] = df.label + " (" + df.model + ")"
    candidates = df[df.config == USED]
    if not candidates.empty:
        pick = candidates[candidates.model == model] if model in set(candidates.model) else candidates
        latest = pick.sort_values("date").index[-1]
        df["used"] = df.index == latest
    return df


def rewrite_results(results_dir: Path = RESULTS_DIR, model: str | None = None) -> pd.DataFrame:
    """The follow-up and Arabic results for ``model`` if it has been evaluated, else the most recent run.
    Adds a ``model`` column."""
    runs = []
    for path in results_dir.glob("rewrite_summary_*.csv"):
        meta_path = path.with_name(path.name.replace("rewrite_summary_", "rewrite_meta_").replace(".csv", ".json"))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        runs.append((meta.get("llm", "").split("/")[-1], meta.get("date", ""), path))
    if not runs:
        return pd.DataFrame()
    chosen = [r for r in runs if r[0] == model] or runs
    name, _, path = max(chosen, key=lambda r: r[1])
    return pd.read_csv(path).assign(model=name)


def _hits(row) -> str:
    return f"<b>{round(row['page_hit@5'] * row['answerable'])} of {int(row['answerable'])}</b>"


def rewrite_findings(df: pd.DataFrame) -> list[str]:
    """Plain-sentence findings from rewrite_summary.csv (HTML with <b> for the numbers)."""
    if df.empty:
        return []
    rows = {(r["set"], r["variant"]): r for r in df.to_dict("records")}
    out = []
    plain, rewritten = rows.get(("follow_up", "as written")), rows.get(("follow_up", "rewritten"))
    if plain and rewritten:
        out.append(
            f"Follow-up questions such as \"what about the sample size?\", asked without choosing a paper"
            f"{' (rewritten by ' + str(rewritten['model']) + ')' if 'model' in rewritten else ''}: the "
            f"evidence page was found for {_hits(plain)} when searched as written, and for {_hits(rewritten)} after "
            f"rewriting them with the conversation (median {rewritten['rewrite_s_p50']:.1f} s per rewrite); "
            f"<b>{int(rewritten['answerable_passed'])} of {int(rewritten['answerable'])}</b> rewritten questions then "
            f"passed the evidence check.")
    en, ar, ar_rw = (rows.get(("arabic", v)) for v in ("English original", "as written", "rewritten"))
    if en and ar and ar_rw:
        out.append(
            f"Arabic questions: the evidence page was found for {_hits(ar)} when searched in Arabic, and for "
            f"{_hits(ar_rw)} after translation into an English search; the original English questions found "
            f"{_hits(en)}.")
    switch = rows.get(("arabic", "rewritten after a turn about another paper"))
    if switch and "page_hit@5_without_turn" in switch:
        n = int(switch["answerable"])
        before = round(switch["page_hit@5_without_turn"] * n)
        out.append(f"After a turn about a different paper, rewritten Arabic questions about one paper found the "
                   f"evidence page for {_hits(switch)}, against <b>{before} of {n}</b> without that turn.")
    gen = next((r for (s, v), r in rows.items() if s == "arabic" and str(v).startswith("answers by")), None)
    if gen:
        model = str(gen["variant"]).removeprefix("answers by ")
        answerable, answered = int(gen["answerable"]), int(gen["answered"])
        refused = answerable - answered
        wrongly = f" ({refused} wrongly refused)" if refused else ""
        out.append(
            f"With {model} answering the Arabic questions, it answered <b>{answered} of {answerable}</b> answerable "
            f"questions{wrongly}, <b>{int(gen['answered_in_arabic'])} of {answered}</b> in Arabic. "
            f"<b>{int(gen['answers_with_citation'])}</b> of those answers carry citations, and "
            f"<b>{gen['citation_precision']:.0%}</b> of the citations point to an evidence page. "
            f"<b>{int(gen['unanswerable_refused'])} of {int(gen['unanswerable'])}</b> unanswerable questions were refused.")
    return out

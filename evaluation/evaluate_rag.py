"""Reproducible evaluation of ResearchMind AI.

Two modes:

  retrieval   No LLM needed. Grid over chunk size, overlap, reference stripping
              and top-k. Measures whether the retrieved passages contain the
              verified evidence, and calibrates the insufficient-evidence
              similarity threshold.

      python evaluation/evaluate_rag.py retrieval

  generation  Full RAG run with an LLM provider. Measures citation accuracy,
              abstention on unanswerable questions, latency, and (optionally,
              with an LLM judge) answer correctness and faithfulness.

      python evaluation/evaluate_rag.py generation --provider ollama
      python evaluation/evaluate_rag.py generation --provider ollama --judge claude --judge-model claude-haiku-5-5

  rerank      No LLM needed. Compares rerankers (none, ms-marco, bge, laya) on
              retrieval quality, the insufficient-evidence check, and latency,
              including short follow-up style questions ("what about the sample
              size?") asked with a single-paper filter. "RANKER+GATE" (for
              example ms-marco+laya) ranks with one model and lets a second
              model judge the final passages. Rows of earlier runs with other
              names are kept, so configurations can be added one at a time.

      python evaluation/evaluate_rag.py rerank
      python evaluation/evaluate_rag.py rerank --rerankers ms-marco+laya
      python evaluation/evaluate_rag.py generation --provider ollama --reranker laya --rerank-min-score 0.5
      python evaluation/evaluate_rag.py generation --provider ollama --reranker ms-marco --rerank-gate laya --rerank-min-score 0.35

  report      Rebuild evaluation/RESULTS.md from the saved result files.

      python evaluation/evaluate_rag.py report

Requires the evaluation papers: python evaluation/download_papers.py
All numbers in RESULTS.md are produced by this script; nothing is hand-entered.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: E402,F401  (macOS OpenMP workaround before torch/faiss)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Settings  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.llm_client import LLMError, get_llm_client  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.reranker import RERANKERS, get_reranker  # noqa: E402

PAPERS_DIR = ROOT / "sample_data" / "papers"
QUESTIONS = Path(__file__).with_name("evaluation_questions.csv")
RESULTS_DIR = Path(__file__).with_name("results")
RESULTS_MD = Path(__file__).with_name("RESULTS.md")


# ------------------------------------------------------------------ helpers
def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_questions() -> pd.DataFrame:
    df = pd.read_csv(QUESTIONS).fillna("")
    df["evidence"] = df["evidence_pages"].apply(
        lambda s: {(x.split(":")[0], int(x.split(":")[1])) for x in s.split(";")} if s else set()
    )
    df["phrases"] = df["evidence_phrases"].apply(lambda s: [norm(p) for p in s.split("||")] if s else [])
    df["answerable"] = df["type"] != "unanswerable"
    return df


def parse_spec(spec: str) -> tuple[str, str]:
    """"ms-marco+laya" -> ("ms-marco", "laya"); "laya" -> ("laya", "none")."""
    ranker, _, gate = spec.partition("+")
    return ranker, gate or "none"


def build_pipeline(settings: Settings, embedder: Embedder, reranker=None, gate=None) -> RAGPipeline:
    papers = sorted(PAPERS_DIR.glob("*.pdf"))
    if not papers:
        sys.exit("No papers found. Run: python evaluation/download_papers.py")
    pipeline = RAGPipeline(settings, embedder, reranker, gate)
    for path in papers:
        result = pipeline.ingest_pdf(path.read_bytes(), path.name)
        if not result.ok:
            sys.exit(f"Failed to index {path.name}: {result.message}")
    return pipeline


def retrieval_metrics(row, retrieved) -> dict:
    pages = [(r.chunk.doc_name, r.chunk.page_number) for r in retrieved]
    first_hit = next((i + 1 for i, p in enumerate(pages) if p in row.evidence), None)
    texts = [norm(r.chunk.text) for r in retrieved]
    evidence_docs = {d for d, _ in row.evidence}
    retrieved_docs = {d for d, _ in pages}
    return {
        "page_hit": first_hit is not None,
        "phrase_hit": any(ph in t for ph in row.phrases for t in texts),
        "rr": 1.0 / first_hit if first_hit else 0.0,
        "doc_coverage": len(evidence_docs & retrieved_docs) / len(evidence_docs) if evidence_docs else np.nan,
        "top_score": retrieved[0].score if retrieved else 0.0,
    }


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- retrieval
def run_retrieval(args) -> None:
    questions = load_questions()
    answerable = questions[questions.answerable]
    base = Settings.from_env()
    embedder = Embedder(base.embedding_model)
    rows, per_question = [], []
    skip_options = {"on": [True], "off": [False], "both": [True, False]}[args.skip_references]
    balance_options = {"on": [True], "off": [False], "both": [True, False]}[args.balance]

    for chunk_size in args.chunk_sizes:
        for overlap in args.overlaps:
            if overlap >= chunk_size:
                continue
            for skip in skip_options:
                settings = base.with_overrides(chunk_size=chunk_size, chunk_overlap=overlap, skip_references=skip)
                t0 = time.perf_counter()
                pipeline = build_pipeline(settings, embedder)
                index_s = time.perf_counter() - t0
                for balance, k in [(b, k) for b in balance_options for k in args.top_k]:
                    pipeline.settings = settings.with_overrides(balance_documents=balance)
                    metrics, latencies = [], []
                    for row in questions.itertuples():
                        t1 = time.perf_counter()
                        retrieved = pipeline.retrieve(row.question, top_k=k)
                        latencies.append((time.perf_counter() - t1) * 1000)
                        m = retrieval_metrics(row, retrieved) if row.answerable else {"top_score": retrieved[0].score}
                        m.update(id=row.id, type=row.type, answerable=row.answerable, chunk_size=chunk_size,
                                 overlap=overlap, skip_references=skip, balance=balance, top_k=k,
                                 n_retrieved=len(retrieved))
                        metrics.append(m)
                    m = pd.DataFrame(metrics)
                    a = m[m.answerable]
                    rows.append({
                        "chunk_size": chunk_size, "overlap": overlap, "skip_refs": skip, "balance": balance, "top_k": k,
                        "chunks": len(pipeline.store), "page_hit@k": a.page_hit.mean(),
                        "phrase_hit@k": a.phrase_hit.mean(), "MRR": a.rr.mean(),
                        "xdoc_coverage": a[a.type == "cross_document"].doc_coverage.mean(),
                        "avg_passages": a.n_retrieved.mean(),
                        "retrieval_ms_p50": float(np.median(latencies)), "index_s": index_s,
                    })
                    per_question.append(m)
                    r = rows[-1]
                    print(f"size={chunk_size:5d} overlap={overlap:4d} skip_refs={skip!s:5} balance={balance!s:5} k={k:2d} | "
                          f"page_hit={r['page_hit@k']:.3f} phrase_hit={r['phrase_hit@k']:.3f} MRR={r['MRR']:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS_DIR / "retrieval_grid.csv", index=False)
    pq = pd.concat(per_question)
    pq.to_csv(RESULTS_DIR / "retrieval_per_question.csv", index=False)

    # Threshold calibration at the default configuration.
    d = pq[(pq.chunk_size == base.chunk_size) & (pq.overlap == base.chunk_overlap)
           & (pq.skip_references == base.skip_references) & (pq.balance == base.balance_documents)
           & (pq.top_k == base.top_k)]
    if not d.empty:
        calib = []
        for t in np.round(np.arange(0.15, 0.51, 0.05), 2):
            calib.append({
                "threshold": t,
                "answerable_rejected": int((d[d.answerable].top_score < t).sum()),
                "unanswerable_rejected": int((d[~d.answerable].top_score < t).sum()),
            })
        pd.DataFrame(calib).to_csv(RESULTS_DIR / "threshold_calibration.csv", index=False)
        scores = d[["id", "type", "top_score"]].sort_values("top_score")
        scores.to_csv(RESULTS_DIR / "top_scores_default.csv", index=False)
    meta = {"date": dt.date.today().isoformat(), "embedding_model": base.embedding_model,
            "questions": int(len(questions)), "answerable": int(len(answerable))}
    (RESULTS_DIR / "retrieval_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {RESULTS_DIR / 'retrieval_grid.csv'}")
    write_report()


# --------------------------------------------------------------- generation
JUDGE_SYSTEM = """You are a strict evaluator of answers produced by a research assistant.
You receive a question, a verified reference answer, the source passages the assistant was given, and the assistant's answer.
Return ONLY a JSON object with these keys:
  "correctness": 2 if the answer contains the key facts of the reference answer without contradicting it, 1 if it is partially correct or incomplete, 0 if it is wrong, missing, or claims insufficient evidence when the reference contains an answer. If the reference says the question is not answerable, give 2 when the assistant says the evidence is insufficient or not reported, and 0 if it invents an answer.
  "faithfulness": a number from 0 to 1, the fraction of factual claims in the assistant's answer that are supported by the source passages (1 if it makes no factual claims beyond saying evidence is missing).
  "unsupported_claims": a list of short strings quoting claims not supported by the passages (empty list if none).
Do not include any text outside the JSON object."""


def judge_prompt(question: str, reference: str, passages: str, answer: str) -> str:
    return (f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
            f"Source passages given to the assistant:\n{passages}\n\nAssistant's answer:\n{answer}")


def parse_judge(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return {
            "correctness": float(data.get("correctness")),
            "faithfulness": float(data.get("faithfulness")),
            "unsupported_claims": json.dumps(data.get("unsupported_claims", [])),
        }
    except (ValueError, TypeError):
        return {}


def run_generation(args) -> None:
    questions = load_questions()
    if args.limit:
        questions = questions.head(args.limit)
    base = Settings.from_env()
    settings = base.with_overrides(
        chunk_size=args.chunk_size or base.chunk_size,
        chunk_overlap=args.overlap if args.overlap is not None else base.chunk_overlap,
        top_k=args.top_k or base.top_k,
        balance_documents=base.balance_documents and not args.no_balance,
        reranker=args.reranker,
        rerank_gate=args.rerank_gate,
        rerank_min_score=args.rerank_min_score,
        min_similarity=args.min_similarity if args.min_similarity is not None else base.min_similarity,
    )
    llm = get_llm_client(args.provider, settings, model=args.model)
    judge = get_llm_client(args.judge, settings, model=args.judge_model) if args.judge != "none" else None
    if judge is not None and judge.provider == llm.provider and judge.model == llm.model:
        print("WARNING: the judge is the same model as the system under test (self-grading bias).")

    pipeline = build_pipeline(settings, Embedder(settings.embedding_model), get_reranker(settings.reranker),
                              get_reranker(settings.rerank_gate))
    print(f"Indexed {len(pipeline.documents)} papers, {len(pipeline.store)} chunks. "
          f"Running {len(questions)} questions with {llm.provider}/{llm.model}…")

    records = []
    for row in questions.itertuples():
        record = {"id": row.id, "type": row.type, "question": row.question, "answerable": row.answerable}
        try:
            answer = pipeline.ask(row.question, llm)
        except LLMError as exc:
            record.update(error=str(exc))
            records.append(record)
            print(f"  {row.id}: ERROR {exc}")
            continue
        cited = [(c.doc_name, c.page_number) for c in answer.citations]
        record.update(
            answer=answer.answer, insufficient=answer.insufficient_evidence,
            n_citations=len(cited), cited_pages=";".join(f"{d}:{p}" for d, p in cited),
            n_invalid_ids=len(answer.invalid_source_ids),
            skipped_llm="generation_ms" not in answer.timings,
            retrieval_ms=answer.timings.get("retrieval_ms"), generation_ms=answer.timings.get("generation_ms"),
            truncated=any("cut off" in w for w in answer.warnings),
        )
        if row.answerable:
            record.update(retrieval_metrics(row, answer.retrieved))
            correct = [p in row.evidence for p in cited]
            record["citation_precision"] = float(np.mean(correct)) if correct else np.nan
            record["cites_evidence"] = any(correct)
        if judge is not None:
            passages = "\n\n".join(
                f"[{r.chunk.doc_name}, page {r.chunk.page_number}] {r.chunk.text}" for r in answer.retrieved
            )
            try:
                verdict = judge.generate(JUDGE_SYSTEM, judge_prompt(row.question, row.reference_answer, passages, answer.answer))
                record.update(parse_judge(verdict.text))
            except LLMError as exc:
                record["judge_error"] = str(exc)
        records.append(record)
        gen = record.get("generation_ms")
        print(f"  {row.id}: citations={record['n_citations']} insufficient={record['insufficient']} "
              f"correctness={record.get('correctness', '-')} gen={(gen or 0) / 1000:.1f}s")

    df = pd.DataFrame(records)
    suffix = "" if settings.balance_documents else "_nobalance"
    if settings.reranker != "none":
        suffix += f"_rr-{settings.reranker}"
    if settings.rerank_gate != "none":
        suffix += f"_gate-{settings.rerank_gate}"
    if settings.reranker == settings.rerank_gate == "none" and args.min_similarity is not None:
        suffix += f"_cut{settings.min_similarity:g}"
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{llm.provider}_{llm.model}{suffix}")
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"generation_{tag}_raw.csv", index=False)  # gitignored (contains paper text)

    ok = df[df.get("error").isna()] if "error" in df else df
    a, u = ok[ok.answerable], ok[~ok.answerable]
    summary = {
        "date": dt.date.today().isoformat(), "provider": llm.provider, "model": llm.model,
        "judge": f"{judge.provider}/{judge.model}" if judge else "none",
        "chunk_size": settings.chunk_size, "overlap": settings.chunk_overlap, "top_k": settings.top_k,
        "balance_documents": settings.balance_documents, "min_similarity": settings.min_similarity,
        "reranker": settings.reranker,
        "rerank_gate": settings.rerank_gate,
        "rerank_min_score": (settings.rerank_min_score
                             if settings.reranker != "none" or settings.rerank_gate != "none" else None),
        "questions": int(len(df)), "errors": int(len(df) - len(ok)),
        "answerable": int(len(a)), "unanswerable": int(len(u)),
        "page_hit@k": float(a.page_hit.mean()) if len(a) else None,
        "answers_with_citation": float((a.n_citations > 0).mean()) if len(a) else None,
        "citation_precision": float(a.citation_precision.mean()) if len(a) else None,
        "answers_citing_evidence_page": float(a.cites_evidence.mean()) if len(a) else None,
        "invalid_citation_ids": int(ok.n_invalid_ids.sum()) if len(ok) else 0,
        "false_abstention_rate": float(a.insufficient.mean()) if len(a) else None,
        "unanswerable_abstention_rate": float(u.insufficient.mean()) if len(u) else None,
        "retrieval_ms_p50": float(ok.retrieval_ms.median()) if len(ok) else None,
        "generation_s_p50": float(ok.generation_ms.dropna().median() / 1000) if ok.generation_ms.notna().any() else None,
        "generation_s_p95": float(ok.generation_ms.dropna().quantile(0.95) / 1000) if ok.generation_ms.notna().any() else None,
        "truncated_answers": int(ok.truncated.sum()) if len(ok) else 0,
    }
    if judge is not None and "correctness" in ok:
        judged = ok.dropna(subset=["correctness"])
        ja, ju = judged[judged.answerable], judged[~judged.answerable]
        summary.update({
            "judged": int(len(judged)),
            "correctness_mean_0to2": float(ja.correctness.mean()) if len(ja) else None,
            "fully_correct_rate": float((ja.correctness == 2).mean()) if len(ja) else None,
            "unanswerable_correct_rate": float((ju.correctness == 2).mean()) if len(ju) else None,
            "faithfulness_mean": float(judged.faithfulness.mean()) if len(judged) else None,
        })
    (RESULTS_DIR / f"generation_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    write_report()


# ------------------------------------------------------------------ rerank
SHORT_TEMPLATES = {
    "sample_size": "ok what about the sample size?",
    "limitations": "what are the limitations?",
    "methodology": "which methods did they use?",
    "findings": "what did they find?",
}


def short_questions(questions: pd.DataFrame) -> pd.DataFrame:
    """Short, follow-up style questions, each asked with a single-paper filter.

    They reuse the verified evidence of the full questions: for each paper and
    question type, the evidence is the union of that type's evidence pages and
    phrases in that paper. They reproduce a failure seen in real use, where a
    short question scored below the similarity cutoff although the right passage
    had been retrieved."""
    rows = []
    single = questions[questions.answerable & (questions.doc_name != "ALL")]
    for (doc, qtype), group in single.groupby(["doc_name", "type"]):
        if qtype not in SHORT_TEMPLATES:
            continue
        evidence = {e for ev in group.evidence for e in ev if e[0] == doc}
        rows.append({
            "id": f"S-{doc.split('.')[0]}-{qtype}", "type": qtype, "doc_name": doc,
            "question": SHORT_TEMPLATES[qtype], "evidence": evidence,
            "phrases": [p for ps in group.phrases for p in ps], "answerable": True,
        })
    return pd.DataFrame(rows)


def auroc(positive: list[float], negative: list[float]) -> float:
    """Probability that a random answerable question outscores a random unanswerable one."""
    if not positive or not negative:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def run_rerank(args) -> None:
    import gc

    # Cutoff 0 for the per-document guarantee, so every configuration returns the same passage budget
    # whatever RERANK_MIN_SCORE says in .env; the evidence cutoffs are analysed from the saved scores.
    questions, base = load_questions(), Settings.from_env().with_overrides(rerank_min_score=0.0)
    short = short_questions(questions)
    pipeline = build_pipeline(base, Embedder(base.embedding_model))
    name_to_id = {info.doc_name: doc_id for doc_id, info in pipeline.documents.items()}
    print(f"Indexed {len(pipeline.documents)} papers, {len(pipeline.store)} chunks; "
          f"{len(questions)} full questions, {len(short)} short questions.")
    summary, per_question, thresholds = [], [], []

    for name in args.rerankers:
        ranker_name, gate_name = parse_spec(name)
        t0 = time.perf_counter()
        reranker, gate = get_reranker(ranker_name), get_reranker(gate_name)
        load_s = time.perf_counter() - t0
        pipeline.reranker, pipeline.gate = reranker, gate
        judged = pipeline.judge is not None
        pipeline.retrieve("warm-up question about the study")  # first call compiles kernels / allocates
        records = []
        for set_name, df in (("full", questions), ("short", short)):
            for row in df.itertuples():
                scope = {name_to_id[row.doc_name]} if set_name == "short" else None
                t1 = time.perf_counter()
                retrieved = pipeline.retrieve(row.question, doc_ids=scope)
                ms = (time.perf_counter() - t1) * 1000
                m = retrieval_metrics(row, retrieved) if row.answerable else {}
                best = max(r.gate_score for r in retrieved) if judged else retrieved[0].score
                m.update(id=row.id, set=set_name, type=row.type, answerable=row.answerable, reranker=name,
                         gate_score=best, ms=ms)
                records.append(m)
        m = pd.DataFrame(records)
        per_question.append(m)
        full, sh = m[m.set == "full"], m[m.set == "short"]
        fa, fu = full[full.answerable], full[~full.answerable]
        # Cutoff that refuses no answerable full-length question (picked on this same set,
        # so it is optimistic; with 27 questions there is no room for a held-out split).
        grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
        cutoff = float(max(t for t in grid if (fa.gate_score < t).sum() == 0))
        for t in grid:
            thresholds.append({
                "reranker": name, "threshold": t,
                "answerable_refused": int((fa.gate_score < t).sum()),
                "unanswerable_refused": int((fu.gate_score < t).sum()),
                "short_refused": int((sh.gate_score < t).sum()),
            })
        row = {
            "reranker": name,
            "model": " + ".join(RERANKERS[n] for n in (ranker_name, gate_name) if n != "none") or "none",
            "page_hit@5": fa.page_hit.mean(), "phrase_hit@5": fa.phrase_hit.mean(), "MRR": fa.rr.mean(),
            "xdoc_coverage": fa[fa.type == "cross_document"].doc_coverage.mean(),
            "short_page_hit@5": sh.page_hit.mean(), "short_MRR": sh.rr.mean(),
            "gate_auroc": auroc(list(fa.gate_score), list(fu.gate_score)),
            "cutoff": cutoff,
            "unanswerable_refused": f"{int((fu.gate_score < cutoff).sum())}/{len(fu)}",
            "short_refused": f"{int((sh.gate_score < cutoff).sum())}/{len(sh)}",
            "ms_p50": float(np.median(m.ms)), "ms_p95": float(np.quantile(m.ms, 0.95)), "load_s": load_s,
        }
        if not judged:
            row["short_refused_at_default"] = f"{int((sh.gate_score < base.min_similarity).sum())}/{len(sh)}"
        summary.append(row)
        print(f"{name:14s} page_hit={row['page_hit@5']:.3f} MRR={row['MRR']:.3f} short_page_hit={row['short_page_hit@5']:.3f} "
              f"auroc={row['gate_auroc']:.3f} cutoff={cutoff:.2f} unans_refused={row['unanswerable_refused']} "
              f"short_refused={row['short_refused']} ms_p50={row['ms_p50']:.0f}")
        pipeline.reranker = pipeline.gate = None
        del reranker, gate
        gc.collect()

    RESULTS_DIR.mkdir(exist_ok=True)

    def save(df: pd.DataFrame, file_name: str) -> None:
        """Replace this run's configurations in the saved file and keep the others."""
        path = RESULTS_DIR / file_name
        if path.exists():
            old = pd.read_csv(path)
            df = pd.concat([old[~old.reranker.isin(args.rerankers)], df], ignore_index=True)
        df.to_csv(path, index=False)

    save(pd.DataFrame(summary), "rerank_summary.csv")
    save(pd.DataFrame(thresholds), "rerank_thresholds.csv")
    save(pd.concat(per_question).drop(columns=["top_score"], errors="ignore"), "rerank_per_question.csv")
    meta = {"date": dt.date.today().isoformat(), "embedding_model": base.embedding_model,
            "top_k": base.top_k, "candidates": base.rerank_candidates,
            "full_questions": int(len(questions)), "short_questions": int(len(short))}
    (RESULTS_DIR / "rerank_meta.json").write_text(json.dumps(meta, indent=2))
    write_report()


# ------------------------------------------------------------------- report
def write_report() -> None:
    parts = [
        "# Evaluation results",
        "",
        "Generated by `python evaluation/evaluate_rag.py report` from the files in `evaluation/results/`. "
        "Do not edit by hand. Methodology: see the Evaluation section of the README.",
        "",
    ]
    meta_path = RESULTS_DIR / "retrieval_meta.json"
    grid_path = RESULTS_DIR / "retrieval_grid.csv"
    if grid_path.exists():
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        grid = pd.read_csv(grid_path)
        parts += [
            "## Retrieval experiments",
            "",
            f"Run {meta.get('date', '?')} with `{meta.get('embedding_model', '?')}` on "
            f"{meta.get('answerable', '?')} answerable questions over all four papers (no document filter).",
            "",
            "* **page_hit@k**: share of questions where at least one retrieved chunk comes from a verified evidence page.",
            "* **phrase_hit@k**: share where a retrieved chunk contains a verified evidence phrase (stricter).",
            "* **MRR**: mean reciprocal rank of the first evidence-page chunk.",
            "* **xdoc_coverage**: for cross-document questions, share of the relevant papers represented in the results.",
            "* **avg_passages**: passages returned (top-k plus the per-document guarantee for multi-paper scope).",
            "",
        ]
        shown = grid.drop(columns=["index_s"]).sort_values(["top_k", "chunk_size", "overlap", "skip_refs", "balance"])
        parts += [md_table(shown.rename(columns={"retrieval_ms_p50": "ms_p50"})), ""]
        best = grid.sort_values(["phrase_hit@k", "MRR"], ascending=False).iloc[0]
        parts += [
            f"Best configuration by phrase_hit@k then MRR: chunk size {int(best.chunk_size)}, overlap "
            f"{int(best.overlap)}, skip_refs={bool(best.skip_refs)}, balance={bool(best.balance)}, top-k {int(best.top_k)} "
            f"(phrase_hit@k {best['phrase_hit@k']:.3f}, MRR {best.MRR:.3f}).",
            "",
        ]
    calib_path = RESULTS_DIR / "threshold_calibration.csv"
    if calib_path.exists():
        parts += [
            "## Insufficient-evidence threshold",
            "",
            "Number of questions whose best passage falls below each similarity threshold (default configuration). "
            "Ideally all unanswerable and no answerable questions are rejected. Questions above the threshold are "
            "still checked by the LLM, which can report insufficient evidence itself.",
            "",
            md_table(pd.read_csv(calib_path), ".2f"),
            "",
        ]
    rr_path = RESULTS_DIR / "rerank_summary.csv"
    if rr_path.exists():
        meta_path = RESULTS_DIR / "rerank_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        rr = pd.read_csv(rr_path)
        parts += [
            "## Reranker comparison",
            "",
            f"Last run {meta.get('date', '?')}. Default chunking, top-k {meta.get('top_k', '?')}; rerankers rescore the "
            f"{meta.get('candidates', '?')} best embedding-search candidates. {meta.get('full_questions', '?')} full "
            f"questions plus {meta.get('short_questions', '?')} short follow-up style questions (\"what about the sample "
            "size?\") asked with a single-paper filter. No LLM is involved.",
            "",
            "* **gate_auroc**: how well the evidence score separates answerable from unanswerable full questions "
            "(1.0 = perfectly, 0.5 = chance). Based on only 4 unanswerable questions.",
            "* **cutoff**: the highest evidence cutoff that refuses no answerable full question, chosen on this same "
            "set (optimistic). **unanswerable_refused** and **short_refused** are counted at that cutoff.",
            "* **short_refused_at_default**: short questions refused by the current default (cosine 0.30).",
            "* **ms_p50 / ms_p95**: retrieval plus reranking time per question on this machine.",
            "* **RANKER+GATE** rows (for example ms-marco+laya): the first model orders the candidates, the second "
            "judges only the final passages, and its score is the evidence score.",
            "",
            md_table(rr.drop(columns=["load_s"]).fillna("")),
            "",
        ]
    gens = sorted(RESULTS_DIR.glob("generation_*.json"))
    if gens:
        parts += ["## End-to-end generation", ""]
        table = pd.DataFrame([json.loads(p.read_text()) for p in gens])
        keep = [c for c in [
            "provider", "model", "judge", "date", "questions", "errors", "page_hit@k", "answers_with_citation",
            "citation_precision", "answers_citing_evidence_page", "invalid_citation_ids", "false_abstention_rate",
            "unanswerable_abstention_rate", "correctness_mean_0to2", "fully_correct_rate",
            "unanswerable_correct_rate", "faithfulness_mean", "retrieval_ms_p50", "generation_s_p50",
            "generation_s_p95", "truncated_answers",
        ] if c in table]
        rr = table["reranker"].fillna("none") if "reranker" in table else pd.Series(["none"] * len(table))
        gate = table["rerank_gate"].fillna("none") if "rerank_gate" in table else pd.Series(["none"] * len(table))
        table["run"] = (table.provider + " / " + table.model + np.where(rr != "none", " + " + rr + " reranker", "")
                        + np.where(gate != "none", " + " + gate + " gate", ""))
        if "min_similarity" in table:
            plain = (rr == "none") & (gate == "none")
            table.loc[plain, "run"] += " (cutoff " + table.loc[plain, "min_similarity"].map("{:g}".format) + ")"
        keep = ["run"] + [c for c in keep if c not in ("provider", "model")]
        transposed = table[keep].set_index("run").T.reset_index().rename(columns={"index": "metric"})
        parts += [
            "* **citation_precision**: share of rendered citations whose document and page are a verified evidence page.",
            "* **answers_citing_evidence_page**: share of answerable questions with at least one citation to an evidence page.",
            "* **false_abstention_rate**: answerable questions answered with 'insufficient evidence'.",
            "* **unanswerable_abstention_rate**: unanswerable questions correctly answered with 'insufficient evidence'.",
            "* **correctness / faithfulness**: LLM-judge scores (see README for caveats); absent when no judge was used.",
            "",
            md_table(transposed),
            "",
        ]
    RESULTS_MD.write_text("\n".join(parts))
    print(f"Wrote {RESULTS_MD}")


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("retrieval", help="retrieval grid (no LLM)")
    r.add_argument("--chunk-sizes", type=int, nargs="+", default=[400, 800, 1200])
    r.add_argument("--overlaps", type=int, nargs="+", default=[0, 150])
    r.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 8])
    r.add_argument("--skip-references", choices=["on", "off", "both"], default="both")
    r.add_argument("--balance", choices=["on", "off", "both"], default="both",
                   help="per-document passage guarantee for multi-document scope")

    g = sub.add_parser("generation", help="end-to-end RAG with an LLM")
    g.add_argument("--provider", default="ollama", choices=["ollama", "gemini", "claude", "extractive"])
    g.add_argument("--model", default=None)
    g.add_argument("--judge", default="none", choices=["none", "ollama", "gemini", "claude"])
    g.add_argument("--judge-model", default=None)
    g.add_argument("--chunk-size", type=int, default=None)
    g.add_argument("--overlap", type=int, default=None)
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--no-balance", action="store_true", help="disable the per-document passage guarantee")
    g.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    g.add_argument("--reranker", default="none", choices=["none", *RERANKERS])
    g.add_argument("--rerank-gate", default="none", choices=["none", *RERANKERS],
                   help="separate model that judges the final passages for the evidence check")
    g.add_argument("--rerank-min-score", type=float, default=0.0, help="evidence cutoff on the gate or reranker score")
    g.add_argument("--min-similarity", type=float, default=None, help="cosine evidence cutoff (no reranker)")

    k = sub.add_parser("rerank", help="compare rerankers (no LLM)")
    k.add_argument("--rerankers", nargs="+", default=["none", *RERANKERS, "ms-marco+laya"],
                   help="names from: none, " + ", ".join(RERANKERS) + "; or RANKER+GATE, e.g. ms-marco+laya")

    sub.add_parser("report", help="rebuild RESULTS.md")
    args = parser.parse_args()
    {"retrieval": run_retrieval, "generation": run_generation, "rerank": run_rerank,
     "report": lambda _: write_report()}[args.mode](args)


if __name__ == "__main__":
    main()

"""The Benchmarks tab: saved evaluation results as plain findings and three bar charts.

Chart colours (validated with the dataviz palette script on the #121A2B surface):
the configuration the app uses is blue #4F7BF2, the others neutral #868A90. The
grey is deliberately low-chroma (de-emphasis, not a categorical hue); every bar
also carries a direct value label, so colour is never the only cue.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.benchmarks import generation_results, retrieval_results, rewrite_findings, rewrite_results

USED_COLOR, OTHER_COLOR = "#4F7BF2", "#868A90"
TEXT, MUTED, GRID = "#E6EAF2", "#93A0B8", "#2C3850"


def _bar_chart(df: pd.DataFrame, value: str, value_title: str, fmt: str, domain: tuple[float, float],
               tooltip: list[alt.Tooltip]) -> alt.Chart:
    order = list(df.sort_values(value, ascending=False).label) if value != "short_refused" else \
        list(df.sort_values(value).label)
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=260, labelColor=TEXT, labelFontSize=13, ticks=False, domain=False,
                              labelPadding=10)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X(f"{value}:Q", title=value_title, scale=alt.Scale(domain=domain),
                axis=alt.Axis(grid=True, gridColor=GRID, domain=False, ticks=False, labelColor=MUTED,
                              titleColor=MUTED, titleFontWeight="normal", format=fmt, tickCount=5)),
        color=alt.condition(alt.datum.used, alt.value(USED_COLOR), alt.value(OTHER_COLOR)),
        tooltip=tooltip,
    )
    labels = base.mark_text(align="left", dx=6, color=TEXT, fontSize=12).encode(
        x=alt.X(f"{value}:Q"), text=alt.Text(f"{value}:Q", format=fmt),
    )
    return (bars + labels).properties(height=46 * len(df)).configure_view(stroke=None).configure(
        background="transparent", font="IBM Plex Sans")


def _title(title: str, note: str) -> None:
    st.markdown(f'<div class="rm-chart-title">{title}</div><div class="rm-chart-note">{note}</div>',
                unsafe_allow_html=True)


def render(model: str | None = None) -> None:
    """``model``: the app's current local model; its results are shown when it has been evaluated."""
    retrieval, generation = retrieval_results(), generation_results(model=model)
    rewrite = rewrite_results(model=model)
    if retrieval.empty and generation.empty:
        st.info("No saved results yet. Run `python evaluation/evaluate_rag.py rerank` and "
                "`python evaluation/evaluate_rag.py generation --provider ollama` to create them.")
        return

    st.markdown('<p class="rm-bench-lede">Measured on 31 questions about 4 open-access papers, plus 15 short '
                'follow-up questions. Every number below is read from the saved evaluation results.</p>', unsafe_allow_html=True)

    findings = []
    if not retrieval.empty and retrieval.used.any():
        used = retrieval[retrieval.used].iloc[0]
        base = retrieval[retrieval.config == "none"]
        findings.append(
            f"<b>{used.unanswerable_refused} of {used.unanswerable_total}</b> questions the papers can't answer were "
            "refused before the language model was called, and no answerable question was refused.")
        if not base.empty:
            b = base.iloc[0]
            findings.append(
                f"Short follow-up questions wrongly refused: <b>{b.short_refused} of {b.short_total}</b> with "
                f"embedding similarity only, <b>{used.short_refused} of {used.short_total}</b> with the reranker "
                "and evidence check.")
            findings.append(
                f"The passage with the answer ranks first more often: MRR <b>{b.mrr:.2f}</b> before, "
                f"<b>{used.mrr:.2f}</b> after reranking.")
    if not generation.empty and generation.used.any():
        g = generation[generation.used].iloc[0]
        findings.append(
            f"With {g.model} writing the answers, <b>{g.citation_precision:.0%}</b> of citations point to a page "
            f"that holds the evidence, and <b>{int(g.invalid_citations)}</b> cite a passage that was never retrieved.")
    findings += rewrite_findings(rewrite)
    st.markdown('<ul class="rm-findings">' + "".join(f"<li>{f}</li>" for f in findings) + "</ul>",
                unsafe_allow_html=True)

    if not retrieval.empty:
        df = retrieval.assign(short_refused_share=retrieval.short_refused / retrieval.short_total)
        _title("Short questions wrongly refused",
               "Out of 15 questions like \"what about the sample size?\" asked about one paper. Lower is better.")
        st.altair_chart(_bar_chart(
            df, "short_refused", "questions refused (of 15)", "d", (0, 15),
            [alt.Tooltip("label:N", title="Setup"), alt.Tooltip("short_refused:Q", title="Refused"),
             alt.Tooltip("cutoff:Q", title="Evidence cutoff", format=".2f"),
             alt.Tooltip("ms_p50:Q", title="Median time (ms)", format=",.0f")]), use_container_width=True)

        _title("How often the answer passage ranks first",
               "Mean reciprocal rank over 27 answerable questions; 1.0 means the evidence page is always first.")
        st.altair_chart(_bar_chart(
            df, "mrr", "MRR", ".2f", (0, 1),
            [alt.Tooltip("label:N", title="Setup"), alt.Tooltip("mrr:Q", title="MRR", format=".3f")]),
            use_container_width=True)

    if not generation.empty:
        _title("Citations that point to the evidence",
               "Share of citations whose page holds the reference evidence, with Llama 3.1 8B writing the answers.")
        st.altair_chart(_bar_chart(
            generation, "citation_precision", "citation precision", ".0%", (0, 1),
            [alt.Tooltip("label:N", title="Setup"),
             alt.Tooltip("citation_precision:Q", title="Citation precision", format=".1%"),
             alt.Tooltip("answers_citing_evidence:Q", title="Answers citing an evidence page", format=".1%"),
             alt.Tooltip("date:N", title="Run date")]), use_container_width=True)

    with st.expander("Caveats and the full numbers"):
        st.markdown(
            "- The question set is small (31 questions, 4 unanswerable) and was written for this project.\n"
            "- Evidence cutoffs were chosen on the same questions, so refusal numbers are optimistic. For the Laya "
            "check, any cutoff from 0.28 to 0.50 refuses all unanswerable and no answerable question; the app "
            "uses 0.35.\n"
            "- Citation precision counts a citation as correct only on the pages listed as evidence, so correct "
            "citations to other pages count as misses.\n"
            "- The full tables are in `evaluation/RESULTS.md`.")
        if not retrieval.empty:
            st.dataframe(retrieval.drop(columns=["used", "config"]), hide_index=True)
        if not generation.empty:
            st.dataframe(generation.drop(columns=["used", "config"]), hide_index=True)

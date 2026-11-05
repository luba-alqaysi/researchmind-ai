"""The app's stylesheet.

Design brief: a dark "lab bench" is the instrument panel (interface text in IBM
Plex Sans); text quoted from papers sits on lit page slips in IBM Plex Serif, so
the typeface shows who is speaking. Highlighter yellow is the only saturated
warm colour and always marks evidence. Status green/red appear only in the
evidence check. One motion moment: the highlight sweep (off with reduced motion).

Custom classes are prefixed ``rm-``; Streamlit containers are targeted only by
their ``key`` (``.st-key-…``), never by internal class names.
"""

CSS = """
<style>
:root {
  --rm-bench: #121A2B;   /* app background */
  --rm-panel: #1B2538;   /* raised instrument surfaces */
  --rm-line: #2C3850;    /* borders and rules */
  --rm-text: #E6EAF2;
  --rm-muted: #93A0B8;
  --rm-signal: #8FB0FF;  /* links, citation tags, focus */
  --rm-page: #F7F8FA;    /* lit page slip */
  --rm-ink: #1A2233;     /* text on the page */
  --rm-ink-muted: #5A6578;
  --rm-hl: #FFE15A;      /* highlighter: evidence only */
  --rm-pass: #4CC38A;
  --rm-refuse: #F2706A;
  --rm-serif: "IBM Plex Serif", Georgia, serif;
}

.block-container { max-width: 880px; padding-top: 4.5rem; padding-bottom: 6rem; }
.rm-num { font-variant-numeric: tabular-nums; }
:focus-visible { outline: 2px solid var(--rm-signal) !important; outline-offset: 2px; }

/* ---------------------------------------------------------------- hero */
.rm-hero-title {
  font-size: clamp(1.7rem, 3.3vw, 2.55rem); line-height: 1.12; font-weight: 600;
  letter-spacing: -0.02em; margin: 0.5rem 0 1.75rem; color: var(--rm-text); padding: 0;
}
.rm-lede { color: var(--rm-muted); font-size: 1.02rem; line-height: 1.55; max-width: 62ch; margin: 1.25rem 0 0; }

/* ------------------------------------------------------------ page slip */
.rm-page {
  display: grid; grid-template-columns: 7.5rem 1fr; gap: 0 1.25rem;
  background: var(--rm-page); color: var(--rm-ink); border-radius: 2px;
  padding: 1.1rem 1.4rem 1.1rem 1rem; margin: 0 0 0.75rem;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.04), 0 12px 32px -18px rgba(0,0,0,0.65);
}
.rm-folio { font-size: 0.78rem; line-height: 1.35; color: var(--rm-ink-muted); border-right: 1px solid #DDE2EA;
            padding-right: 0.9rem; overflow-wrap: anywhere; }
.rm-folio b { display: block; font-size: 1.35rem; font-weight: 500; color: var(--rm-ink); margin-bottom: 0.2rem; }
figure.rm-page blockquote.rm-quote {
  font-family: var(--rm-serif); font-size: 0.97rem; line-height: 1.68; margin: 0; max-width: 68ch;
  white-space: pre-wrap; border: none !important; padding: 0 !important; color: var(--rm-ink);
}
.rm-slip-meta { grid-column: 2; margin-top: 0.55rem; font-size: 0.76rem; color: var(--rm-ink-muted); }

/* The one bold element: a highlighter stroke over the supporting sentence. */
mark.rm-hl {
  color: inherit; padding: 0.05em 0.15em; margin: 0 -0.15em; border-radius: 0.2em 0.35em 0.25em 0.3em;
  background-image: linear-gradient(100deg, rgba(255,225,90,0.75), rgba(255,225,90,0.95) 6%, rgba(255,214,60,0.9) 94%, rgba(255,225,90,0.6));
  background-repeat: no-repeat; background-size: 100% 88%; background-position: 0 60%;
  -webkit-box-decoration-break: clone; box-decoration-break: clone;
}
.rm-sweep mark.rm-hl { animation: rm-sweep 900ms cubic-bezier(.2,.7,.2,1) 250ms both; }
@keyframes rm-sweep { from { background-size: 0% 88%; } to { background-size: 100% 88%; } }
@media (prefers-reduced-motion: reduce) { .rm-sweep mark.rm-hl { animation: none; } }

/* --------------------------------------------------------- question/answer */
.rm-q { font-size: 1.22rem; line-height: 1.35; font-weight: 500; color: var(--rm-text); margin: 2.25rem 0 0.6rem; max-width: 60ch; }
.rm-q-first { margin-top: 0.5rem; }
.rm-answer { font-size: 1rem; line-height: 1.62; max-width: 68ch; }
.rm-answer p { margin-bottom: 0.7rem; }
.rm-answer table { font-size: 0.9rem; }
.rm-rtl, .rm-rtl p, .rm-rtl li { direction: rtl; text-align: right; font-family: "IBM Plex Sans Arabic", "IBM Plex Sans", sans-serif; line-height: 1.8; }
.rm-cite {
  display: inline-block; font-size: 0.76rem; line-height: 1.3; font-weight: 500; white-space: nowrap;
  color: var(--rm-signal); border: 1px solid rgba(143,176,255,0.45); border-radius: 3px;
  padding: 0.02rem 0.38rem; margin: 0 0 0 0.15rem; vertical-align: 0.08em; direction: ltr; unicode-bidi: isolate;
}
.rm-warn { color: #F5C26B; font-size: 0.86rem; margin: 0.25rem 0; }

/* Answer the app declined to write */
.rm-refusal { border-left: 3px solid var(--rm-refuse); padding: 0.2rem 0 0.2rem 1rem; max-width: 66ch; }
.rm-refusal h4 { margin: 0 0 0.35rem; font-size: 1.05rem; font-weight: 600; color: var(--rm-text); }
.rm-refusal p { margin: 0; color: var(--rm-muted); line-height: 1.55; }

/* ------------------------------------------------- "How this answer was made" */
details.rm-how { margin: 1.1rem 0 1rem; }
details.rm-how > summary { cursor: pointer; color: var(--rm-muted); font-size: 0.88rem; list-style: none; width: fit-content; }
details.rm-how > summary::-webkit-details-marker { display: none; }
details.rm-how > summary::before { content: "+"; display: inline-block; width: 1.1em; color: var(--rm-signal); }
details.rm-how[open] > summary::before { content: "\\2212"; }
ol.rm-trace {
  list-style: none; margin: 0.7rem 0 0; padding: 0; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
  border-top: 1px solid var(--rm-line); counter-reset: step;
}
ol.rm-trace > li { position: relative; padding: 0.85rem 0.9rem 0.2rem 0; counter-increment: step; min-width: 0; }
ol.rm-trace > li::before {
  content: counter(step); position: absolute; top: -0.62rem; left: 0; width: 1.2rem; height: 1.2rem; border-radius: 50%;
  background: var(--rm-bench); border: 1px solid var(--rm-line); color: var(--rm-muted);
  font-size: 0.68rem; line-height: 1.12rem; text-align: center; font-variant-numeric: tabular-nums;
}
ol.rm-trace > li.ok::before { border-color: var(--rm-signal); color: var(--rm-signal); }
ol.rm-trace > li.pass::before { border-color: var(--rm-pass); color: var(--rm-pass); }
ol.rm-trace > li.refuse::before { border-color: var(--rm-refuse); color: var(--rm-refuse); }
ol.rm-trace > li.skip { opacity: 0.55; }
.rm-step-name { font-weight: 600; font-size: 0.86rem; color: var(--rm-text); margin-bottom: 0.2rem; }
.rm-step-detail { font-size: 0.8rem; line-height: 1.45; color: var(--rm-muted); overflow-wrap: anywhere; }
.rm-step-detail q { color: var(--rm-text); }
li.refuse .rm-step-name { color: var(--rm-refuse); }
.rm-meter { position: relative; height: 6px; border-radius: 3px; background: var(--rm-line); margin: 0.45rem 0 0.3rem; max-width: 9rem; }
.rm-meter > .fill { position: absolute; inset: 0 auto 0 0; border-radius: 3px; background: var(--rm-pass); }
li.refuse .rm-meter > .fill { background: var(--rm-refuse); }
.rm-meter > .mark { position: absolute; top: -4px; width: 2px; height: 14px; background: var(--rm-text); border-radius: 1px; }

/* ---------------------------------------------------------- sources */
.rm-sources-title { font-size: 0.88rem; color: var(--rm-muted); margin: 1.1rem 0 0.55rem; font-weight: 500; }
details.rm-more > summary { cursor: pointer; color: var(--rm-muted); font-size: 0.85rem; margin: 0.4rem 0 0.6rem; width: fit-content; }

/* ---------------------------------------------------------- sidebar */
.rm-wordmark { font-weight: 600; font-size: 1.2rem; letter-spacing: -0.01em; margin: 0 0 0.15rem; }
.rm-wordmark-sub { color: var(--rm-muted); font-size: 0.84rem; line-height: 1.45; margin: 0 0 1.1rem; }
.rm-side-h { font-weight: 600; font-size: 0.92rem; margin: 1.2rem 0 0.45rem; }
.rm-paper { border-left: 2px solid var(--rm-hl); padding: 0.1rem 0 0.1rem 0.7rem; margin: 0.6rem 0 0.15rem; }
.rm-paper-name { font-size: 0.88rem; line-height: 1.35; overflow-wrap: anywhere; }
.rm-paper-meta { font-size: 0.76rem; color: var(--rm-muted); }
.st-key-rm-papers [data-testid="stButton"] { margin: -0.35rem 0 0.35rem 0.75rem; }
.st-key-rm-papers [data-testid="stButton"] button { min-height: 0; padding: 0; color: var(--rm-muted); }
.st-key-rm-papers [data-testid="stButton"] button p { font-size: 0.76rem; }
.st-key-rm-papers [data-testid="stButton"] button:hover p { color: var(--rm-refuse); }

/* ---------------------------------------------------------- benchmarks */
.rm-bench-lede { font-size: 1.12rem; line-height: 1.5; max-width: 60ch; margin: 0.2rem 0 1.4rem; }
ul.rm-findings { margin: 0 0 1.6rem !important; padding: 0 !important; list-style: none; max-width: 66ch; }
ul.rm-findings li { margin: 0 !important; padding: 0.55rem 0; border-bottom: 1px solid var(--rm-line); line-height: 1.5; }
.rm-findings b { font-variant-numeric: tabular-nums; }
.rm-chart-title { font-weight: 600; font-size: 1rem; margin: 1.6rem 0 0.1rem; }
.rm-chart-note { color: var(--rm-muted); font-size: 0.84rem; margin: 0 0 0.4rem; max-width: 70ch; }

@media (max-width: 720px) {
  ol.rm-trace { grid-template-columns: 1fr; border-top: none; border-left: 1px solid var(--rm-line); margin-left: 0.6rem; }
  ol.rm-trace > li { padding: 0 0 0.9rem 1.2rem; }
  ol.rm-trace > li::before { top: 0; left: -0.62rem; }
  .rm-page { grid-template-columns: 1fr; }
  .rm-folio { border-right: none; border-bottom: 1px solid #DDE2EA; padding: 0 0 0.5rem; margin-bottom: 0.6rem; }
  .rm-slip-meta { grid-column: 1; }
}
</style>
"""

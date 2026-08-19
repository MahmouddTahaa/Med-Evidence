"""Clinician / student pharmacology chat — grounded generation only.

Operator ingest/eval stays in streamlit_app.py (port 8501). This app is port 8502.
Does not import eval.grid or retune winning.yaml.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import streamlit as st

from clinical_rag.adapters.llms import CascadeClient, build_cascade
from clinical_rag.config import get_settings
from clinical_rag.errors import IngestError, QueryError
from clinical_rag.generate.engine import GroundedEngine, StreamEventKind
from clinical_rag.generate.models import (
    DISCLAIMER,
    ChatMessage,
    Confidence,
    Outcome,
    TurnRequest,
    TurnResult,
)
from clinical_rag.query import RetrievalSession

st.set_page_config(
    page_title="Med-Evidence",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="expanded",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METRICS_PATH = _REPO_ROOT / "data" / "eval" / "product_lock_metrics.json"
_TEMPLATES_PATH = _REPO_ROOT / "data" / "eval" / "statpearls_pharmacology_templates.jsonl"

# Demo pools — one random question per button press.
DEMO_REFUSAL_POOL = [
    "What's the weather in Boston today?",
    "Who won the NBA finals last year?",
    "Write me a Python sorting algorithm.",
    "Should I buy Bitcoin this week?",
    "I'm a patient — should I take ampicillin for my sore throat?",
    "My doctor prescribed metoprolol; can I stop it if I feel better?",
    "I have chest pain and fever — what medication should I take?",
]

DEMO_OUT_OF_CORPUS_POOL = [
    "What is the USPSTF breast cancer screening interval for average-risk adults?",
    "What is first-line pharmacologic treatment for essential hypertension per ACC/AHA?",
    "When should adults start colorectal cancer screening according to USPSTF?",
    "What is the recommended A1C target for most non-pregnant adults with type 2 diabetes?",
    "How often should adults get influenza vaccination per CDC?",
    "What is the CHADS2 score threshold for anticoagulation in atrial fibrillation?",
]

_CONFIDENCE_LANGUAGE: dict[Confidence, str] = {
    Confidence.high: "The source supports this educational point.",
    Confidence.medium: (
        "The source suggests this, though it may not fully address the question."
    ),
    Confidence.low: "Limited evidence found in the retrieved passages.",
    Confidence.insufficient: (
        "Evidence is below the retrieval gate — no grounded recommendation."
    ),
}

_SOURCE_NOTE = (
    "Corpus: StatPearls pharmacology (NCBI Bookshelf / open educational reuse). "
    "Answers are for learning only — attribute the indexed article title and section "
    "when citing. Not a substitute for the full primary reference."
)


@st.cache_resource(show_spinner="Loading frozen retrieval stack…")
def _load_session() -> RetrievalSession:
    return RetrievalSession.open()


@st.cache_resource(show_spinner="Resolving LLM providers…")
def _load_cascade() -> CascadeClient:
    return build_cascade(get_settings().generation)


@st.cache_data(show_spinner=False)
def _load_product_metrics() -> dict:
    if _METRICS_PATH.is_file():
        return json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
    return {"run_id": "unknown", "aggregates": {}}


@st.cache_data(show_spinner=False)
def _load_in_scope_pool() -> list[str]:
    if not _TEMPLATES_PATH.is_file():
        return ["What are the contraindications for Ampicillin?"]
    queries: list[str] = []
    for line in _TEMPLATES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        q = str(row.get("query") or "").strip()
        if q:
            queries.append(q)
    return queries or ["What are the contraindications for Ampicillin?"]


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[{role, content, result?, metrics?}]
    if "last_provider" not in st.session_state:
        st.session_state.last_provider = ""
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def _history_for_engine() -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in st.session_state.messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role in ("user", "assistant") and content:
            out.append(ChatMessage(role=role, content=content))
    return out


def _page_label(page_number: int | None) -> str:
    if page_number is not None:
        return f"p. {page_number}"
    return "p. n/a (section-indexed source)"


def _confidence_phrase(conf: Confidence | str) -> str:
    if isinstance(conf, Confidence):
        return _CONFIDENCE_LANGUAGE.get(conf, conf.value)
    try:
        return _CONFIDENCE_LANGUAGE[Confidence(str(conf))]
    except ValueError:
        return str(conf)


def _queue_demo(text: str) -> None:
    st.session_state.pending_prompt = text


@st.cache_resource(show_spinner="Loading claim support scorer…")
def _load_entailment_scorer():
    from clinical_rag.generate.claim_metrics import build_entailment_scorer

    # MiniLM CE (same family as rerank) — MNLI under-scores paraphrased answers.
    return build_entailment_scorer(prefer_nli=False, device="cpu")


def _citation_accuracy(result: TurnResult) -> float | None:
    from clinical_rag.generate.turn_metrics import citation_accuracy

    return citation_accuracy(result)


def _turn_generation_metrics(result: TurnResult, *, latency_ms: float | None) -> dict:
    """Claim-support faithfulness + citation bind rate."""
    from clinical_rag.generate.claim_metrics import score_claims

    if result.outcome is Outcome.refusal:
        return {
            "outcome": "refusal",
            "insufficient_evidence": False,
            "exclude_from_session": True,
            "faithfulness": None,
            "citation_accuracy": None,
            "entailment_rate": None,
            "n_claims": 0,
            "top_hit_score": None,
            "n_hits": 0,
            "n_citations": 0,
            "latency_ms": latency_ms,
            "note": "Refusal — generation metrics N/A (correct behavior).",
        }

    cit_acc = _citation_accuracy(result)
    premises = [h.text or "" for h in result.hits]
    faithfulness: float | None = None
    entailment_rate: float | None = None
    n_claims = 0
    scorer_name = ""
    note = ""
    auto_filled = bool(getattr(result, "citations_auto_filled", False))
    proposed = int(getattr(result, "citations_proposed", 0) or 0)
    bound = int(getattr(result, "citations_bound", len(result.citations)) or 0)
    insufficient = bool(result.insufficient_evidence)

    if insufficient:
        note = (
            "Out of corpus — Case C; no claim scoring."
            if bool(getattr(result, "out_of_corpus", False))
            else (
                "Insufficient / extractive — claim entailment skipped; "
                "citation accuracy N/A (auto-filled)."
                if auto_filled
                else "Insufficient / extractive — claim entailment skipped."
            )
        )
    elif result.recommendation:
        scorer = _load_entailment_scorer()
        report = score_claims(result.recommendation, premises, scorer)
        scorer_name = report.scorer_name
        n_claims = len(report.claims)
        faithfulness = report.faithfulness
        entailment_rate = report.entailment_rate
        note = (
            f"Claim support ({scorer_name}): {report.note or f'{n_claims} claims'}."
        )
    else:
        note = "No recommendation text to score."

    top = result.hits[0].score if result.hits else None
    return {
        "outcome": result.outcome.value,
        "insufficient_evidence": insufficient,
        "exclude_from_session": insufficient or result.outcome is Outcome.refusal,
        "faithfulness": faithfulness,
        "citation_accuracy": cit_acc,
        "entailment_rate": entailment_rate,
        "n_claims": n_claims,
        "scorer": scorer_name,
        "citations_proposed": proposed,
        "citations_bound": bound,
        "citations_auto_filled": auto_filled,
        "top_hit_score": top,
        "n_hits": len(result.hits),
        "n_citations": len(result.citations),
        "latency_ms": latency_ms,
        "note": note,
    }


def _fmt_metric(value: float | None, *, pct: bool = False) -> str:
    if value is None:
        return "N/A"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _render_turn_metrics(metrics: dict) -> None:
    with st.expander("This turn — generation metrics", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Faithfulness", _fmt_metric(metrics.get("faithfulness")))
        c2.metric("Citation acc.", _fmt_metric(metrics.get("citation_accuracy")))
        lat = metrics.get("latency_ms")
        c3.metric("Latency", f"{lat:.0f} ms" if isinstance(lat, (int, float)) else "N/A")
        st.caption(
            f"Claims {metrics.get('n_claims', 0)} · "
            f"Hard support rate {_fmt_metric(metrics.get('entailment_rate'))} · "
            f"Cites proposed/bound "
            f"{metrics.get('citations_proposed', 0)}/"
            f"{metrics.get('citations_bound', 0)}"
            f"{' (auto-filled)' if metrics.get('citations_auto_filled') else ''} · "
            f"Hits {metrics.get('n_hits', 0)} · "
            f"Top score {_fmt_metric(metrics.get('top_hit_score'))}"
        )
        if metrics.get("scorer"):
            st.caption(f"Claim support scorer: `{metrics['scorer']}`")
        if metrics.get("note"):
            st.caption(str(metrics["note"]))


def _render_session_generation_metrics() -> None:
    from clinical_rag.generate.turn_metrics import include_in_session_average

    gen_rows = [
        m.get("metrics")
        for m in st.session_state.messages
        if m.get("role") == "assistant" and isinstance(m.get("metrics"), dict)
    ]
    gen_rows = [r for r in gen_rows if include_in_session_average(r)]
    if not gen_rows:
        return

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in gen_rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    st.markdown(
        f"**This session — generation** "
        f"(mean over {len(gen_rows)} grounded turn"
        f"{'s' if len(gen_rows) != 1 else ''}; excludes refusal / Case C)"
    )
    g1, g2, g3 = st.columns(3)
    g1.metric("Faithfulness", _fmt_metric(_mean("faithfulness")))
    g2.metric("Citation acc.", _fmt_metric(_mean("citation_accuracy")))
    avg_lat = _mean("latency_ms")
    g3.metric("Avg latency", f"{avg_lat:.0f} ms" if avg_lat is not None else "N/A")


def _render_retrieval_metrics_panel() -> None:
    payload = _load_product_metrics()
    agg = payload.get("aggregates") or {}
    run_id = payload.get("run_id", "—")
    n_q = payload.get("n_questions", "—")

    with st.expander("Locked retrieval scorecard (product eval)", expanded=False):
        st.caption(
            f"Product lock `{run_id}` · {n_q} labeled queries · StatPearls pharmacology. "
            "These are retrieval bakeoff numbers, not medical correctness."
        )
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("MRR", _fmt_metric(agg.get("mrr")))
        r2.metric("P@5", _fmt_metric(agg.get("precision@5")))
        r3.metric("R@5", _fmt_metric(agg.get("recall@5")))
        r4.metric("nDCG@5", _fmt_metric(agg.get("ndcg@5")))
        st.caption(
            "P@5 ceiling on this gold set is 0.40 (many singleton labels). "
            "See docs/evaluation.md."
        )
        st.markdown(
            f"| Metric | Value |\n| --- | --- |\n"
            f"| Precision@1 / @3 / @5 / @10 | "
            f"{_fmt_metric(agg.get('precision@1'))} / "
            f"{_fmt_metric(agg.get('precision@3'))} / "
            f"{_fmt_metric(agg.get('precision@5'))} / "
            f"{_fmt_metric(agg.get('precision@10'))} |\n"
            f"| Recall@1 / @3 / @5 / @10 | "
            f"{_fmt_metric(agg.get('recall@1'))} / "
            f"{_fmt_metric(agg.get('recall@3'))} / "
            f"{_fmt_metric(agg.get('recall@5'))} / "
            f"{_fmt_metric(agg.get('recall@10'))} |\n"
            f"| Hit@1 / @3 / @5 / @10 | "
            f"{_fmt_metric(agg.get('hit@1'))} / "
            f"{_fmt_metric(agg.get('hit@3'))} / "
            f"{_fmt_metric(agg.get('hit@5'))} / "
            f"{_fmt_metric(agg.get('hit@10'))} |\n"
            f"| nDCG@1 / @3 / @5 / @10 | "
            f"{_fmt_metric(agg.get('ndcg@1'))} / "
            f"{_fmt_metric(agg.get('ndcg@3'))} / "
            f"{_fmt_metric(agg.get('ndcg@5'))} / "
            f"{_fmt_metric(agg.get('ndcg@10'))} |\n"
            f"| Latency p50 / p95 | "
            f"{agg.get('latency_ms_p50', '—')} / {agg.get('latency_ms_p95', '—')} ms |\n"
        )

    _render_session_generation_metrics()


def _render_result_blocks(result: TurnResult, metrics: dict | None = None) -> None:
    if result.outcome is Outcome.refusal:
        st.warning("Refused — out of scope or unsafe for this educational guide.")
    elif bool(getattr(result, "out_of_corpus", False)):
        st.warning(
            "Out of corpus — guideline / screening / risk-score topic; "
            "no grounded recommendation from this pharmacology index."
        )
    elif result.insufficient_evidence:
        st.info("Insufficient / partial evidence — no unbound recommendation.")
    elif result.outcome is Outcome.complex and not result.recommendation:
        st.info("Extractive fallback — citations only (generation unavailable).")

    if result.recommendation:
        st.markdown("**Recommendation**")
        st.write(result.recommendation)

    if result.retrieval_query and (
        result.outcome is Outcome.refusal
        or result.insufficient_evidence
        or result.outcome is Outcome.complex
    ):
        st.caption(f"Searched: {result.retrieval_query}")
    elif result.retrieval_query and result.outcome is not Outcome.refusal:
        st.caption(f"Retrieved with: {result.retrieval_query}")

    if result.evidence:
        st.markdown("**Evidence**")
        for q in result.evidence:
            st.markdown(f"- {q.text} (`{q.chunk_id}`)")

    if result.citations:
        st.markdown("**Citations**")
        for c in result.citations:
            st.markdown(
                f"- `{c.chunk_id}` — {c.document_name} · {c.section_title} · "
                f"{_page_label(c.page_number)}"
            )
    elif result.recommendation and result.outcome is Outcome.success:
        st.warning(
            "Recommendation had no bound citations — treat as unsupported and verify sources."
        )

    conf = result.confidence
    label = conf.value if isinstance(conf, Confidence) else str(conf)
    st.markdown(f"**Confidence:** {label}")
    st.caption(_confidence_phrase(conf if isinstance(conf, Confidence) else label))
    st.info(result.disclaimer or DISCLAIMER)

    if metrics:
        _render_turn_metrics(metrics)

    if result.hits:
        with st.expander("Sources / evidence", expanded=False):
            for i, hit in enumerate(result.hits, start=1):
                st.markdown(
                    f"**[{i}]** `{hit.chunk_id}` · score {hit.score:.3f} · "
                    f"{hit.document_name} · {hit.section_title} · "
                    f"{_page_label(hit.page_number)}"
                )
                st.write(hit.text)
                st.divider()


def _render_sidebar(session: RetrievalSession, cascade: CascadeClient) -> None:
    gen = get_settings().generation
    with st.sidebar:
        st.header("Med-Evidence")
        st.caption("Educational pharmacology guide")
        st.markdown(f"**Job:** `{session.job_id or '—'}`")
        st.markdown(f"**Collection:** `{session.collection}`")
        providers = cascade.providers
        st.markdown(
            "**Providers:** "
            + (", ".join(providers) if providers else "none (extractive fallback)")
        )
        if st.session_state.last_provider:
            st.markdown(f"**Last provider:** `{st.session_state.last_provider}`")

        st.divider()
        st.subheader("Disclaimer")
        st.info(DISCLAIMER)

        st.subheader("Responsible AI")
        st.markdown(
            "- No answer replaces clinical judgment.\n"
            "- Uncertainty language matches evidence strength.\n"
            "- Refusals are never softened for a demo.\n"
            "- Disclaimer stays visible (see above)."
        )

        st.subheader("Retrieval gate")
        st.markdown(
            f"Weak-score threshold: **`{gen.weak_score}`** "
            "(cross-encoder score in 0–1). Below this → insufficient evidence / "
            "no recommendation."
        )

        st.subheader("Source & legal")
        st.caption(_SOURCE_NOTE)

        st.divider()
        st.caption("Demo buttons: In-scope · Refusal · Out-of-corpus (random each press).")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_provider = ""
            st.session_state.pending_prompt = None
            st.rerun()


def _run_turn(prompt: str, session: RetrievalSession, cascade: CascadeClient) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    engine = GroundedEngine(session, cascade, settings=get_settings().generation)
    request = TurnRequest(message=prompt, history=_history_for_engine()[:-1])

    with st.chat_message("assistant"):
        status = st.empty()
        rec_box = st.empty()
        final: TurnResult | None = None
        streamed_rec = ""
        t0 = time.perf_counter()
        try:
            for event in engine.stream_turn(request):
                if event.kind is StreamEventKind.status:
                    status.caption(event.text.replace("_", " ").title() + "…")
                elif event.kind is StreamEventKind.recommendation_delta:
                    streamed_rec += event.text
                    rec_box.markdown("**Recommendation**\n\n" + streamed_rec)
                elif event.kind is StreamEventKind.result and event.result is not None:
                    final = event.result
        except QueryError as exc:
            status.empty()
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            status.empty()
            st.error(f"Turn failed: {exc}")
            return
        latency_ms = (time.perf_counter() - t0) * 1000.0

        status.empty()
        if final is None:
            st.error("No answer produced.")
            return

        metrics = _turn_generation_metrics(final, latency_ms=latency_ms)
        st.session_state.last_provider = final.provider_used or st.session_state.last_provider
        rec_box.empty()
        _render_result_blocks(final, metrics)

        if final.outcome is Outcome.refusal:
            display = final.recommendation or "Refused."
        elif final.insufficient_evidence:
            display = final.recommendation or (
                "Out of corpus."
                if bool(getattr(final, "out_of_corpus", False))
                else "Insufficient evidence."
            )
        else:
            display = final.recommendation or ""
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": display or "",
                "result": final,
                "metrics": metrics,
            }
        )
        # Rerun so session averages / sidebar last-provider refresh without a duplicate turn.
        st.rerun()


def main() -> None:
    _init_state()

    try:
        session = _load_session()
        cascade = _load_cascade()
    except IngestError as exc:
        st.error(f"Cannot load retrieval session: {exc}")
        st.info(
            "Copy `configs/serving.example.yaml` → `configs/serving.yaml` and set "
            "`job_id` to a frozen ingest under `artifacts/jobs/`."
        )
        return
    except QueryError as exc:
        st.error(f"Cannot initialize LLM cascade: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Startup failed: {exc}")
        return

    _render_sidebar(session, cascade)

    st.title("Med-Evidence")
    st.caption(
        "Ask an educational pharmacology question. Answers are grounded in retrieved sources."
    )

    st.markdown("**Demo samples** (random question from each pool)")
    in_scope_pool = _load_in_scope_pool()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("In-scope", use_container_width=True):
            _queue_demo(random.choice(in_scope_pool))
            st.rerun()
    with c2:
        if st.button("Refusal", use_container_width=True):
            _queue_demo(random.choice(DEMO_REFUSAL_POOL))
            st.rerun()
    with c3:
        if st.button("Out-of-corpus", use_container_width=True):
            _queue_demo(random.choice(DEMO_OUT_OF_CORPUS_POOL))
            st.rerun()

    _render_retrieval_metrics_panel()
    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                result = msg.get("result")
                metrics = msg.get("metrics")
                if isinstance(result, TurnResult):
                    _render_result_blocks(
                        result, metrics if isinstance(metrics, dict) else None
                    )
                else:
                    st.write(msg.get("content") or "")

    # Always mount the chat input so the dialog box never disappears.
    typed = st.chat_input("Ask about a drug, contraindication, or interaction…")
    prompt = st.session_state.pending_prompt or typed
    if st.session_state.pending_prompt:
        st.session_state.pending_prompt = None

    if not prompt:
        return

    _run_turn(prompt, session, cascade)


if __name__ == "__main__":
    main()

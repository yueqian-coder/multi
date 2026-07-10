from __future__ import annotations

import os
from html import escape

import streamlit as st

from claimscope.llm import OpenAICompatibleClient
from claimscope.service import ClaimScopeService
from claimscope.ui_components import (
    confidence_label,
    feedback_jsonl,
    provider_host,
    render_activity,
    render_candidate,
    render_critique_summary,
    render_discovery_evidence,
    report_markdown,
)


DEFAULT_DIRECTION = "RAG can reliably reduce hallucination in LLM-generated answers"


def inject_style() -> None:
    st.markdown("""
    <style>
    :root { --ink:#202638; --muted:#667085; --line:#d7dce5; --surface:#fff; --cool:#f5f7fa; --teal:#087f86; --indigo:#3f5bc8; --amber:#c98500; --red:#c03636; }
    .stApp { background:var(--cool); color:var(--ink); }
    .block-container { max-width:1500px; padding:1rem 1.1rem 2.5rem; }
    [data-testid="stHeader"] { background:var(--surface); }
    h1,h2,h3,h4,p { letter-spacing:0; }
    .topbar { background:var(--surface); border:1px solid var(--line); padding:12px 16px; display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:14px; }
    .brand { font-size:25px; font-weight:800; white-space:nowrap; }
    .provider { color:var(--muted); font-size:13px; }
    .provider-dot { color:#07845e; font-size:18px; vertical-align:-1px; }
    .panel, .artifact { background:var(--surface); border:1px solid var(--line); padding:14px; margin-bottom:12px; }
    .panel-title { font-size:18px; font-weight:800; margin-bottom:4px; }
    .muted { color:var(--muted); font-size:12px; }
    .selected-claim { font-size:20px; line-height:1.3; font-weight:750; margin:8px 0 12px; }
    .claim-grid { display:grid; grid-template-columns:1.1fr 1.9fr; gap:12px; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; border-top:1px solid var(--line); padding-top:10px; }
    .metric { border-right:1px solid var(--line); padding-right:8px; }.metric:last-child { border:0; }
    .metric-label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; font-weight:700; }.metric strong { font-size:19px; }
    .candidate { border:1px solid var(--line); padding:10px 12px; margin:7px 0; }.candidate-selected { border-left:4px solid var(--teal); background:#f7fcfc; }
    .candidate-top { display:flex; align-items:center; gap:9px; font-size:13px; }.radio { color:var(--teal); font-size:18px; }.candidate-score { margin-left:auto; color:var(--teal); }.candidate-claim { font-weight:700; margin:6px 0; line-height:1.35; }.candidate-meta { color:#3f4758; font-size:12px; line-height:1.5; }
    .activity-row { display:grid; grid-template-columns:30px 1fr auto; gap:10px; align-items:start; border-top:1px solid var(--line); padding:10px 0; }.agent-number { background:var(--indigo); color:#fff; width:24px; height:24px; text-align:center; padding-top:3px; font-weight:800; }.activity-row p { margin:3px 0 0; color:var(--muted); font-size:12px; }.status { margin-left:10px; color:var(--indigo); font-size:12px; font-weight:700; }.status-failed { color:var(--red); }.status-pending { color:var(--muted); }
    .critique-row { display:grid; grid-template-columns:160px 1fr 45px; gap:10px; border-top:1px solid var(--line); padding:10px 0; font-size:13px; }.critique-row b { color:var(--amber); }
    .evidence-item { border-top:1px solid var(--line); padding:11px 0; }.evidence-item p { font-size:13px; line-height:1.45; margin:5px 0; }.warning { color:var(--amber); }
    .section-label { font-size:16px; font-weight:800; margin:12px 0 7px; }.warning-box { border-left:3px solid var(--amber); padding:8px 10px; background:#fffaf0; color:#815600; font-size:13px; margin:8px 0; }
    @media (max-width: 760px) { .block-container { padding:0.65rem 0.65rem 2rem; }.topbar { padding:10px; }.brand { font-size:22px; }.provider { display:none; }.claim-grid { display:block; }.metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }.metric:nth-child(2) { border:0; }.activity-row { grid-template-columns:28px 1fr; }.activity-row > .muted { grid-column:2; }.critique-row { grid-template-columns:1fr auto; }.critique-row span { grid-column:1 / -1; order:3; }.selected-claim { font-size:18px; } }
    </style>
    """, unsafe_allow_html=True)


def session_defaults() -> None:
    defaults = {"core_claim_result": None, "discovery_result": None, "direction": DEFAULT_DIRECTION, "provider_key": "", "provider_base": os.getenv("OPENAI_BASE_URL", ""), "provider_model": os.getenv("MODEL_NAME", "gpt-4o-mini")}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_service() -> ClaimScopeService:
    key = st.session_state.get("provider_key", "").strip()
    base = st.session_state.get("provider_base", "").strip()
    model = st.session_state.get("provider_model", "").strip() or "gpt-4o-mini"
    client = OpenAICompatibleClient(api_key=st.session_state["provider_key"], base_url=base, model=model) if key and base else OpenAICompatibleClient.from_env()
    return ClaimScopeService(llm_client=client)


def render_topbar(mode: str) -> None:
    with st.container():
        left, right = st.columns([1.6, 1.4])
        with left:
            st.markdown('<div class="topbar"><span class="brand">ClaimScope</span></div>', unsafe_allow_html=True)
        with right:
            with st.popover("Provider"):
                st.text_input("API key", type="password", key="provider_key", help="Session-only; never exported or logged.")
                st.text_input("Base URL", key="provider_base")
                st.text_input("Model", key="provider_model")
                st.caption(f"Host: {provider_host(st.session_state.get('provider_base', ''))}")
            health = "configured" if st.session_state.get("provider_key") and st.session_state.get("provider_base") else "demo / heuristic"
            st.markdown(f'<div class="topbar provider"><span class="provider-dot">●</span> Provider: {escape(health)}</div>', unsafe_allow_html=True)


def render_core_claim(result: dict | None) -> None:
    if not result:
        st.info("Run the arena to compare public candidate claims and critiques.")
        return
    selected = result.get("selected_candidate") or {}
    confidence = float(selected.get("confidence", 0))
    st.markdown(f"""<section class="panel"><div class="panel-title">Selected Core Claim</div><div class="selected-claim">{escape(str(result.get('selected_claim', '')))}</div><div class="metrics">{''.join([f'<div class="metric"><span class="metric-label">Confidence</span><strong>{confidence:.2f}</strong><small>{confidence_label(confidence)}</small></div>', f'<div class="metric"><span class="metric-label">Mode</span><strong>{escape(str(result.get("mode", "heuristic")))}</strong></div>', f'<div class="metric"><span class="metric-label">Falsification test</span><strong>{escape(str(selected.get("falsification_test", "Not specified")))}</strong></div>', f'<div class="metric"><span class="metric-label">Missing information</span><strong>{len(result.get("unresolved_ambiguities", []))}</strong></div>'])}</div></section>""", unsafe_allow_html=True)
    if result.get("degraded"):
        st.markdown('<div class="warning-box">This run used a deterministic fallback for one or more agent stages.</div>', unsafe_allow_html=True)
    st.markdown('<section class="panel"><div class="panel-title">Agent activity</div>' + render_activity(result) + '</section>', unsafe_allow_html=True)
    candidates = result.get("candidates", [])
    st.markdown('<section class="panel"><div class="panel-title">Candidate comparison</div>' + "".join(render_candidate(candidate, candidate is selected or candidate.get("claim") == selected.get("claim"), index) for index, candidate in enumerate(candidates, 1)) + '</section>', unsafe_allow_html=True)
    st.markdown('<section class="panel"><div class="panel-title">Critique summary</div>' + render_critique_summary(result) + '</section>', unsafe_allow_html=True)


def render_discovery(report: dict | None) -> None:
    if not report:
        st.info("Run Full Discovery to map assumptions, evidence, and opportunities.")
        return
    st.markdown(f'<section class="panel"><div class="panel-title">Selected Core Claim</div><div class="selected-claim">{escape(str(report.get("claim", "")))}</div></section>', unsafe_allow_html=True)
    overview, ledger, evidence, opportunities, trace, export = st.tabs(["Overview", "Assumption Ledger", "Evidence", "Opportunities", "Trace", "Export"])
    with overview:
        st.markdown("### Workflow overview")
        st.write({"assumptions": len(report.get("assumptions", [])), "papers": len(report.get("papers", [])), "opportunities": len(report.get("idea_opportunities", []))})
        for warning in report.get("warnings", []): st.markdown(f'<div class="warning-box">{escape(str(warning))}</div>', unsafe_allow_html=True)
    with ledger:
        for assumption in report.get("assumptions", []):
            st.markdown(f'<article class="artifact"><strong>{escape(str(assumption.get("text", "")))}</strong><span class="status">{escape(str(assumption.get("status", "unknown")))}</span><p class="muted">Risk: {escape(str(assumption.get("risk", "medium")))}</p></article>', unsafe_allow_html=True)
    with evidence:
        st.markdown(render_discovery_evidence(report), unsafe_allow_html=True)
    with opportunities:
        for opportunity in report.get("idea_opportunities", []): st.markdown(f'<article class="artifact"><strong>{escape(str(opportunity.get("title", "")))}</strong><p>{escape(str(opportunity.get("rationale", "")))}</p><p class="muted">Next: {escape(str(opportunity.get("next_step", "")))}</p></article>', unsafe_allow_html=True)
    with trace:
        st.markdown('<section class="panel"><div class="panel-title">Agent activity</div>' + render_activity({"events": report.get("events", [])}) + '</section>', unsafe_allow_html=True)
    with export:
        markdown = report_markdown(report)
        st.download_button("Download Markdown report", markdown, "claimscope_report.md", "text/markdown")
        st.code(markdown, language="markdown")


def render_feedback(direction: str, observed: str) -> None:
    with st.expander("Evaluate observed claim"):
        expected = st.text_area("Expected claim", key="feedback_expected")
        rating = st.selectbox("Rating", ["Pass", "Partial", "Fail"], key="feedback_rating")
        notes = st.text_area("Notes", key="feedback_notes")
        st.download_button("Download feedback JSONL", feedback_jsonl(direction, observed, expected, rating, notes), "claimscope_feedback.jsonl", "application/jsonl")


def main() -> None:
    st.set_page_config(page_title="ClaimScope", page_icon="CS", layout="wide")
    session_defaults()
    inject_style()
    if hasattr(st, "segmented_control"):
        mode = st.segmented_control("Mode", ["Core Claim Arena", "Full Discovery"], default="Core Claim Arena", label_visibility="collapsed")
    else:
        mode = st.radio("Mode", ["Core Claim Arena", "Full Discovery"], horizontal=True, label_visibility="collapsed")
    render_topbar(mode)
    left, right = st.columns([1, 1.35])
    with left:
        st.markdown('<section class="panel"><div class="panel-title">Research direction</div><p class="muted">Describe the fuzzy direction you want to stress-test.</p>', unsafe_allow_html=True)
        direction = st.text_area("Research direction", key="direction", height=130, label_visibility="collapsed")
        st.caption("Examples: retrieval quality and hallucination; causal mechanism and measurable outcome.")
        if mode == "Core Claim Arena":
            st.caption("Retrieval controls are disabled in Core Claim mode.")
        else:
            online = st.toggle("Use online academic retrieval", value=False)
            limit = st.slider("Maximum papers", 5, 30, 12)
        run = st.button("Run Arena" if mode == "Core Claim Arena" else "Run Discovery", type="primary", use_container_width=True)
        st.markdown('</section>', unsafe_allow_html=True)
    with right:
        if mode == "Core Claim Arena": render_core_claim(st.session_state.get("core_claim_result"))
        else: render_discovery(st.session_state.get("discovery_result"))
    if run:
        if not direction.strip(): st.warning("Enter a research direction first.")
        else:
            service = build_service()
            with st.spinner("Running public research analysis..."):
                if mode == "Core Claim Arena": st.session_state["core_claim_result"] = service.extract_core_claim(direction)
                else: st.session_state["discovery_result"] = service.analyze_research_direction(direction, online=online, limit=limit)
            st.rerun()
    if mode == "Core Claim Arena":
        result = st.session_state.get("core_claim_result") or {}
        render_feedback(direction, str(result.get("selected_claim", "")))


if __name__ == "__main__":
    main()

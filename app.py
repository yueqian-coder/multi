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
WORKFLOW_LABELS = [
    "Direction",
    "Core claim",
    "Variants",
    "Assumptions",
    "Queries",
    "Evidence",
    "Opportunities",
]


def inject_style() -> None:
    st.markdown("""
    <style>
    :root { --ink:#17212b; --muted:#5c6a73; --line:#d6dfe2; --surface:#ffffff; --canvas:#f3f6f7; --teal:#0b6f70; --navy:#244b5a; --blue:#3d5a80; --amber:#996300; --green:#277a57; --red:#b42318; --soft-teal:#edf7f6; --soft-blue:#eef3f8; --soft-amber:#fff8e8; }
    .stApp { background:var(--canvas); color:var(--ink); font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    .block-container { max-width:1500px; padding:4.5rem 1.1rem 2.5rem; }
    [data-testid="stHeader"] { background:var(--surface); }
    h1,h2,h3,h4,p { letter-spacing:0; }
    .brand-lockup { min-width:190px; }
    .brand { color:var(--navy); font-size:25px; line-height:1.05; font-weight:800; white-space:nowrap; }
    .brand-kicker { color:var(--muted); font-size:10px; font-weight:700; letter-spacing:0; margin-top:5px; text-transform:uppercase; }
    .provider-status { color:var(--muted); font-size:12px; display:flex; align-items:center; justify-content:flex-end; gap:7px; white-space:nowrap; }
    .provider-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green); flex:0 0 auto; }
    .panel, .artifact { background:var(--surface); border:1px solid var(--line); border-radius:6px; box-shadow:0 1px 2px rgba(23,33,43,.04); padding:16px; margin-bottom:12px; }
    .panel-title { color:var(--navy); font-size:17px; font-weight:800; margin-bottom:4px; }
    .eyebrow { color:var(--teal); font-size:10px; font-weight:800; letter-spacing:0; text-transform:uppercase; }
    .muted { color:var(--muted); font-size:12px; }
    .selected-claim { color:var(--ink); font-size:20px; line-height:1.35; font-weight:750; margin:8px 0 14px; max-width:70ch; overflow-wrap:anywhere; }
    .claim-grid { display:grid; grid-template-columns:1.1fr 1.9fr; gap:12px; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:0; border-top:1px solid var(--line); padding-top:12px; }
    .metric { min-height:72px; border-right:1px solid var(--line); padding:0 12px; }.metric:first-child { padding-left:0; }.metric:last-child { border:0; }
    .metric-label { display:block; color:var(--muted); font-size:10px; letter-spacing:0; text-transform:uppercase; font-weight:750; }.metric strong { display:block; font-size:18px; line-height:1.25; margin-top:4px; overflow-wrap:anywhere; }.metric:nth-child(3) strong { font-size:13px; line-height:1.35; font-weight:650; }.metric small { display:block; color:var(--teal); margin-top:3px; }
    .overview-metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border:1px solid var(--line); border-radius:6px; background:var(--surface); margin:10px 0 14px; }
    .overview-metric { min-height:86px; border-right:1px solid var(--line); padding:14px; }.overview-metric:last-child { border:0; }.overview-metric b { color:var(--navy); display:block; font-size:24px; line-height:1; margin-bottom:6px; }.overview-metric span { color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; }
    .candidate { border:1px solid var(--line); border-radius:5px; padding:11px 12px; margin:8px 0; transition:border-color .16s ease,background .16s ease; }.candidate:hover { border-color:#9eb4bb; }.candidate-selected { border-left:4px solid var(--teal); background:var(--soft-teal); }
    .candidate-top { display:flex; align-items:center; gap:9px; font-size:13px; }.radio { display:inline-block; width:14px; height:14px; border:1px solid var(--teal); border-radius:50%; flex:0 0 auto; }.radio-selected { background:var(--teal); box-shadow:inset 0 0 0 3px #fff; }.candidate-score { margin-left:auto; color:var(--teal); }.candidate-claim { font-weight:700; margin:6px 0; line-height:1.35; }.candidate-meta { color:#3f4758; font-size:12px; line-height:1.5; }
    .activity-row { display:grid; grid-template-columns:30px 1fr auto; gap:10px; align-items:start; border-top:1px solid var(--line); padding:10px 0; }.agent-number { background:var(--blue); color:#fff; border-radius:4px; width:24px; height:24px; text-align:center; padding-top:3px; font-weight:800; }.activity-row p { margin:3px 0 0; color:var(--muted); font-size:12px; }.status { margin-left:10px; color:var(--blue); font-size:11px; font-weight:750; text-transform:uppercase; }.status-failed { color:var(--red); }.status-pending { color:var(--muted); }.status-degraded { color:var(--amber); }
    .critique-row { display:grid; grid-template-columns:160px 1fr 45px; gap:10px; border-top:1px solid var(--line); padding:10px 0; font-size:13px; }.critique-row b { color:var(--amber); }
    .evidence-item { border-top:1px solid var(--line); padding:12px 0; }.evidence-item p { font-size:13px; line-height:1.5; margin:5px 0; }.warning { color:var(--amber); }
    .section-label { color:var(--navy); font-size:15px; font-weight:800; margin:12px 0 7px; }.warning-box { border:1px solid #ead8a9; border-left:3px solid var(--amber); border-radius:4px; padding:9px 11px; background:var(--soft-amber); color:#704b00; font-size:13px; margin:8px 0; }
    .workflow-wrap { overflow-x:auto; margin:0 0 12px; padding-bottom:2px; }.workflow-rail { min-width:770px; display:grid; grid-template-columns:repeat(7,minmax(100px,1fr)); border:1px solid var(--line); border-radius:6px; background:var(--surface); }.workflow-step { position:relative; min-height:68px; border-right:1px solid var(--line); padding:11px 9px 9px 38px; }.workflow-step:last-child { border:0; }.workflow-step-num { position:absolute; left:10px; top:12px; width:20px; height:20px; border-radius:4px; background:var(--soft-blue); color:var(--blue); text-align:center; font-size:11px; font-weight:800; padding-top:2px; }.workflow-step strong { display:block; color:var(--ink); font-size:11px; line-height:1.25; }.workflow-step span { color:var(--green); font-size:10px; font-weight:700; text-transform:uppercase; }
    .artifact-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }.artifact-title { font-size:13px; line-height:1.45; }.badge-row { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:5px; }.badge { display:inline-block; border:1px solid var(--line); border-radius:4px; background:#f7f9fa; color:var(--muted); font-size:10px; font-weight:750; padding:3px 6px; text-transform:uppercase; white-space:nowrap; }.badge-mixed { background:var(--soft-amber); border-color:#ead8a9; color:#704b00; }.badge-high { background:#fff0ee; border-color:#f0c1bc; color:var(--red); }.badge-opportunity { background:var(--soft-teal); border-color:#b9dcda; color:var(--teal); }.artifact p { font-size:13px; line-height:1.5; margin:8px 0 0; }.artifact-next { border-top:1px solid var(--line); color:var(--muted); margin-top:10px !important; padding-top:8px; }
    .empty-state { min-height:230px; display:grid; align-content:center; border:1px dashed #b7c5ca; border-radius:6px; background:rgba(255,255,255,.55); padding:32px; text-align:center; }.empty-state strong { color:var(--navy); font-size:16px; }.empty-state p { color:var(--muted); font-size:13px; margin:6px auto 0; max-width:46ch; }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { min-height:44px; border-radius:4px; font-weight:700; transition:background .16s ease,border-color .16s ease; }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { border-radius:4px; }
    [data-testid="stPopover"] button { white-space:nowrap; }
    button:focus-visible, input:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible { outline:3px solid rgba(11,111,112,.28) !important; outline-offset:2px; }
    [role="tab"] { min-height:42px; }
    @media (max-width: 1050px) { .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }.metric { min-height:76px; border-bottom:1px solid var(--line); padding:8px 10px; }.metric:first-child { padding-left:10px; }.metric:nth-child(2) { border-right:0; }.metric:nth-child(n+3) { border-bottom:0; } }
    @media (max-width: 760px) { .block-container { padding:4rem 0.65rem 2rem; }.brand { font-size:22px; }.provider-status { justify-content:flex-start; white-space:normal; }.claim-grid { display:block; }.metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }.metric { min-height:78px; border-bottom:1px solid var(--line); padding:8px 10px; }.metric:nth-child(2) { border-right:0; }.metric:nth-child(n+3) { border-bottom:0; }.overview-metrics { grid-template-columns:1fr; }.overview-metric { border-right:0; border-bottom:1px solid var(--line); }.activity-row { grid-template-columns:28px 1fr; }.activity-row > .muted { grid-column:2; }.critique-row { grid-template-columns:1fr auto; }.critique-row span { grid-column:1 / -1; order:3; }.selected-claim { font-size:18px; }.artifact-head { display:block; }.badge-row { justify-content:flex-start; margin-top:8px; } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior:auto !important; transition-duration:.01ms !important; animation-duration:.01ms !important; animation-iteration-count:1 !important; } }
    </style>
    """, unsafe_allow_html=True)


def session_defaults() -> None:
    defaults = {"core_claim_result": None, "discovery_result": None, "direction": DEFAULT_DIRECTION, "provider_key": "", "provider_base": os.getenv("OPENAI_BASE_URL", ""), "provider_model": os.getenv("MODEL_NAME", "gpt-4o-mini"), "provider_allow_remote": False}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_service() -> ClaimScopeService:
    key = st.session_state.get("provider_key", "").strip()
    base = st.session_state.get("provider_base", "").strip()
    model = st.session_state.get("provider_model", "").strip() or "gpt-4o-mini"
    allow_remote = bool(st.session_state.get("provider_allow_remote"))
    client = OpenAICompatibleClient(api_key=key, base_url=base, model=model) if key and base and allow_remote else OpenAICompatibleClient.from_env() if not key else None
    return ClaimScopeService(llm_client=client)


def workflow_rail(report: dict) -> str:
    steps = report.get("workflow_steps", [])
    items = []
    for index, label in enumerate(WORKFLOW_LABELS, 1):
        status = str(steps[index - 1].get("status", "complete")) if index <= len(steps) else "complete"
        items.append(
            f'<div class="workflow-step"><span class="workflow-step-num">{index}</span>'
            f'<strong>{escape(label)}</strong><span>{escape(status)}</span></div>'
        )
    return '<div class="workflow-wrap"><div class="workflow-rail">' + "".join(items) + "</div></div>"


def empty_state(title: str, detail: str) -> str:
    return (
        '<div class="empty-state"><strong>'
        + escape(title)
        + "</strong><p>"
        + escape(detail)
        + "</p></div>"
    )


def render_header() -> str:
    has_provider = bool(st.session_state.get("provider_key") and st.session_state.get("provider_base"))
    health = "configured" if has_provider and st.session_state.get("provider_allow_remote") else "local only" if has_provider else "demo / heuristic"
    with st.container(border=True):
        brand_column, mode_column, provider_column = st.columns([0.8, 1.35, 1.15], vertical_alignment="center")
        with brand_column:
            st.markdown('<div class="brand-lockup"><div class="brand">ClaimScope</div><div class="brand-kicker">Research claim audit</div></div>', unsafe_allow_html=True)
        with mode_column:
            if hasattr(st, "segmented_control"):
                mode = st.segmented_control("Mode", ["Core Claim Arena", "Full Discovery"], default="Core Claim Arena", label_visibility="collapsed")
            else:
                mode = st.radio("Mode", ["Core Claim Arena", "Full Discovery"], horizontal=True, label_visibility="collapsed")
        with provider_column:
            status_column, settings_column = st.columns([1.7, 1], vertical_alignment="center")
            with status_column:
                st.markdown(f'<div class="provider-status"><span class="provider-dot" aria-hidden="true"></span><span>{escape(health)}</span></div>', unsafe_allow_html=True)
            with settings_column:
                with st.popover("Provider", use_container_width=True):
                    st.text_input("API key", type="password", key="provider_key", help="Session-only; never exported or logged.")
                    st.text_input("Base URL", key="provider_base")
                    st.text_input("Model", key="provider_model")
                    st.checkbox(
                        "Allow research directions to be sent to this provider",
                        key="provider_allow_remote",
                    )
                    st.caption(f"Host: {provider_host(st.session_state.get('provider_base', ''))}")
                    st.caption("LLM mode sends the research direction to the configured provider.")
    return mode or "Core Claim Arena"


def render_research_controls(mode: str) -> tuple[str, bool, int, bool]:
    online = False
    online_consent = True
    limit = 12
    with st.container(border=True):
        st.markdown('<div class="panel-title">Research direction</div><p class="muted">Describe the fuzzy direction you want to stress-test.</p>', unsafe_allow_html=True)
        direction = st.text_area("Research direction", key="direction", height=130, label_visibility="collapsed")
        st.caption("Examples: retrieval quality and hallucination; causal mechanism and measurable outcome.")
        if mode == "Core Claim Arena":
            st.caption("Retrieval controls are disabled in Core Claim mode.")
        else:
            online = st.toggle("Use online academic retrieval", value=False)
            if online:
                st.warning("Online retrieval sends generated search queries to public academic APIs.")
                online_consent = st.checkbox(
                    "Allow generated queries to be sent to arXiv and Semantic Scholar"
                )
            limit = st.slider("Maximum papers", 5, 30, 12)
        run = st.button("Run Arena" if mode == "Core Claim Arena" else "Run Discovery", type="primary", use_container_width=True, disabled=online and not online_consent)
    return direction, online, limit, run


def render_core_claim(result: dict | None) -> None:
    if not result:
        st.markdown(empty_state("No claim analyzed", "Enter a research direction and run the arena."), unsafe_allow_html=True)
        return
    selected = result.get("selected_candidate") or {}
    confidence = float(selected.get("confidence", 0))
    st.markdown(f"""<section class="panel"><div class="eyebrow">Selected artifact</div><div class="panel-title">Core Claim</div><div class="selected-claim">{escape(str(result.get('selected_claim', '')))}</div><div class="metrics">{''.join([f'<div class="metric"><span class="metric-label">Candidate score</span><strong>{confidence:.2f}</strong><small>Structural completeness: {confidence_label(confidence)}</small></div>', f'<div class="metric"><span class="metric-label">Mode</span><strong>{escape(str(result.get("mode", "heuristic")))}</strong></div>', f'<div class="metric"><span class="metric-label">Falsification test</span><strong>{escape(str(selected.get("falsification_test", "Not specified")))}</strong></div>', f'<div class="metric"><span class="metric-label">Open slots</span><strong>{len(result.get("unresolved_ambiguities", []))}</strong></div>'])}</div></section>""", unsafe_allow_html=True)
    open_slots = [escape(str(item)) for item in result.get("unresolved_ambiguities", [])]
    if open_slots:
        st.markdown(
            f'<div class="warning-box"><strong>Open claim slots:</strong> {"; ".join(open_slots)}</div>',
            unsafe_allow_html=True,
        )
    if result.get("degraded"):
        st.markdown('<div class="warning-box">This run used a deterministic fallback for one or more agent stages.</div>', unsafe_allow_html=True)
    st.markdown('<section class="panel"><div class="panel-title">Agent activity</div>' + render_activity(result) + '</section>', unsafe_allow_html=True)
    candidates = result.get("candidates", [])
    st.markdown('<section class="panel"><div class="panel-title">Candidate comparison</div>' + "".join(render_candidate(candidate, candidate is selected or candidate.get("claim") == selected.get("claim"), index) for index, candidate in enumerate(candidates, 1)) + '</section>', unsafe_allow_html=True)
    st.markdown('<section class="panel"><div class="panel-title">Critique summary</div>' + render_critique_summary(result) + '</section>', unsafe_allow_html=True)


def render_discovery(report: dict | None) -> None:
    if not report:
        st.markdown(empty_state("No discovery report", "Run the workflow to build an assumption and evidence map."), unsafe_allow_html=True)
        return
    st.markdown(f'<section class="panel"><div class="eyebrow">Anchor artifact</div><div class="panel-title">Selected Core Claim</div><div class="selected-claim">{escape(str(report.get("claim", "")))}</div></section>', unsafe_allow_html=True)
    st.markdown(workflow_rail(report), unsafe_allow_html=True)
    overview, ledger, evidence, opportunities, trace, export = st.tabs(["Overview", "Assumption Ledger", "Evidence", "Opportunities", "Trace", "Export"])
    with overview:
        assumptions = len(report.get("assumptions", []))
        papers = len(report.get("papers", []))
        opportunity_count = len(report.get("idea_opportunities", []))
        st.markdown(
            '<div class="overview-metrics">'
            f'<div class="overview-metric"><b>{assumptions}</b><span>Testable assumptions</span></div>'
            f'<div class="overview-metric"><b>{papers}</b><span>Retrieved papers</span></div>'
            f'<div class="overview-metric"><b>{opportunity_count}</b><span>Opportunity slots</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        for warning in report.get("warnings", []): st.markdown(f'<div class="warning-box">{escape(str(warning))}</div>', unsafe_allow_html=True)
    with ledger:
        for index, assumption in enumerate(report.get("assumptions", []), 1):
            status = str(assumption.get("status", "unknown"))
            risk = str(assumption.get("risk", "medium"))
            evidence_count = len(assumption.get("evidence", []))
            st.markdown(
                f'<article class="artifact"><div class="artifact-head"><strong class="artifact-title">{index}. {escape(str(assumption.get("text", "")))}</strong>'
                f'<div class="badge-row"><span class="badge badge-{escape(status)}">signal: {escape(status)}</span><span class="badge badge-{escape(risk)}">risk: {escape(risk)}</span><span class="badge">{evidence_count} evidence</span></div></div></article>',
                unsafe_allow_html=True,
            )
    with evidence:
        st.markdown(render_discovery_evidence(report), unsafe_allow_html=True)
    with opportunities:
        for index, opportunity in enumerate(report.get("idea_opportunities", []), 1):
            kind = str(opportunity.get("kind", "opportunity")).replace("_", " ")
            score = opportunity.get("score", "-")
            st.markdown(
                f'<article class="artifact"><div class="artifact-head"><strong class="artifact-title">{index}. {escape(str(opportunity.get("title", "")))}</strong>'
                f'<div class="badge-row"><span class="badge badge-opportunity">{escape(kind)}</span><span class="badge">score {escape(str(score))}</span></div></div>'
                f'<p>{escape(str(opportunity.get("rationale", "")))}</p><p class="artifact-next"><strong>Next:</strong> {escape(str(opportunity.get("next_step", "")))}</p></article>',
                unsafe_allow_html=True,
            )
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
    mode = render_header()
    left, right = st.columns([1, 1.35])
    with left:
        direction, online, limit, run = render_research_controls(mode)
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

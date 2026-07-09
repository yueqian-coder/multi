from __future__ import annotations

from html import escape

import streamlit as st

from claimscope.cli import DEMO_PAPERS
from claimscope.llm import OpenAICompatibleClient
from claimscope.models import AnalysisReport, Assumption, IdeaOpportunity
from claimscope.pipeline import ClaimScopePipeline
from claimscope.planner import HeuristicClaimPlanner, LLMClaimPlanner
from claimscope.retrievers import (
    ArxivRetriever,
    CombinedRetriever,
    SemanticScholarRetriever,
    StaticPaperRetriever,
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #667085;
            --line: #d9e2ec;
            --surface: #ffffff;
            --soft: #f5f8fb;
            --teal: #007c89;
            --indigo: #4f46e5;
            --amber: #b45309;
            --red: #c2410c;
            --green: #047857;
        }
        .stApp {
            background:
                linear-gradient(180deg, #f8fbfd 0%, #eef4f7 100%);
            color: var(--ink);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.35rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: 0;
        }
        div.stButton > button:first-child {
            background: linear-gradient(135deg, var(--teal), var(--indigo));
            color: #ffffff;
            border: 0;
            border-radius: 8px;
            min-height: 46px;
            font-weight: 800;
            box-shadow: 0 12px 26px rgba(0, 124, 137, 0.22);
        }
        div.stButton > button:first-child:hover {
            color: #ffffff;
            border: 0;
            filter: brightness(0.98);
        }
        button[data-baseweb="tab"] {
            font-weight: 700;
        }
        textarea {
            border-radius: 8px !important;
            border-color: var(--line) !important;
            background: #ffffff !important;
        }
        .hero {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 26px 28px;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.07);
            margin-bottom: 18px;
        }
        .hero h1 {
            font-size: 42px;
            line-height: 1.05;
            margin: 0 0 8px 0;
            letter-spacing: 0;
        }
        .hero p {
            color: var(--muted);
            font-size: 16px;
            margin: 0;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 4px 0;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .metric-value {
            color: var(--ink);
            font-size: 27px;
            line-height: 1.15;
            font-weight: 800;
            margin-top: 6px;
        }
        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 10px;
            margin: 4px 0 18px 0;
        }
        .workflow-step {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 13px 12px;
            min-height: 154px;
        }
        .workflow-index {
            width: 26px;
            height: 26px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--teal);
            color: white;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 9px;
        }
        .workflow-title {
            font-size: 13px;
            font-weight: 800;
            color: var(--ink);
            line-height: 1.25;
            min-height: 34px;
        }
        .workflow-output {
            color: var(--muted);
            font-size: 12px;
            line-height: 1.35;
            margin-top: 8px;
        }
        .callout {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 16px 18px;
            margin: 10px 0 16px 0;
        }
        .section-title {
            font-size: 20px;
            font-weight: 850;
            letter-spacing: 0;
            margin: 8px 0 6px 0;
        }
        .opportunity-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
        }
        .opportunity-head {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: flex-start;
        }
        .opportunity-title {
            font-size: 15px;
            font-weight: 820;
            color: var(--ink);
            line-height: 1.35;
        }
        .score-pill {
            min-width: 62px;
            text-align: center;
            border-radius: 8px;
            padding: 8px 10px;
            color: #ffffff;
            background: linear-gradient(135deg, var(--indigo), var(--teal));
            font-weight: 850;
        }
        .small-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            margin-top: 10px;
        }
        .body-copy {
            color: #344054;
            font-size: 13px;
            line-height: 1.55;
            margin-top: 6px;
        }
        .status-supported { color: var(--green); font-weight: 800; }
        .status-mixed { color: var(--amber); font-weight: 800; }
        .status-unsupported { color: var(--red); font-weight: 800; }
        .status-unknown { color: var(--indigo); font-weight: 800; }
        @media (max-width: 980px) {
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .workflow-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hero h1 { font-size: 34px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(report: AnalysisReport) -> None:
    evidence_cards = sum(len(item.evidence) for item in report.assumptions)
    html = f"""
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-label">Claim Variants</div><div class="metric-value">{len(report.claim_variants)}</div></div>
        <div class="metric-card"><div class="metric-label">Assumptions</div><div class="metric-value">{len(report.assumptions)}</div></div>
        <div class="metric-card"><div class="metric-label">Evidence Cards</div><div class="metric-value">{evidence_cards}</div></div>
        <div class="metric-card"><div class="metric-label">Opportunities</div><div class="metric-value">{len(report.idea_opportunities)}</div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def workflow_trace(report: AnalysisReport) -> None:
    cards = []
    for idx, step in enumerate(report.workflow_steps(), 1):
        cards.append(
            f"""
            <div class="workflow-step">
                <div class="workflow-index">{idx}</div>
                <div class="workflow-title">{escape(step.name)}</div>
                <div class="workflow-output">{escape(step.output)}</div>
            </div>
            """
        )
    st.markdown('<div class="workflow-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def status_class(status: str) -> str:
    return {
        "supported": "status-supported",
        "mixed": "status-mixed",
        "unsupported": "status-unsupported",
        "unknown": "status-unknown",
    }.get(status, "status-unknown")


def render_opportunity_card(opportunity: IdeaOpportunity, rank: int) -> None:
    linked = ", ".join(opportunity.linked_evidence) if opportunity.linked_evidence else "No linked evidence yet"
    st.markdown(
        f"""
        <div class="opportunity-card">
            <div class="opportunity-head">
                <div>
                    <div class="small-label">Opportunity {rank} / {escape(opportunity.kind.replace("_", " "))}</div>
                    <div class="opportunity-title">{escape(opportunity.title)}</div>
                </div>
                <div class="score-pill">{opportunity.score}</div>
            </div>
            <div class="small-label">Why</div>
            <div class="body-copy">{escape(opportunity.rationale)}</div>
            <div class="small-label">Next Step</div>
            <div class="body-copy">{escape(opportunity.next_step)}</div>
            <div class="small-label">Linked Evidence</div>
            <div class="body-copy">{escape(linked)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assumption(assumption: Assumption) -> None:
    label = f"{assumption.status.upper()} | risk: {assumption.risk}"
    with st.expander(f"{assumption.text}  -  {label}", expanded=False):
        st.markdown(
            f"<span class='{status_class(assumption.status)}'>{escape(label)}</span>",
            unsafe_allow_html=True,
        )
        st.write(
            {
                "support": assumption.support_query,
                "contradict": assumption.contradict_query,
                "limitation": assumption.limitation_query,
                "null_result": assumption.null_result_query,
            }
        )
        if assumption.evidence:
            for evidence in assumption.evidence:
                st.markdown(
                    f"**[{evidence.stance}] {evidence.paper_title} ({evidence.year})**"
                )
                st.caption(evidence.snippet)
        else:
            st.caption("No direct evidence found in the retrieved set.")


st.set_page_config(page_title="ClaimScope", page_icon="CS", layout="wide")
inject_style()

llm_client = OpenAICompatibleClient.from_env()

with st.sidebar:
    st.markdown("### Planning")
    planner_options = ["Heuristic planner"]
    if llm_client:
        planner_options = ["LLM planner", "Heuristic planner"]
    planner_mode = st.radio(
        "Claim planner",
        planner_options,
        index=0,
        help="The LLM planner uses your configured OpenAI-compatible endpoint.",
    )
    if llm_client:
        st.success("API planner available")
        st.caption("LLM mode sends the research direction to your configured endpoint.")
    else:
        st.warning("API planner not configured")
        st.caption("Set OPENAI_API_KEY and OPENAI_BASE_URL to enable LLM planning.")
    workflow_mode = st.radio(
        "Test module",
        ["Core Claim Test", "Full Discovery"],
        index=0,
        help="Core Claim Test runs only Research Direction -> Core Claim.",
    )

    st.markdown("### Retrieval")
    mode = st.radio(
        "Paper source",
        ["Demo papers", "arXiv + Semantic Scholar"],
        help="Demo mode is deterministic. Online mode uses public academic search APIs.",
    )
    limit = st.slider("Max papers", 5, 30, 12)
    st.markdown("### Scope")
    st.selectbox(
        "Research stage",
        ["Pre-ideation / exploration", "Claim stress test", "Opportunity ranking"],
    )
    st.selectbox(
        "Domain lens",
        ["All domains", "AI / CS", "Medical AI", "RAG / GraphRAG"],
    )

st.markdown(
    """
    <div class="hero">
        <h1>ClaimScope</h1>
        <p>Assumption-Centric Research Discovery / Claim-to-Opportunity Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

claim = st.text_area(
    "Research direction",
    value="RAG can reliably reduce hallucination in LLM-generated answers",
    height=105,
)

button_label = "Extract Core Claim" if workflow_mode == "Core Claim Test" else "Run Discovery"
analyze = st.button(button_label, type="primary", use_container_width=True)

if analyze:
    if not claim.strip():
        st.warning("Enter a research direction or claim first.")
        st.stop()

    planner = HeuristicClaimPlanner()
    if planner_mode == "LLM planner" and llm_client:
        planner = LLMClaimPlanner(llm_client=llm_client)

    if workflow_mode == "Core Claim Test":
        with st.spinner("Extracting core claim only..."):
            core_claim = ClaimScopePipeline(
                retriever=StaticPaperRetriever([]),
                planner=planner,
            ).extract_core_claim(claim)

        if isinstance(planner, LLMClaimPlanner):
            if planner.used_planner == "heuristic":
                st.warning(planner.fallback_reason)
            else:
                st.info("LLM Core Claim module used. Downstream modules did not run.")

        st.markdown(
            f"""
            <div class="callout">
                <div class="small-label">Input Research Direction</div>
                <div class="body-copy">{escape(claim)}</div>
            </div>
            <div class="callout">
                <div class="small-label">Core Claim</div>
                <div class="body-copy">{escape(core_claim)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Module Test Status")
        st.table(
            [
                {"Module": "Research Direction", "Status": "ran", "Output": claim},
                {"Module": "Core Claim", "Status": "ran", "Output": core_claim},
                {"Module": "Claim Variants / Boundary Conditions", "Status": "off", "Output": ""},
                {"Module": "Hidden Assumptions", "Status": "off", "Output": ""},
                {"Module": "Evidence Queries", "Status": "off", "Output": ""},
                {"Module": "Evidence Cards", "Status": "off", "Output": ""},
                {"Module": "Idea Opportunities", "Status": "off", "Output": ""},
            ]
        )
        st.markdown("#### Your Evaluation")
        st.code(
            "Input:\n"
            f"{claim}\n\n"
            "Observed Core Claim:\n"
            f"{core_claim}\n\n"
            "Expected Core Claim:\n\n"
            "Pass / Partial / Fail:\n\n"
            "Notes:\n",
            language="text",
        )
        st.stop()

    if mode == "arXiv + Semantic Scholar":
        retriever = CombinedRetriever([SemanticScholarRetriever(), ArxivRetriever()])
    else:
        retriever = StaticPaperRetriever(DEMO_PAPERS)

    with st.spinner("Running assumption-centric discovery workflow..."):
        report = ClaimScopePipeline(retriever=retriever, planner=planner).analyze(
            claim, limit=limit
        )

    if isinstance(planner, LLMClaimPlanner):
        if planner.used_planner == "heuristic":
            st.warning(planner.fallback_reason)
        else:
            st.info("LLM planner used for claim decomposition and query generation.")

    st.markdown(
        f"""
        <div class="callout">
            <div class="small-label">Core Claim</div>
            <div class="body-copy">{escape(report.claim)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cards(report)

    workflow_tab, matrix_tab, assumptions_tab, opportunities_tab, evidence_tab, markdown_tab = st.tabs(
        [
            "Workflow Trace",
            "Evidence Matrix",
            "Assumptions",
            "Opportunities",
            "Evidence",
            "Markdown",
        ]
    )

    with workflow_tab:
        st.markdown('<div class="section-title">Claim-to-Opportunity Workflow</div>', unsafe_allow_html=True)
        workflow_trace(report)
        left, right = st.columns([1.2, 0.8])
        with left:
            st.markdown("#### Claim Variants / Boundary Conditions")
            for variant in report.claim_variants:
                st.markdown(f"**{variant.text}**")
                st.caption(variant.rationale)
        with right:
            st.markdown("#### Top Opportunity Signals")
            for idx, opportunity in enumerate(report.idea_opportunities[:3], 1):
                render_opportunity_card(opportunity, idx)

    with matrix_tab:
        st.markdown('<div class="section-title">Assumption Evidence Matrix</div>', unsafe_allow_html=True)
        st.table(report.assumption_matrix())

    with assumptions_tab:
        st.markdown('<div class="section-title">Hidden Assumptions and Query Matrix</div>', unsafe_allow_html=True)
        for assumption in report.assumptions:
            render_assumption(assumption)

    with opportunities_tab:
        st.markdown('<div class="section-title">Ranked Idea Opportunities</div>', unsafe_allow_html=True)
        for idx, opportunity in enumerate(report.idea_opportunities, 1):
            render_opportunity_card(opportunity, idx)

    with evidence_tab:
        paper_col, negative_col = st.columns([1.1, 0.9])
        with paper_col:
            st.markdown("#### Retrieved Papers")
            if report.papers:
                for paper in report.papers:
                    st.markdown(f"**{paper.title}** ({paper.year or 'n.d.'})")
                    st.caption(f"{paper.source} | {', '.join(paper.authors[:3])}")
                    st.write(paper.abstract)
                    if paper.url:
                        st.markdown(f"[Open paper]({paper.url})")
                    st.divider()
            else:
                st.info("No papers were retrieved.")
        with negative_col:
            st.markdown("#### Negative Evidence")
            if report.negative_evidence:
                for item in report.negative_evidence:
                    st.markdown(f"**{item.kind.replace('_', ' ').title()}**")
                    st.write(f"{item.paper_title} ({item.year}): {item.text}")
                    st.caption(item.implication)
                    st.divider()
            else:
                st.info("No explicit negative evidence was found.")

    with markdown_tab:
        markdown = report.to_markdown()
        st.download_button(
            "Download Markdown Report",
            data=markdown,
            file_name="claimscope_report.md",
            mime="text/markdown",
        )
        st.code(markdown, language="markdown")
else:
    st.markdown(
        """
        <div class="callout">
            <div class="small-label">Current Test Mode</div>
            <div class="body-copy">Start with Core Claim Test. It runs only Research Direction -> Core Claim; downstream modules stay off until you switch to Full Discovery.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

from __future__ import annotations

import streamlit as st

from claimscope.cli import DEMO_PAPERS
from claimscope.llm import OpenAICompatibleClient
from claimscope.pipeline import ClaimScopePipeline
from claimscope.planner import HeuristicClaimPlanner, LLMClaimPlanner
from claimscope.retrievers import (
    ArxivRetriever,
    CombinedRetriever,
    SemanticScholarRetriever,
    StaticPaperRetriever,
)


st.set_page_config(page_title="ClaimScope", page_icon="CS", layout="wide")

st.title("ClaimScope")
st.caption(
    "Pre-ideation claim lineage, assumption-gap, and negative-evidence mapping for research directions."
)

with st.sidebar:
    st.header("Planning")
    llm_client = OpenAICompatibleClient.from_env()
    planner_options = ["Heuristic planner"]
    if llm_client:
        planner_options.append("LLM planner")
    planner_mode = st.radio(
        "Claim planner",
        planner_options,
        index=0,
        help="The LLM planner creates domain-specific claim variants and assumption queries.",
    )
    if llm_client:
        st.caption("LLM planner is available; selecting it sends the input direction to your configured endpoint.")
    else:
        st.caption("Set OPENAI_API_KEY and OPENAI_BASE_URL to enable the LLM planner.")

    st.header("Retrieval")
    mode = st.radio(
        "Paper source",
        ["Demo papers", "arXiv + Semantic Scholar"],
        help="Demo mode is deterministic. Online mode uses public academic search APIs.",
    )
    limit = st.slider("Max papers", 5, 30, 12)
    st.header("Output Focus")
    st.write("Claim variants, assumption gaps, negative evidence, idea opportunities")

claim = st.text_area(
    "Research direction or claim",
    value="RAG can reliably reduce hallucination in LLM-generated answers",
    height=110,
)

if st.button("Analyze", type="primary"):
    if not claim.strip():
        st.warning("Enter a research direction or claim first.")
        st.stop()

    if mode == "arXiv + Semantic Scholar":
        retriever = CombinedRetriever([SemanticScholarRetriever(), ArxivRetriever()])
    else:
        retriever = StaticPaperRetriever(DEMO_PAPERS)

    planner = HeuristicClaimPlanner()
    if planner_mode == "LLM planner" and llm_client:
        planner = LLMClaimPlanner(llm_client=llm_client)

    with st.spinner("Mapping claim variants, assumptions, and negative evidence..."):
        report = ClaimScopePipeline(retriever=retriever, planner=planner).analyze(
            claim, limit=limit
        )
    if isinstance(planner, LLMClaimPlanner):
        if planner.used_planner == "heuristic":
            st.warning(planner.fallback_reason)
        else:
            st.info("LLM planner used for claim variants and assumption queries.")

    overview, papers, assumptions, negative, opportunities, markdown_tab = st.tabs(
        [
            "Overview",
            "Papers",
            "Assumptions",
            "Negative Evidence",
            "Idea Slots",
            "Markdown",
        ]
    )

    with overview:
        st.subheader("Claim Variants")
        for variant in report.claim_variants:
            st.markdown(f"**{variant.text}**")
            st.caption(variant.rationale)

    with papers:
        for paper in report.papers:
            st.markdown(f"**{paper.title}** ({paper.year or 'n.d.'})")
            st.caption(f"{paper.source} | {', '.join(paper.authors[:3])}")
            st.write(paper.abstract)
            if paper.url:
                st.markdown(f"[Open paper]({paper.url})")
            st.divider()

    with assumptions:
        for assumption in report.assumptions:
            st.markdown(
                f"**{assumption.text}**  \nStatus: `{assumption.status}` | Risk: `{assumption.risk}`"
            )
            with st.expander("Evidence query matrix"):
                st.write(
                    {
                        "support": assumption.support_query,
                        "contradict": assumption.contradict_query,
                        "limitation": assumption.limitation_query,
                        "null_result": assumption.null_result_query,
                    }
                )
            for evidence in assumption.evidence:
                st.write(
                    f"- [{evidence.stance}] {evidence.paper_title} ({evidence.year}): {evidence.snippet}"
                )
            st.divider()

    with negative:
        for item in report.negative_evidence:
            st.markdown(f"**{item.kind.replace('_', ' ').title()}**")
            st.write(f"{item.paper_title} ({item.year}): {item.text}")
            st.caption(item.implication)
            st.divider()

    with opportunities:
        for opportunity in report.idea_opportunities:
            st.markdown(f"**{opportunity.title}**")
            st.caption(opportunity.kind)
            st.write(opportunity.rationale)
            st.info(opportunity.next_step)
            if opportunity.linked_evidence:
                st.write("Linked evidence:", ", ".join(opportunity.linked_evidence))
            st.divider()

    with markdown_tab:
        st.download_button(
            "Download Markdown Report",
            data=report.to_markdown(),
            file_name="claimscope_report.md",
            mime="text/markdown",
        )
        st.code(report.to_markdown(), language="markdown")
else:
    st.info("Enter a claim and click Analyze. Demo mode works offline.")

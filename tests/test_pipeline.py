from claimscope.models import Paper
from claimscope.pipeline import ClaimScopePipeline
from claimscope.retrievers import StaticPaperRetriever
from claimscope.llm import OpenAICompatibleClient


def sample_papers():
    return [
        Paper(
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            year=2020,
            authors=["Lewis et al."],
            source="fixture",
            url="https://example.org/rag",
            abstract=(
                "Retrieval-augmented generation improves factuality for knowledge-intensive "
                "question answering by conditioning answers on retrieved evidence. However, "
                "performance depends strongly on retrieval quality, and noisy passages can "
                "hurt generation."
            ),
        ),
        Paper(
            title="Evaluating Retrieval-Augmented Language Models for Factuality",
            year=2023,
            authors=["Chen et al."],
            source="fixture",
            url="https://example.org/eval-rag",
            abstract=(
                "RAG can reduce hallucination on open-domain QA when evidence is relevant. "
                "The gains are smaller under domain shift, and citation mismatch remains a "
                "common failure mode."
            ),
        ),
        Paper(
            title="On the Limits of Retrieval-Augmented Generation",
            year=2024,
            authors=["Patel et al."],
            source="fixture",
            url="https://example.org/rag-limits",
            abstract=(
                "We find no consistent improvement for long-form generation when retrieved "
                "documents are irrelevant. Limitations include retrieval noise, unsupported "
                "claims, and brittle evaluation metrics."
            ),
        ),
    ]


def build_report():
    pipeline = ClaimScopePipeline(retriever=StaticPaperRetriever(sample_papers()))
    return pipeline.analyze(
        "RAG can reliably reduce hallucination in LLM-generated answers"
    )


def test_pipeline_maps_assumptions_to_traceable_evidence_and_negative_findings():
    report = build_report()

    assert report.claim
    assert len(report.claim_variants) >= 2
    assert report.assumptions
    assert report.negative_evidence
    assert any(
        assumption.status in {"mixed", "unsupported", "unknown"}
        for assumption in report.assumptions
    )
    assert any(
        finding.kind in {"limitation", "failure_mode", "negative_result"}
        for finding in report.negative_evidence
    )
    assert all(
        evidence.paper_title
        for assumption in report.assumptions
        for evidence in assumption.evidence
    )


def test_pipeline_generates_actionable_opportunities_from_gaps():
    report = build_report()

    assert report.idea_opportunities
    assert any(
        opportunity.kind in {"assumption_gap", "negative_evidence", "boundary_condition"}
        for opportunity in report.idea_opportunities
    )
    assert any(
        "experiment" in opportunity.next_step.lower()
        or "dataset" in opportunity.next_step.lower()
        or "evaluate" in opportunity.next_step.lower()
        for opportunity in report.idea_opportunities
    )


def test_report_markdown_contains_pre_ideation_traceability_sections():
    report = build_report()
    markdown = report.to_markdown()

    assert "Claim Variants" in markdown
    assert "Assumption Gaps" in markdown
    assert "Negative Evidence" in markdown
    assert "Idea Opportunities" in markdown
    assert "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" in markdown


def test_report_uses_readable_claim_text_in_generated_sections():
    report = build_report()
    markdown = report.to_markdown()

    assert "rag reliably reduce" not in markdown.lower()
    assert "answers works only" not in markdown
    assert "RAG can reliably reduce hallucination" in markdown


def test_unrelated_claim_does_not_attach_irrelevant_demo_evidence():
    pipeline = ClaimScopePipeline(retriever=StaticPaperRetriever(sample_papers()))
    report = pipeline.analyze(
        "Diffusion models improve MRI tumor segmentation with limited labels"
    )

    attached_evidence = [
        evidence
        for assumption in report.assumptions
        for evidence in assumption.evidence
    ]
    assert attached_evidence == []
    assert any(assumption.status == "unknown" for assumption in report.assumptions)
    assert any(
        opportunity.kind in {"assumption_gap", "boundary_condition"}
        for opportunity in report.idea_opportunities
    )


def test_neutral_mentions_do_not_count_as_negative_evidence():
    papers = [
        Paper(
            title="A Survey of Retrieval Systems",
            year=2022,
            authors=["Ng et al."],
            source="fixture",
            abstract="This paper surveys retrieval systems and generation models.",
        )
    ]
    report = ClaimScopePipeline(retriever=StaticPaperRetriever(papers)).analyze(
        "Retrieval systems improve factual generation"
    )

    assert all(
        evidence.stance != "mention"
        for assumption in report.assumptions
        for evidence in assumption.evidence
    )
    assert all(
        assumption.status in {"supported", "unknown"}
        for assumption in report.assumptions
    )


def test_openai_client_requires_explicit_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert OpenAICompatibleClient.from_env() is None

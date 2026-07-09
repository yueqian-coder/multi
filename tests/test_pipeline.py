from claimscope.models import Paper
from claimscope.pipeline import ClaimScopePipeline
from claimscope.planner import HeuristicClaimPlanner, LLMClaimPlanner
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


class ExplodingRetriever:
    def search(self, query: str, limit: int = 20):
        raise AssertionError("Core Claim tests must not call paper retrieval.")


def test_extract_core_claim_does_not_run_retrieval_or_downstream_modules():
    pipeline = ClaimScopePipeline(retriever=ExplodingRetriever())

    claim = pipeline.extract_core_claim(
        "  RAG can reliably reduce hallucination in LLM-generated answers. "
    )

    assert claim == "RAG can reliably reduce hallucination in LLM-generated answers"


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
    scores = [opportunity.score for opportunity in report.idea_opportunities]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)
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

    assert "Workflow Trace" in markdown
    assert "Assumption Evidence Matrix" in markdown
    assert "Claim Variants" in markdown
    assert "Assumption Gaps" in markdown
    assert "Negative Evidence" in markdown
    assert "Idea Opportunities" in markdown
    assert "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" in markdown


def test_report_exposes_seven_step_workflow_trace():
    report = build_report()
    steps = report.workflow_steps()

    assert [step.name for step in steps] == [
        "Research Direction",
        "Core Claim",
        "Claim Variants / Boundary Conditions",
        "Hidden Assumptions",
        "Evidence Queries",
        "Evidence Cards",
        "Idea Opportunities",
    ]
    assert all(step.status == "complete" for step in steps)
    assert steps[0].output == report.query
    assert steps[-1].artifact_count == len(report.idea_opportunities)


def test_assumption_matrix_counts_adversarial_evidence_buckets():
    report = build_report()
    matrix = report.assumption_matrix()

    assert matrix
    assert any(row["Support"] > 0 for row in matrix)
    assert any(row["Contradict"] > 0 for row in matrix)
    assert any(row["Limitation"] > 0 for row in matrix)
    assert any(row["Null Result"] > 0 for row in matrix)
    assert all("Opportunity Signal" in row for row in matrix)


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


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def chat(self, messages, temperature=0.2):
        self.calls.append({"messages": messages, "temperature": temperature})
        return self.response


class FailingLLMClient:
    def chat(self, messages, temperature=0.2):
        raise RuntimeError("gateway unavailable")


def test_llm_core_claim_extraction_accepts_core_claim_only_json():
    planner = LLMClaimPlanner(
        llm_client=FakeLLMClient(
            '{"normalized_claim": "Error maps can guide lightweight refinement for promptable medical segmentation"}'
        ),
        fallback=HeuristicClaimPlanner(),
    )

    claim = ClaimScopePipeline(
        retriever=ExplodingRetriever(),
        planner=planner,
    ).extract_core_claim("use error map + loop to improve promptable medical segmentation")

    assert claim == (
        "Error maps can guide lightweight refinement for promptable medical segmentation"
    )
    assert planner.used_planner == "llm"
    assert planner.llm_client.calls[0]["temperature"] == 0.1


def test_llm_planner_supplies_domain_specific_claim_variants_and_assumptions():
    response = """
    ```json
    {
      "normalized_claim": "Diffusion models improve MRI tumor segmentation with limited labels",
      "claim_variants": [
        {
          "text": "Diffusion-based augmentation improves limited-label MRI tumor segmentation",
          "rationale": "Separates data generation from segmentation-model architecture."
        },
        {
          "text": "Label scarcity is the boundary condition for the proposed gain",
          "rationale": "Makes the setting testable before proposing a method."
        },
        {
          "text": "External validation is required before claiming clinical robustness",
          "rationale": "Separates internal benchmark gains from deployable evidence."
        }
      ],
      "assumptions": [
        {
          "text": "Limited-label segmentation gains are not caused by data leakage or duplicated patients.",
          "support_query": "diffusion augmentation MRI tumor segmentation limited labels patient split improvement",
          "contradict_query": "diffusion augmentation MRI segmentation data leakage duplicated patients no improvement",
          "limitation_query": "diffusion MRI tumor segmentation limited labels limitation external validation",
          "null_result_query": "diffusion MRI segmentation limited labels null result no significant improvement"
        },
        {
          "text": "Synthetic MRI samples preserve tumor morphology needed for segmentation.",
          "support_query": "diffusion synthetic MRI tumor morphology segmentation support",
          "contradict_query": "diffusion synthetic MRI tumor morphology artifacts segmentation worse",
          "limitation_query": "synthetic MRI diffusion tumor segmentation artifacts limitation",
          "null_result_query": "synthetic MRI tumor segmentation diffusion null result"
        },
        {
          "text": "The improvement holds on external hospitals and scanners.",
          "support_query": "diffusion MRI tumor segmentation external validation scanner hospital improvement",
          "contradict_query": "diffusion MRI tumor segmentation external validation domain shift worse",
          "limitation_query": "MRI tumor segmentation diffusion domain shift scanner limitation",
          "null_result_query": "diffusion MRI tumor segmentation external validation no improvement"
        },
        {
          "text": "Reported Dice gains remain meaningful against strong augmentation baselines.",
          "support_query": "diffusion MRI tumor segmentation Dice strong augmentation baseline improvement",
          "contradict_query": "diffusion MRI tumor segmentation strong augmentation baseline no improvement",
          "limitation_query": "diffusion MRI tumor segmentation Dice metric limitation baseline",
          "null_result_query": "diffusion MRI tumor segmentation augmentation baseline null result"
        }
      ]
    }
    ```
    """
    planner = LLMClaimPlanner(
        llm_client=FakeLLMClient(response),
        fallback=HeuristicClaimPlanner(),
    )
    pipeline = ClaimScopePipeline(
        retriever=StaticPaperRetriever([]),
        planner=planner,
    )

    report = pipeline.analyze(
        "Does diffusion help MRI tumor segmentation with few labels?"
    )

    assert report.claim == (
        "Diffusion models improve MRI tumor segmentation with limited labels"
    )
    assert any("Label scarcity" in variant.text for variant in report.claim_variants)
    assert len(report.claim_variants) >= 3
    assert len(report.assumptions) >= 4
    assert report.assumptions[0].text.startswith("Limited-label segmentation gains")
    assert "patient split" in report.assumptions[0].support_query
    assert planner.llm_client.calls[0]["temperature"] == 0.1
    assert planner.used_planner == "llm"
    assert planner.fallback_reason == ""


def test_llm_planner_falls_back_to_heuristic_plan_when_json_is_invalid():
    planner = LLMClaimPlanner(
        llm_client=FakeLLMClient("not json"),
        fallback=HeuristicClaimPlanner(),
    )
    pipeline = ClaimScopePipeline(
        retriever=StaticPaperRetriever([]),
        planner=planner,
    )

    report = pipeline.analyze(
        "Diffusion models improve MRI tumor segmentation with limited labels"
    )

    assert report.claim == (
        "Diffusion models improve MRI tumor segmentation with limited labels"
    )
    assert len(report.assumptions) == 4
    assert any(
        "claimed improvement is real" in assumption.text.lower()
        for assumption in report.assumptions
    )
    assert planner.used_planner == "heuristic"
    assert "invalid" in planner.fallback_reason.lower()


def test_llm_planner_falls_back_when_json_shape_is_incomplete():
    response = """
    {
      "normalized_claim": "Diffusion models improve MRI tumor segmentation",
      "assumptions": [
        {"text": "The dataset split is valid."}
      ]
    }
    """
    planner = LLMClaimPlanner(
        llm_client=FakeLLMClient(response),
        fallback=HeuristicClaimPlanner(),
    )
    report = ClaimScopePipeline(
        retriever=StaticPaperRetriever([]),
        planner=planner,
    ).analyze("Diffusion models improve MRI tumor segmentation")

    assert report.claim == "Diffusion models improve MRI tumor segmentation"
    assert len(report.assumptions) == 4
    assert planner.used_planner == "heuristic"
    assert "shape" in planner.fallback_reason.lower()


def test_llm_planner_falls_back_and_records_reason_when_call_fails():
    planner = LLMClaimPlanner(
        llm_client=FailingLLMClient(),
        fallback=HeuristicClaimPlanner(),
    )
    report = ClaimScopePipeline(
        retriever=StaticPaperRetriever([]),
        planner=planner,
    ).analyze("Retrieval improves factual generation")

    assert report.claim == "Retrieval improves factual generation"
    assert planner.used_planner == "heuristic"
    assert "llm planner failed" in planner.fallback_reason.lower()


def test_openai_client_repr_does_not_expose_api_key():
    client = OpenAICompatibleClient(
        api_key="secret-test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )

    assert "secret-test-key" not in repr(client)


def test_heuristic_planner_keeps_full_width_period_normalization():
    report = ClaimScopePipeline(
        retriever=StaticPaperRetriever([])
    ).analyze("Retrieval improves factual generation。")

    assert report.claim == "Retrieval improves factual generation"

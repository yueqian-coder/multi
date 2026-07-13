import json

import claimscope.pipeline as pipeline_module
import pytest
from claimscope.models import Paper
from claimscope.pipeline import ClaimScopePipeline
from claimscope.planner import HeuristicClaimPlanner, LLMClaimPlanner
from claimscope.retrievers import CombinedRetriever, StaticPaperRetriever
from claimscope.llm import OpenAICompatibleClient
from claimscope.text_utils import split_sentences


EXPECTED_STAGE_NAMES = [
    "research_direction",
    "core_claim",
    "claim_boundaries",
    "hidden_assumptions",
    "evidence_queries",
    "evidence_retrieval",
    "evidence_adjudication",
    "opportunity_synthesis",
    "quality_review",
]


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


def test_report_exposes_stage_events_and_retrieval_limitations():
    report = build_report()

    assert [event.stage for event in report.events] == EXPECTED_STAGE_NAMES
    assert all(event.public_summary for event in report.events)
    assert all(
        event.status in {"complete", "degraded", "failed", "skipped"}
        for event in report.events
    )
    assert "abstract-only" in " ".join(report.warnings).lower()
    assert all("secret" not in event.public_summary.lower() for event in report.events)


class ExplodingPlanner:
    def build(self, query: str):
        raise RuntimeError("secret-token-123 planner private chain-of-thought leaked")


def test_planner_failure_emits_sanitized_failed_event_and_continues():
    report = ClaimScopePipeline(
        retriever=StaticPaperRetriever(sample_papers()),
        planner=ExplodingPlanner(),
    ).analyze("RAG can reliably reduce hallucination in LLM-generated answers")

    assert [event.stage for event in report.events] == EXPECTED_STAGE_NAMES
    assert len(report.events) == len(EXPECTED_STAGE_NAMES)
    stage_counts = {
        stage: [event.stage for event in report.events].count(stage)
        for stage in EXPECTED_STAGE_NAMES
    }
    assert stage_counts == {stage: 1 for stage in EXPECTED_STAGE_NAMES}
    assert report.events[1].stage == "core_claim"
    assert report.events[1].status == "failed"
    assert report.claim == "RAG can reliably reduce hallucination in LLM-generated answers"
    assert report.assumptions
    assert report.idea_opportunities

    public_text = " ".join(
        [event.public_summary for event in report.events] + report.warnings
    ).lower()
    assert "secret-token-123" not in public_text
    assert "chain-of-thought" not in public_text
    assert "planner private" not in public_text
    assert any("planner failed" in warning.lower() for warning in report.warnings)


def test_failed_adjudication_stage_uses_typed_fallbacks_and_continues(monkeypatch):
    def explode_assumptions(*args, **kwargs):
        raise RuntimeError("secret patient identifier from private notes")

    monkeypatch.setattr(pipeline_module, "_build_assumptions", explode_assumptions)

    report = ClaimScopePipeline(StaticPaperRetriever(sample_papers())).analyze(
        "RAG can reliably reduce hallucination in LLM-generated answers"
    )

    assert [event.stage for event in report.events] == EXPECTED_STAGE_NAMES
    assert len(report.events) == len(EXPECTED_STAGE_NAMES)
    adjudication_event = next(
        event for event in report.events if event.stage == "evidence_adjudication"
    )
    assert adjudication_event.status == "failed"
    assert report.assumptions == []
    assert report.idea_opportunities
    assert report.events[-1].stage == "quality_review"

    public_text = " ".join(
        [event.public_summary for event in report.events] + report.warnings
    ).lower()
    assert "secret patient" not in public_text
    assert "private notes" not in public_text
    assert any(
        "evidence adjudication failed" in warning.lower()
        for warning in report.warnings
    )


def test_retrieval_miss_stays_unknown_not_unsupported():
    report = ClaimScopePipeline(StaticPaperRetriever([])).analyze(
        "Diffusion models improve MRI tumor segmentation with limited labels"
    )

    assert report.papers == []
    assert report.assumptions
    assert all(item.status == "unknown" for item in report.assumptions)
    assert any("zero papers" in warning.lower() for warning in report.warnings)


def test_strict_discovery_rejects_zero_retrieved_papers():
    pipeline = ClaimScopePipeline(
        StaticPaperRetriever([]),
        require_retrieval_evidence=True,
    )

    with pytest.raises(pipeline_module.EvidenceRetrievalRequired):
        pipeline.analyze("Diffusion models improve MRI segmentation")


def test_paper_and_evidence_expose_auditable_provenance_defaults():
    report = build_report()

    assert all(paper.external_id for paper in report.papers if paper.url)
    assert all(paper.is_fixture for paper in report.papers)
    assert all(
        evidence.method == "abstract_heuristic"
        for assumption in report.assumptions
        for evidence in assumption.evidence
    )
    assert all(
        evidence.paper_source
        and evidence.matched_query
        and evidence.source_span_start >= 0
        and evidence.is_fixture
        for assumption in report.assumptions
        for evidence in assumption.evidence
    )
    assert any(
        "heuristic abstract matches" in warning.lower()
        for warning in report.warnings
    )
    assert all(
        item.paper_source and item.source_span_start >= 0 and item.is_fixture
        for item in report.negative_evidence
    )


def test_limitation_signal_is_not_treated_as_direct_disproof():
    from claimscope.pipeline import _build_assumptions
    from claimscope.planner import PlannedAssumption

    plan = PlannedAssumption(
        text="Retrieval quality generalizes under domain shift.",
        support_query="retrieval quality domain shift",
        contradict_query="retrieval quality domain shift",
        limitation_query="retrieval quality domain shift",
        null_result_query="retrieval quality domain shift",
    )
    paper = Paper(
        title="Retrieval Quality Boundaries",
        year=2025,
        authors=["A. Researcher"],
        source="fixture",
        abstract="A limitation appears under domain shift for retrieval quality.",
    )

    assumption = _build_assumptions([plan], [paper])[0]

    assert assumption.status == "mixed"
    assert all(item.stance == "limit" for item in assumption.evidence)


def test_targeted_search_deduplicates_by_external_id_before_title():
    papers = [
        Paper(
            title="RAG Factuality Study Preprint",
            year=2024,
            authors=["A"],
            source="fixture",
            url="https://arxiv.org/abs/2401.12345v2",
            abstract="RAG improves factuality and reduces hallucination in question answering.",
        ),
        Paper(
            title="Retitled RAG Factuality Camera Ready",
            year=2024,
            authors=["A"],
            source="fixture",
            url="https://arxiv.org/pdf/2401.12345v2",
            abstract="RAG improves factuality and reduces hallucination in question answering.",
        ),
    ]

    report = ClaimScopePipeline(StaticPaperRetriever(papers)).analyze(
        "RAG improves factuality and reduces hallucination"
    )

    assert len(report.papers) == 1
    assert report.papers[0].external_id == "arxiv:2401.12345"


class SecretFailingRetriever:
    def search(self, query: str, limit: int = 20):
        raise RuntimeError("secret-token-123 backend exploded")


def test_combined_retriever_failures_become_sanitized_pipeline_warnings():
    report = ClaimScopePipeline(
        CombinedRetriever(
            [SecretFailingRetriever(), StaticPaperRetriever(sample_papers())]
        )
    ).analyze("RAG can reliably reduce hallucination in LLM-generated answers")

    joined = " ".join(report.warnings)
    assert report.papers
    assert "SecretFailingRetriever" in joined
    assert "secret-token-123" not in joined


def test_multilingual_and_null_result_markers_classify_conservatively():
    papers = [
        Paper(
            title="中文检索增强研究",
            year=2024,
            authors=["Li"],
            source="fixture",
            abstract=(
                "检索增强生成可以提升问答事实性。"
                "在噪声文档下没有显著提升。"
                "本文还介绍检索系统。"
            ),
        )
    ]

    report = ClaimScopePipeline(StaticPaperRetriever(papers)).analyze(
        "检索增强生成可以提升问答事实性"
    )

    evidence = [
        item
        for assumption in report.assumptions
        for item in assumption.evidence
    ]
    assert any(item.stance == "support" for item in evidence)
    assert any(item.stance == "contradict" for item in evidence)
    assert all("介绍检索系统" not in item.snippet for item in evidence)
    assert any(row["Null Result"] > 0 for row in report.assumption_matrix())


def test_split_sentences_handles_mixed_chinese_and_english_punctuation():
    text = "检索增强生成提升事实性。However, noisy retrieval hurts accuracy.仍需外部验证？Yes!"

    assert split_sentences(text) == [
        "检索增强生成提升事实性。",
        "However, noisy retrieval hurts accuracy.",
        "仍需外部验证？",
        "Yes!",
    ]


def test_normal_heuristic_planner_does_not_warn_about_fallback():
    report = ClaimScopePipeline(StaticPaperRetriever([])).analyze(
        "Retrieval improves factual generation"
    )

    assert not any("heuristic fallback" in warning.lower() for warning in report.warnings)


def test_llm_planner_fallback_warns_about_heuristic_recovery():
    planner = LLMClaimPlanner(
        llm_client=FakeLLMClient("not json"),
        fallback=HeuristicClaimPlanner(),
    )
    report = ClaimScopePipeline(
        retriever=StaticPaperRetriever([]),
        planner=planner,
    ).analyze("Retrieval improves factual generation")

    assert planner.used_planner == "heuristic"
    assert any("heuristic fallback" in warning.lower() for warning in report.warnings)


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


def test_openai_client_reads_explicit_claude_profile_for_mcp(monkeypatch):
    monkeypatch.setenv("CLAIMSCOPE_PROVIDER", "claude")
    monkeypatch.setenv("CLAIMSCOPE_CLAUDE_API_KEY", "claude-test-key")
    monkeypatch.setenv("CLAIMSCOPE_CLAUDE_MODEL", "claude-sonnet-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = OpenAICompatibleClient.from_env()

    assert client is not None
    assert client.api_key == "claude-test-key"
    assert client.model == "claude-sonnet-test"


def test_openai_client_reads_explicit_gpt_profile_for_mcp(monkeypatch):
    monkeypatch.setenv("CLAIMSCOPE_PROVIDER", "gpt")
    monkeypatch.setenv("CLAIMSCOPE_GPT_API_KEY", "gpt-test-key")
    monkeypatch.setenv("CLAIMSCOPE_GPT_MODEL", "gpt-test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = OpenAICompatibleClient.from_env()

    assert client is not None
    assert client.api_key == "gpt-test-key"
    assert client.model == "gpt-test-model"


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


class StrictDiscoveryFakeLLM:
    claim = (
        "Evidence-grounded RAG reduces unsupported medical QA answers versus a "
        "no-retrieval baseline under clinician-reviewed evaluation."
    )

    def chat(self, messages, temperature=0.2, timeout=None):
        content = "\n".join(message["content"] for message in messages)
        if "research-planning module" in content:
            return json.dumps(
                {
                    "normalized_claim": "must be replaced by arena claim",
                    "claim_variants": [
                        {"text": f"Boundary {index}", "rationale": "Test boundary"}
                        for index in range(1, 4)
                    ],
                    "assumptions": [
                        {
                            "text": f"Assumption {index}",
                            "support_query": f"support {index}",
                            "contradict_query": f"contradict {index}",
                            "limitation_query": f"limitation {index}",
                            "null_result_query": f"null result {index}",
                        }
                        for index in range(1, 5)
                    ],
                }
            )
        if "judge" in content:
            return json.dumps(
                {
                    "selected_candidate_id": "operationalizer",
                    "final_claim": self.claim,
                    "falsification_test": "Compare unsupported-answer rates.",
                    "unresolved_ambiguities": [],
                }
            )
        if "critic" in content:
            return json.dumps(
                {
                    "critiques": [
                        {
                            "candidate_id": role,
                            "rubric_scores": {
                                "specificity": 4,
                                "falsifiability": 4,
                                "mechanism": 4,
                                "scope": 4,
                                "measurability": 4,
                                "risk_awareness": 4,
                            },
                            "reason_codes": ["bounded"],
                            "revision": self.claim,
                        }
                        for role in [
                            "operationalizer",
                            "mechanism_analyst",
                            "skeptical_empiricist",
                        ]
                    ]
                }
            )
        return json.dumps(
            {
                "claim": self.claim,
                "method_or_mechanism": "evidence-grounded RAG",
                "target_or_task": "medical QA",
                "expected_effect": "reduces unsupported answers",
                "conditions": ["clinician-reviewed evaluation"],
                "falsification_test": "Compare unsupported-answer rates.",
                "missing_information": [],
                "confidence": 0.8,
            }
        )


def test_strict_discovery_runs_six_role_arena_before_planning():
    streamed = []
    planner = LLMClaimPlanner(
        llm_client=StrictDiscoveryFakeLLM(),
        strict=True,
    )
    report = ClaimScopePipeline(
        retriever=StaticPaperRetriever(sample_papers()),
        planner=planner,
        allow_planner_fallback=False,
        require_retrieval_evidence=True,
    ).analyze(
        "Can RAG make medical QA safer?",
        on_event=streamed.append,
    )

    expected_roles = {
        "operationalizer",
        "mechanism_analyst",
        "skeptical_empiricist",
        "falsifiability_critic",
        "scope_critic",
        "judge",
    }
    assert report.claim == StrictDiscoveryFakeLLM.claim.rstrip(".")
    assert expected_roles <= {event.role for event in streamed}
    assert {
        "core_claim",
        "evidence_retrieval",
        "evidence_adjudication",
        "opportunity_synthesis",
        "quality_review",
    } <= {event.stage for event in streamed}
    retrieval_events = [
        event for event in streamed if event.stage == "evidence_retrieval"
    ]
    assert [event.status for event in retrieval_events] == ["running", "complete"]
    assert expected_roles <= {event.role for event in report.events}


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

from __future__ import annotations

from dataclasses import dataclass, field
import time

from .core_claim import AgentEventCallback
from .models import (
    AgentEvent,
    AnalysisReport,
    Assumption,
    ClaimVariant,
    EvidenceItem,
    IdeaOpportunity,
    NegativeEvidence,
    Paper,
)
from .planner import (
    ClaimPlanner,
    HeuristicClaimPlanner,
    LLMClaimPlanner,
    PlannedAssumption,
    PlannedClaimVariant,
)
from .retrievers import PaperRetriever
from .text_utils import keywords, overlap_score, split_sentences, truncate


class EvidenceRetrievalRequired(RuntimeError):
    pass


SUPPORT_MARKERS = (
    "improve",
    "improves",
    "improved",
    "increase",
    "reduce",
    "reduces",
    "reduced",
    "effective",
    "outperform",
    "benefit",
    "supports",
    "提升",
    "降低",
    "有效",
    "改善",
    "减少",
)

NEGATIVE_MARKERS = (
    "however",
    "limitation",
    "limitations",
    "fail",
    "fails",
    "failure",
    "no consistent",
    "no improvement",
    "hurt",
    "hurts",
    "smaller",
    "brittle",
    "noise",
    "noisy",
    "mismatch",
    "under domain shift",
    "unsupported",
    "risk",
    "not significant",
    "non-significant",
    "negative result",
    "no statistically significant",
    "no significant improvement",
    "限制",
    "失败",
    "无效",
    "噪声",
    "没有显著提升",
)


NULL_RESULT_MARKERS = (
    "no consistent",
    "no improvement",
    "null result",
    "no significant",
    "not significant",
    "non-significant",
    "does not improve",
    "marginal gain",
    "negative result",
    "没有显著提升",
    "无显著提升",
    "无效",
)


@dataclass
class ClaimScopePipeline:
    retriever: PaperRetriever
    planner: ClaimPlanner = field(default_factory=HeuristicClaimPlanner)
    allow_planner_fallback: bool = True
    require_retrieval_evidence: bool = False

    @classmethod
    def from_env(cls, retriever: PaperRetriever) -> "ClaimScopePipeline":
        from .llm import OpenAICompatibleClient

        llm_client = OpenAICompatibleClient.from_env()
        if llm_client:
            planner: ClaimPlanner = LLMClaimPlanner(llm_client=llm_client)
        else:
            planner = HeuristicClaimPlanner()
        return cls(retriever=retriever, planner=planner)

    def analyze(
        self,
        query: str,
        limit: int = 20,
        on_event: AgentEventCallback | None = None,
    ) -> AnalysisReport:
        events: list[AgentEvent] = []
        warnings: list[str] = []
        planner_recovered = False

        started = time.perf_counter()
        _emit_event(
            events,
            "research_direction",
            "user_input",
            started,
            "complete",
            "Captured the public research direction.",
            {"query_characters": len(query)},
            on_event=on_event,
        )

        started = _begin_event(
            on_event,
            "core_claim",
            "claim_planner",
            "Selecting and normalizing a testable core claim.",
        )
        try:
            if isinstance(self.planner, LLMClaimPlanner):
                plan = self.planner.build(query, on_event=on_event)
            else:
                plan = self.planner.build(query)
            claim = plan.claim
            arena_result = getattr(self.planner, "last_core_claim_result", None)
            if arena_result is not None:
                events.extend(arena_result.events)
            planner_fallback = _planner_used_heuristic_fallback(self.planner)
            if planner_fallback:
                warnings.append(
                    "Heuristic fallback used for planning; verify generated assumptions."
                )
            _emit_event(
                events,
                "core_claim",
                "claim_planner",
                started,
                "degraded" if planner_fallback else "complete",
                "Selected a normalized, testable core claim.",
                {"claim_present": bool(claim)},
                on_event=on_event,
            )
        except Exception:
            if not self.allow_planner_fallback:
                raise
            planner_recovered = True
            warnings.append(
                "Planner failed; heuristic fallback used for planning; verify generated assumptions."
            )
            plan = HeuristicClaimPlanner().build(query)
            claim = plan.claim
            _emit_event(
                events,
                "core_claim",
                "claim_planner",
                started,
                "failed",
                "Planner failed; recovered with a conservative heuristic plan.",
                {"claim_present": bool(claim), "recovered": True},
                on_event=on_event,
            )

        started = _begin_event(
            on_event,
            "claim_boundaries",
            "claim_planner",
            "Preparing claim variants and boundary-condition probes.",
        )
        try:
            planned_variants = list(plan.claim_variants)
            boundary_status = "complete" if planned_variants else "skipped"
        except Exception:
            planned_variants = []
            boundary_status = "failed"
            warnings.append(
                "Claim boundary preparation failed; continuing without claim variants."
            )
        _emit_event(
            events,
            "claim_boundaries",
            "claim_planner",
            started,
            boundary_status,
            "Prepared claim variants and boundary-condition probes.",
            {"variant_count": len(planned_variants)},
            on_event=on_event,
        )

        started = _begin_event(
            on_event,
            "hidden_assumptions",
            "claim_planner",
            "Identifying hidden assumptions for evidence checks.",
        )
        try:
            planned_assumptions = list(plan.assumptions)
            assumption_status = "complete" if planned_assumptions else "skipped"
        except Exception:
            planned_assumptions = []
            assumption_status = "failed"
            warnings.append(
                "Hidden assumption preparation failed; continuing without planned assumptions."
            )
        _emit_event(
            events,
            "hidden_assumptions",
            "claim_planner",
            started,
            assumption_status,
            "Identified hidden assumptions to test against retrieved evidence.",
            {"assumption_count": len(planned_assumptions)},
            on_event=on_event,
        )

        started = _begin_event(
            on_event,
            "evidence_queries",
            "query_builder",
            "Generating adversarial evidence queries.",
        )
        try:
            retrieval_queries = [claim]
            for item in planned_assumptions:
                retrieval_queries.extend(item.retrieval_queries)
            evidence_query_status = "complete" if retrieval_queries else "skipped"
        except Exception:
            retrieval_queries = [claim]
            evidence_query_status = "failed"
            warnings.append(
                "Evidence query generation failed; continuing with the core claim only."
            )
        _emit_event(
            events,
            "evidence_queries",
            "query_builder",
            started,
            evidence_query_status,
            "Generated adversarial evidence queries.",
            {"query_count": len([item for item in retrieval_queries if item])},
            on_event=on_event,
        )

        started = _begin_event(
            on_event,
            "evidence_retrieval",
            "paper_retriever",
            "Searching external scholarly indexes and deduplicating papers.",
        )
        retrieval_failed = False
        try:
            papers, retrieval_warnings, retrieval_failed = _targeted_search(
                self.retriever, retrieval_queries, claim, limit
            )
        except Exception:
            papers = []
            retrieval_warnings = [
                "Evidence retrieval failed; continuing without retrieved papers."
            ]
            retrieval_failed = True
        warnings.extend(retrieval_warnings)
        retrieval_status = "complete"
        if retrieval_failed:
            retrieval_status = "failed"
        elif retrieval_warnings or not papers:
            retrieval_status = "degraded"
        _emit_event(
            events,
            "evidence_retrieval",
            "paper_retriever",
            started,
            retrieval_status,
            "Retrieved and deduplicated candidate papers.",
            {"paper_count": len(papers)},
            on_event=on_event,
        )
        if self.require_retrieval_evidence and not papers:
            raise EvidenceRetrievalRequired(
                "Strict discovery requires at least one retrieved paper."
            )

        started = _begin_event(
            on_event,
            "evidence_adjudication",
            "evidence_mapper",
            "Mapping abstract evidence to assumptions conservatively.",
        )
        adjudication_failed = False
        try:
            variants = _build_claim_variants(planned_variants, claim, papers)
        except Exception:
            variants = []
            adjudication_failed = True
            warnings.append(
                "Evidence adjudication failed while building claim variants; continuing without variants."
            )
        try:
            assumptions = _build_assumptions(planned_assumptions, papers)
        except Exception:
            assumptions = []
            adjudication_failed = True
            warnings.append(
                "Evidence adjudication failed while mapping assumptions; continuing without assumption evidence."
            )
        try:
            negative_evidence = _mine_negative_evidence(papers)
        except Exception:
            negative_evidence = []
            adjudication_failed = True
            warnings.append(
                "Evidence adjudication failed while mining negative evidence; continuing without negative evidence."
            )
        direct_evidence_count = sum(
            len(assumption.evidence) for assumption in assumptions
        )
        if adjudication_failed:
            adjudication_status = "failed"
        elif not direct_evidence_count:
            adjudication_status = "degraded"
        else:
            adjudication_status = "complete"
        _emit_event(
            events,
            "evidence_adjudication",
            "evidence_mapper",
            started,
            adjudication_status,
            "Mapped abstract evidence to assumptions with conservative stance labels.",
            {"direct_evidence_count": direct_evidence_count},
            on_event=on_event,
        )

        started = _begin_event(
            on_event,
            "opportunity_synthesis",
            "opportunity_builder",
            "Synthesizing opportunity slots from gaps and negative evidence.",
        )
        opportunity_failed = False
        try:
            opportunities = _build_idea_opportunities(assumptions, negative_evidence)
        except Exception:
            opportunities = []
            opportunity_failed = True
            warnings.append(
                "Opportunity synthesis failed; continuing without synthesized opportunities."
            )
        if opportunity_failed:
            opportunity_status = "failed"
        elif opportunities:
            opportunity_status = "complete"
        else:
            opportunity_status = "skipped"
        _emit_event(
            events,
            "opportunity_synthesis",
            "opportunity_builder",
            started,
            opportunity_status,
            "Synthesized opportunity slots from gaps and negative evidence.",
            {"opportunity_count": len(opportunities)},
            on_event=on_event,
        )

        started = _begin_event(
            on_event,
            "quality_review",
            "quality_reviewer",
            "Reviewing retrieval coverage and evidence limitations.",
        )
        quality_failed = False
        try:
            warnings.extend(
                _quality_warnings(
                    papers=papers,
                    assumptions=assumptions,
                    direct_evidence_count=direct_evidence_count,
                    planner=self.planner,
                    planner_recovered=planner_recovered,
                )
            )
        except Exception:
            quality_failed = True
            warnings.append("Quality review failed; warnings may be incomplete.")
        warnings = _dedupe_warnings(warnings)
        if quality_failed:
            quality_status = "failed"
        elif warnings:
            quality_status = "degraded"
        else:
            quality_status = "complete"
        _emit_event(
            events,
            "quality_review",
            "quality_reviewer",
            started,
            quality_status,
            "Reviewed retrieval coverage, evidence limitations, and fallback signals.",
            {"warning_count": len(warnings)},
            on_event=on_event,
        )
        return AnalysisReport(
            query=query,
            claim=claim,
            papers=papers,
            claim_variants=variants,
            assumptions=assumptions,
            negative_evidence=negative_evidence,
            idea_opportunities=opportunities,
            events=events,
            warnings=warnings,
        )

    def extract_core_claim(self, query: str) -> str:
        extractor = getattr(self.planner, "extract_core_claim", None)
        if extractor:
            return extractor(query)
        return self.planner.build(query).claim


def _build_claim_variants(
    plan: list[PlannedClaimVariant], claim: str, papers: list[Paper]
) -> list[ClaimVariant]:
    return [
        ClaimVariant(
            text=item.text,
            rationale=item.rationale,
            evidence=_collect_evidence(
                item.evidence_query or f"{claim} {item.text}", papers, max_items=3
            ),
        )
        for item in plan
    ]


def _build_assumptions(
    plan: list[PlannedAssumption], papers: list[Paper]
) -> list[Assumption]:
    assumptions: list[Assumption] = []
    for item in plan:
        evidence = _collect_evidence(item.retrieval_queries, papers, max_items=5)
        support = sum(1 for item in evidence if item.stance == "support")
        contradict = sum(1 for item in evidence if item.stance == "contradict")
        limitations = sum(1 for item in evidence if item.stance == "limit")
        if support and (contradict or limitations):
            status = "mixed"
            risk = "high"
        elif support:
            status = "supported"
            risk = "medium"
        elif contradict:
            status = "unsupported"
            risk = "high"
        elif limitations:
            status = "mixed"
            risk = "high"
        else:
            status = "unknown"
            risk = "high"
        assumptions.append(
            Assumption(
                text=item.text,
                status=status,
                evidence=evidence,
                risk=risk,
                support_query=item.support_query,
                contradict_query=item.contradict_query,
                limitation_query=item.limitation_query,
                null_result_query=item.null_result_query,
            )
        )
    return assumptions


def _collect_evidence(
    evidence_queries: str | list[str], papers: list[Paper], max_items: int = 4
) -> list[EvidenceItem]:
    queries = [evidence_queries] if isinstance(evidence_queries, str) else evidence_queries
    query_kinds = (
        ["support", "contradict", "limitation", "null_result"]
        if len(queries) == 4
        else ["general"] * len(queries)
    )
    claim_terms = set()
    for query in queries:
        claim_terms.update(keywords(query, 12))
    candidates: list[EvidenceItem] = []
    for paper in papers:
        for sentence in split_sentences(paper.abstract):
            combined_text = f"{paper.title} {sentence}"
            query_matches = [
                (overlap_score(query, combined_text), query, query_kind)
                for query, query_kind in zip(queries, query_kinds)
            ]
            score, matched_query, query_kind = max(query_matches, key=lambda item: item[0])
            sentence_terms = set(keywords(combined_text, 20))
            shared_terms = claim_terms & sentence_terms
            stance = _sentence_stance(sentence)
            if stance == "mention":
                continue
            if score < 0.08 or len(shared_terms) < 2:
                cjk_stance_match = _contains_cjk(combined_text) and (
                    score >= 0.08 or stance == "contradict"
                )
                if not cjk_stance_match:
                    continue
            candidates.append(
                EvidenceItem(
                    paper_title=paper.title,
                    year=paper.year,
                    snippet=truncate(sentence),
                    stance=stance,
                    score=score,
                    paper_source=paper.source,
                    paper_url=paper.url,
                    paper_external_id=paper.external_id,
                    is_fixture=paper.is_fixture,
                    matched_query=matched_query,
                    query_kind=query_kind,
                    source_span_start=max(0, paper.abstract.find(sentence)),
                )
            )
    candidates.sort(key=lambda item: (item.score, item.stance == "support"), reverse=True)
    return candidates[:max_items]


def _mine_negative_evidence(papers: list[Paper]) -> list[NegativeEvidence]:
    findings: list[NegativeEvidence] = []
    for paper in papers:
        for sentence in split_sentences(paper.abstract):
            lower = sentence.lower()
            if not _contains_any(lower, NEGATIVE_MARKERS):
                continue
            if _contains_any(lower, NULL_RESULT_MARKERS):
                findings.append(
                    NegativeEvidence(
                        kind="negative_result",
                        text=truncate(sentence),
                        paper_title=paper.title,
                        year=paper.year,
                        implication=(
                            "Treat this as a candidate boundary condition or "
                            "null-result replication slot."
                        ),
                        paper_source=paper.source,
                        paper_url=paper.url,
                        paper_external_id=paper.external_id,
                        is_fixture=paper.is_fixture,
                        source_span_start=max(0, paper.abstract.find(sentence)),
                    )
                )
                continue
            if "no consistent" in lower or "no improvement" in lower or "无效" in lower:
                kind = "negative_result"
                implication = "Treat this as a candidate boundary condition or null-result replication slot."
            elif "fail" in lower or "mismatch" in lower or "brittle" in lower:
                kind = "failure_mode"
                implication = "Use this as a stress-test case before proposing a new method."
            else:
                kind = "limitation"
                implication = "Narrow the future idea so this limitation is explicitly controlled."
            findings.append(
                NegativeEvidence(
                    kind=kind,
                    text=truncate(sentence),
                    paper_title=paper.title,
                    year=paper.year,
                    implication=implication,
                    paper_source=paper.source,
                    paper_url=paper.url,
                    paper_external_id=paper.external_id,
                    is_fixture=paper.is_fixture,
                    source_span_start=max(0, paper.abstract.find(sentence)),
                )
            )
    return findings[:8]


def _targeted_search(
    retriever: PaperRetriever, queries: list[str], original_claim: str, limit: int
) -> tuple[list[Paper], list[str], bool]:
    seen: set[str] = set()
    papers: list[Paper] = []
    warnings: list[str] = []
    had_failure = False
    per_query_limit = max(5, min(limit, 10))
    for query in queries:
        try:
            results = retriever.search(query, limit=per_query_limit)
        except Exception:
            had_failure = True
            warnings.append(
                f"Retriever {_retriever_name(retriever)} failed; retrieval may be incomplete."
            )
            continue
        warnings.extend(_consume_retriever_warnings(retriever))
        for paper in results:
            key = _paper_identity_key(paper)
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append(paper)
    ranked = sorted(
        papers,
        key=lambda paper: _relevance_score(original_claim, paper),
        reverse=True,
    )
    relevant = [
        paper
        for paper in ranked
        if _relevance_score(original_claim, paper) >= 0.08
    ]
    return relevant[:limit], _dedupe_warnings(warnings), had_failure


def _build_idea_opportunities(
    assumptions: list[Assumption], negative_evidence: list[NegativeEvidence]
) -> list[IdeaOpportunity]:
    opportunities: list[IdeaOpportunity] = []
    for assumption in assumptions:
        if assumption.status in {"mixed", "unsupported", "unknown"}:
            opportunities.append(
                IdeaOpportunity(
                    title=f"Validate assumption: {assumption.text}",
                    kind="assumption_gap",
                    rationale=(
                        "This assumption is not cleanly supported by the retrieved literature, "
                        f"so it can become a focused pre-idea experiment."
                    ),
                    next_step=(
                        "Design a small controlled experiment with an explicit dataset, baseline, "
                        "and evaluation metric for this assumption."
                    ),
                    linked_evidence=[item.paper_title for item in assumption.evidence],
                    score=_assumption_opportunity_score(assumption),
                )
            )
    for finding in negative_evidence[:3]:
        opportunities.append(
            IdeaOpportunity(
                title=f"Turn {finding.kind.replace('_', ' ')} into a research slot",
                kind="negative_evidence",
                rationale=(
                    "Negative evidence often points to publishable boundary conditions, "
                    "robustness studies, or benchmark gaps."
                ),
                next_step=(
                    "Build an evaluation setting that reproduces this failure and tests a targeted repair."
                ),
                linked_evidence=[finding.paper_title],
                score=_negative_evidence_score(finding),
            )
        )
    if not opportunities:
        opportunities.append(
            IdeaOpportunity(
                title="Expand retrieval before ideation",
                kind="boundary_condition",
                rationale="The retrieved set does not expose enough assumptions or negative evidence.",
                next_step="Retrieve more papers, especially surveys and papers with limitation sections.",
                linked_evidence=[],
                score=55,
            )
        )
    opportunities.sort(key=lambda item: item.score, reverse=True)
    return opportunities[:8]


def _assumption_opportunity_score(assumption: Assumption) -> int:
    support = sum(1 for item in assumption.evidence if item.stance == "support")
    negative = sum(1 for item in assumption.evidence if item.stance in {"limit", "contradict"})
    if assumption.status == "mixed":
        base = 86
    elif assumption.status == "unknown":
        base = 76
    elif assumption.status == "unsupported":
        base = 64
    else:
        base = 48
    evidence_bonus = min(8, support + negative)
    risk_bonus = 4 if assumption.risk == "high" else 0
    return min(99, base + evidence_bonus + risk_bonus)


def _negative_evidence_score(finding: NegativeEvidence) -> int:
    scores = {
        "negative_result": 88,
        "failure_mode": 84,
        "limitation": 78,
    }
    return scores.get(finding.kind, 70)


def _sentence_stance(sentence: str) -> str:
    lower = sentence.lower()
    if _contains_any(lower, NEGATIVE_MARKERS):
        if _contains_any(lower, NULL_RESULT_MARKERS):
            return "contradict"
        return "limit"
    if _contains_any(lower, SUPPORT_MARKERS):
        return "support"
    return "mention"


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in markers)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _paper_identity_key(paper: Paper) -> str:
    if paper.external_id:
        return f"id:{paper.external_id.lower().strip()}"
    return f"title:{' '.join(paper.title.lower().split())}"


def _retriever_name(retriever: PaperRetriever) -> str:
    return retriever.__class__.__name__


def _consume_retriever_warnings(retriever: PaperRetriever) -> list[str]:
    warnings = getattr(retriever, "last_warnings", [])
    if not isinstance(warnings, list):
        return []
    return [warning for warning in warnings if isinstance(warning, str)]


def _quality_warnings(
    *,
    papers: list[Paper],
    assumptions: list[Assumption],
    direct_evidence_count: int,
    planner: ClaimPlanner,
    planner_recovered: bool = False,
) -> list[str]:
    warnings: list[str] = []
    if not papers:
        warnings.append(
            "Quality review: zero papers retrieved; all assumption statuses remain provisional."
        )
    if direct_evidence_count == 0:
        warnings.append(
            "Quality review: zero direct evidence cards were mapped from retrieved abstracts."
        )
    if assumptions and all(assumption.status == "unknown" for assumption in assumptions):
        warnings.append("Quality review: all assumptions remain unknown after retrieval.")
    if papers:
        warnings.append(
            "Quality review: abstract-only evidence; inspect full papers before relying on the report."
        )
        warnings.append(
            "Quality review: evidence statuses are heuristic abstract matches, not scientific adjudication."
        )
    if planner_recovered or _planner_used_heuristic_fallback(planner):
        warnings.append(
            "Quality review: heuristic fallback used; validate the claim plan with domain expertise."
        )
    return warnings


def _planner_used_heuristic_fallback(planner: ClaimPlanner) -> bool:
    return (
        getattr(planner, "used_planner", "") == "heuristic"
        and bool(getattr(planner, "fallback_reason", ""))
    )


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for warning in warnings:
        key = warning.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(warning)
    return deduped


def _emit_event(
    events: list[AgentEvent],
    stage: str,
    role: str,
    started: float,
    status: str,
    public_summary: str,
    artifacts: dict[str, object] | None = None,
    *,
    on_event: AgentEventCallback | None = None,
) -> None:
    event = AgentEvent(
        stage=stage,
        role=role,
        status=status,
        public_summary=public_summary,
        duration_ms=max(0, round((time.perf_counter() - started) * 1000)),
        artifacts=artifacts or {},
    )
    events.append(event)
    _notify_pipeline_event(on_event, event)


def _begin_event(
    on_event: AgentEventCallback | None,
    stage: str,
    role: str,
    public_summary: str,
) -> float:
    _notify_pipeline_event(
        on_event,
        AgentEvent(
            stage=stage,
            role=role,
            status="running",
            public_summary=public_summary,
            duration_ms=0,
            artifacts={},
        ),
    )
    return time.perf_counter()


def _notify_pipeline_event(
    on_event: AgentEventCallback | None,
    event: AgentEvent,
) -> None:
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        return


def _relevance_score(claim: str, paper: Paper) -> float:
    paper_text = f"{paper.title} {paper.abstract}"
    return max(
        overlap_score(claim, paper_text),
        overlap_score(_expand_research_aliases(claim), _expand_research_aliases(paper_text)),
    )


def _expand_research_aliases(text: str) -> str:
    expanded = text
    replacements = {
        "RAG": "RAG retrieval augmented generation",
        "rag": "rag retrieval augmented generation",
        "hallucination": "hallucination factuality unsupported claims",
        "hallucinations": "hallucinations factuality unsupported claims",
        "LLM-generated answers": "LLM-generated answers generation question answering",
        "llm-generated answers": "llm-generated answers generation question answering",
    }
    for source, target in replacements.items():
        expanded = expanded.replace(source, target)
    return expanded

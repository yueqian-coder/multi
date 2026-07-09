from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
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
    "限制",
    "失败",
    "无效",
    "噪声",
)


@dataclass
class ClaimScopePipeline:
    retriever: PaperRetriever
    planner: ClaimPlanner = field(default_factory=HeuristicClaimPlanner)

    @classmethod
    def from_env(cls, retriever: PaperRetriever) -> "ClaimScopePipeline":
        from .llm import OpenAICompatibleClient

        llm_client = OpenAICompatibleClient.from_env()
        if llm_client:
            planner: ClaimPlanner = LLMClaimPlanner(llm_client=llm_client)
        else:
            planner = HeuristicClaimPlanner()
        return cls(retriever=retriever, planner=planner)

    def analyze(self, query: str, limit: int = 20) -> AnalysisReport:
        plan = self.planner.build(query)
        claim = plan.claim
        retrieval_queries = [claim]
        for item in plan.assumptions:
            retrieval_queries.extend(item.retrieval_queries)
        papers = _targeted_search(self.retriever, retrieval_queries, claim, limit)
        variants = _build_claim_variants(plan.claim_variants, claim, papers)
        assumptions = _build_assumptions(plan.assumptions, papers)
        negative_evidence = _mine_negative_evidence(papers)
        opportunities = _build_idea_opportunities(assumptions, negative_evidence)
        return AnalysisReport(
            query=query,
            claim=claim,
            papers=papers,
            claim_variants=variants,
            assumptions=assumptions,
            negative_evidence=negative_evidence,
            idea_opportunities=opportunities,
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
        negative = sum(1 for item in evidence if item.stance in {"limit", "contradict"})
        if support and negative:
            status = "mixed"
            risk = "high"
        elif support:
            status = "supported"
            risk = "medium"
        elif negative:
            status = "unsupported"
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
    claim_terms = set()
    for query in queries:
        claim_terms.update(keywords(query, 12))
    candidates: list[EvidenceItem] = []
    for paper in papers:
        for sentence in split_sentences(paper.abstract):
            combined_text = f"{paper.title} {sentence}"
            score = max(overlap_score(query, combined_text) for query in queries)
            sentence_terms = set(keywords(combined_text, 20))
            shared_terms = claim_terms & sentence_terms
            if score < 0.08 or len(shared_terms) < 2:
                continue
            stance = _sentence_stance(sentence)
            if stance == "mention":
                continue
            candidates.append(
                EvidenceItem(
                    paper_title=paper.title,
                    year=paper.year,
                    snippet=truncate(sentence),
                    stance=stance,
                    score=score,
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
                )
            )
    return findings[:8]


def _targeted_search(
    retriever: PaperRetriever, queries: list[str], original_claim: str, limit: int
) -> list[Paper]:
    seen: set[str] = set()
    papers: list[Paper] = []
    per_query_limit = max(5, min(limit, 10))
    for query in queries:
        try:
            results = retriever.search(query, limit=per_query_limit)
        except Exception:
            continue
        for paper in results:
            key = paper.title.lower().strip()
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
    return relevant[:limit]


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
        if "no consistent" in lower or "no improvement" in lower:
            return "contradict"
        return "limit"
    if _contains_any(lower, SUPPORT_MARKERS):
        return "support"
    return "mention"


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in markers)


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

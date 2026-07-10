from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re


@dataclass(frozen=True)
class Paper:
    title: str
    year: int | None
    authors: list[str]
    abstract: str
    source: str
    url: str = ""
    external_id: str = ""
    is_fixture: bool = False

    def __post_init__(self) -> None:
        if not self.external_id:
            object.__setattr__(
                self,
                "external_id",
                _derive_external_id(self.source, self.url),
            )
        if not self.is_fixture and self.source.lower() in {"demo", "fixture"}:
            object.__setattr__(self, "is_fixture", True)

    @property
    def citation(self) -> str:
        year = self.year if self.year is not None else "n.d."
        authors = ", ".join(self.authors[:3]) if self.authors else "Unknown authors"
        return f"{self.title} ({year}) - {authors}"


@dataclass(frozen=True)
class EvidenceItem:
    paper_title: str
    year: int | None
    snippet: str
    stance: str
    score: float = 0.0
    method: str = "abstract_heuristic"


@dataclass(frozen=True)
class ClaimVariant:
    text: str
    rationale: str
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass(frozen=True)
class Assumption:
    text: str
    status: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    risk: str = "medium"
    support_query: str = ""
    contradict_query: str = ""
    limitation_query: str = ""
    null_result_query: str = ""


@dataclass(frozen=True)
class NegativeEvidence:
    kind: str
    text: str
    paper_title: str
    year: int | None
    implication: str


@dataclass(frozen=True)
class IdeaOpportunity:
    title: str
    kind: str
    rationale: str
    next_step: str
    linked_evidence: list[str] = field(default_factory=list)
    score: int = 0


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    description: str
    output: str
    artifact_count: int
    status: str = "complete"


@dataclass(frozen=True)
class ClaimCandidate:
    claim: str
    method_or_mechanism: str = ""
    target_or_task: str = ""
    expected_effect: str = ""
    conditions: list[str] = field(default_factory=list)
    falsification_test: str = (
        "Compare the stated outcome against a baseline under the stated condition."
    )
    missing_information: list[str] = field(default_factory=list)
    confidence: float = 0.0
    proposer: str = "heuristic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _clamp_confidence(self.confidence))

    def score(self) -> float:
        covered = sum(
            bool(value)
            for value in (
                self.method_or_mechanism,
                self.target_or_task,
                self.expected_effect,
                self.falsification_test,
            )
        )
        return round(covered / 4 * 100 - len(self.missing_information) * 5, 1)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimCritique:
    candidate_id: str = ""
    critic: str = ""
    rubric_scores: dict[str, float] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    revision: str = ""
    public_summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AgentEvent:
    stage: str
    role: str
    status: str
    public_summary: str
    duration_ms: int = 0
    artifacts: dict[str, object] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoreClaimResult:
    direction: str
    selected_claim: str = ""
    selected_candidate: ClaimCandidate | None = None
    candidates: list[ClaimCandidate] = field(default_factory=list)
    critiques: list[ClaimCritique] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    unresolved_ambiguities: list[str] = field(default_factory=list)
    mode: str = "heuristic"
    degraded: bool = False

    def __post_init__(self) -> None:
        if self.selected_claim:
            return
        if self.selected_candidate is not None:
            object.__setattr__(self, "selected_claim", self.selected_candidate.claim)
            return
        if self.candidates:
            object.__setattr__(self, "selected_claim", self.candidates[0].claim)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisReport:
    query: str
    claim: str
    papers: list[Paper]
    claim_variants: list[ClaimVariant]
    assumptions: list[Assumption]
    negative_evidence: list[NegativeEvidence]
    idea_opportunities: list[IdeaOpportunity]
    events: list[AgentEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def workflow_steps(self) -> list[WorkflowStep]:
        evidence_query_count = sum(
            1
            for assumption in self.assumptions
            for query in [
                assumption.support_query,
                assumption.contradict_query,
                assumption.limitation_query,
                assumption.null_result_query,
            ]
            if query
        )
        evidence_card_count = sum(
            len(assumption.evidence) for assumption in self.assumptions
        ) + len(self.negative_evidence)
        return [
            WorkflowStep(
                name="Research Direction",
                description="User-provided fuzzy idea or research direction.",
                output=self.query,
                artifact_count=1,
            ),
            WorkflowStep(
                name="Core Claim",
                description="Normalized testable claim used as the retrieval anchor.",
                output=self.claim,
                artifact_count=1,
            ),
            WorkflowStep(
                name="Claim Variants / Boundary Conditions",
                description="Alternative claim formulations, boundaries, and failure variants.",
                output=f"{len(self.claim_variants)} variants generated",
                artifact_count=len(self.claim_variants),
            ),
            WorkflowStep(
                name="Hidden Assumptions",
                description="Implicit assumptions that must hold for the direction to work.",
                output=f"{len(self.assumptions)} assumptions identified",
                artifact_count=len(self.assumptions),
            ),
            WorkflowStep(
                name="Evidence Queries",
                description="Adversarial support, contradiction, limitation, and null-result searches.",
                output=f"{evidence_query_count} targeted queries generated",
                artifact_count=evidence_query_count,
            ),
            WorkflowStep(
                name="Evidence Cards",
                description="Traceable paper snippets mapped to assumptions and limitations.",
                output=f"{evidence_card_count} evidence cards extracted",
                artifact_count=evidence_card_count,
            ),
            WorkflowStep(
                name="Idea Opportunities",
                description="Ranked opportunity slots synthesized from weak assumptions and failures.",
                output=f"{len(self.idea_opportunities)} opportunities ranked",
                artifact_count=len(self.idea_opportunities),
            ),
        ]

    def assumption_matrix(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for assumption in self.assumptions:
            counts = _evidence_counts(assumption.evidence)
            rows.append(
                {
                    "Assumption": assumption.text,
                    "Status": assumption.status,
                    "Risk": assumption.risk,
                    "Support": counts["support"],
                    "Contradict": counts["contradict"],
                    "Limitation": counts["limitation"],
                    "Null Result": counts["null_result"],
                    "Opportunity Signal": _opportunity_signal(assumption.status, counts),
                }
            )
        return rows

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# ClaimScope Report",
            "",
            f"**Input direction / claim:** {self.query}",
            f"**Normalized claim:** {self.claim}",
            f"**Generated:** {self.generated_at.isoformat(timespec='seconds')}",
            "",
            "## Workflow Trace",
        ]
        for idx, step in enumerate(self.workflow_steps(), 1):
            lines.append(f"{idx}. **{step.name}** - `{step.status}`")
            lines.append(f"   - {step.description}")
            lines.append(f"   - Output: {step.output}")

        lines.extend(["", "## Retrieved Papers"])
        if self.papers:
            for idx, paper in enumerate(self.papers, 1):
                fixture_label = " [synthetic fixture]" if paper.is_fixture else ""
                lines.append(f"{idx}. {paper.citation}{fixture_label}")
                if paper.external_id:
                    lines.append(f"   - External ID: {paper.external_id}")
                if paper.url:
                    lines.append(f"   - URL: {paper.url}")
        else:
            lines.append("No papers were retrieved.")

        if self.warnings:
            lines.extend(["", "## Quality Warnings"])
            for warning in self.warnings:
                lines.append(f"- {warning}")

        lines.extend(["", "## Assumption Evidence Matrix"])
        if self.assumptions:
            lines.append(
                "| Assumption | Status | Support | Contradict | Limitation | Null Result | Opportunity Signal |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---|")
            for row in self.assumption_matrix():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row["Assumption"]).replace("|", "/"),
                            str(row["Status"]),
                            str(row["Support"]),
                            str(row["Contradict"]),
                            str(row["Limitation"]),
                            str(row["Null Result"]),
                            str(row["Opportunity Signal"]),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("No assumptions were generated.")

        lines.extend(["", "## Claim Variants"])
        for variant in self.claim_variants:
            lines.append(f"- **{variant.text}**")
            lines.append(f"  - Why it matters: {variant.rationale}")

        lines.extend(["", "## Assumption Gaps"])
        for assumption in self.assumptions:
            lines.append(
                f"- **{assumption.text}** - status: `{assumption.status}`, risk: `{assumption.risk}`"
            )
            if any(
                [
                    assumption.support_query,
                    assumption.contradict_query,
                    assumption.limitation_query,
                    assumption.null_result_query,
                ]
            ):
                lines.append("  - Evidence queries:")
                if assumption.support_query:
                    lines.append(f"    - Support: `{assumption.support_query}`")
                if assumption.contradict_query:
                    lines.append(f"    - Contradict: `{assumption.contradict_query}`")
                if assumption.limitation_query:
                    lines.append(f"    - Limitation: `{assumption.limitation_query}`")
                if assumption.null_result_query:
                    lines.append(f"    - Null result: `{assumption.null_result_query}`")
            if assumption.evidence:
                for evidence in assumption.evidence:
                    lines.append(
                        f"  - [{evidence.stance}] {evidence.paper_title} ({evidence.year}): {evidence.snippet}"
                    )
            else:
                lines.append("  - No direct evidence found in the retrieved set.")

        lines.extend(["", "## Negative Evidence"])
        if self.negative_evidence:
            for item in self.negative_evidence:
                lines.append(
                    f"- **{item.kind}** from {item.paper_title} ({item.year}): {item.text}"
                )
                lines.append(f"  - Implication: {item.implication}")
        else:
            lines.append("No explicit negative evidence was found.")

        lines.extend(["", "## Idea Opportunities"])
        if self.idea_opportunities:
            for opportunity in self.idea_opportunities:
                lines.append(
                    f"- **{opportunity.title}** (`{opportunity.kind}`, score: `{opportunity.score}`)"
                )
                lines.append(f"  - Rationale: {opportunity.rationale}")
                lines.append(f"  - Next step: {opportunity.next_step}")
                if opportunity.linked_evidence:
                    lines.append(
                        "  - Linked evidence: "
                        + "; ".join(opportunity.linked_evidence)
                    )
        else:
            lines.append("No opportunity slots were generated.")

        return "\n".join(lines)


def _evidence_counts(evidence: list[EvidenceItem]) -> dict[str, int]:
    counts = {
        "support": 0,
        "contradict": 0,
        "limitation": 0,
        "null_result": 0,
    }
    for item in evidence:
        if item.stance == "support":
            counts["support"] += 1
        elif item.stance == "contradict":
            counts["contradict"] += 1
        elif item.stance == "limit":
            counts["limitation"] += 1
        lower = item.snippet.lower()
        if any(
            marker in lower
            for marker in [
                "no consistent",
                "no improvement",
                "null result",
                "no significant",
                "does not improve",
                "marginal gain",
                "not significant",
                "non-significant",
                "negative result",
                "没有显著提升",
                "无显著提升",
                "无效",
            ]
        ):
            counts["null_result"] += 1
    return counts


def _opportunity_signal(status: str, counts: dict[str, int]) -> str:
    if counts["support"] and (counts["contradict"] or counts["limitation"]):
        return "high: contested boundary"
    if status == "unknown":
        return "medium: untested assumption"
    if counts["null_result"]:
        return "high: null-result gap"
    if counts["limitation"]:
        return "medium: boundary condition"
    if status == "supported":
        return "low: already supported"
    return "medium: needs targeted search"


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _derive_external_id(source: str, url: str) -> str:
    arxiv_id = _extract_arxiv_id(url)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if "semantic" in source.lower():
        semantic_id = _extract_semantic_scholar_id(url)
        if semantic_id:
            return f"semantic_scholar:{semantic_id}"
    if url:
        return f"url:{url.lower().strip()}"
    return ""


def _extract_arxiv_id(url: str) -> str:
    if not url:
        return ""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", url, flags=re.IGNORECASE)
    if not match:
        return ""
    raw_id = match.group(1).removesuffix(".pdf")
    return re.sub(r"v\d+$", "", raw_id)


def _extract_semantic_scholar_id(url: str) -> str:
    if not url:
        return ""
    match = re.search(r"semanticscholar\.org/paper/(?:[^/]+/)?([A-Za-z0-9]+)", url)
    return match.group(1) if match else ""

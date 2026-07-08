from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Paper:
    title: str
    year: int | None
    authors: list[str]
    abstract: str
    source: str
    url: str = ""

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


@dataclass(frozen=True)
class AnalysisReport:
    query: str
    claim: str
    papers: list[Paper]
    claim_variants: list[ClaimVariant]
    assumptions: list[Assumption]
    negative_evidence: list[NegativeEvidence]
    idea_opportunities: list[IdeaOpportunity]
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# ClaimScope Report",
            "",
            f"**Input direction / claim:** {self.query}",
            f"**Normalized claim:** {self.claim}",
            f"**Generated:** {self.generated_at.isoformat(timespec='seconds')}",
            "",
            "## Retrieved Papers",
        ]
        if self.papers:
            for idx, paper in enumerate(self.papers, 1):
                lines.append(f"{idx}. {paper.citation}")
                if paper.url:
                    lines.append(f"   - URL: {paper.url}")
        else:
            lines.append("No papers were retrieved.")

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
                lines.append(f"- **{opportunity.title}** (`{opportunity.kind}`)")
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

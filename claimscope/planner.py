from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .text_utils import keywords


@dataclass(frozen=True)
class PlannedClaimVariant:
    text: str
    rationale: str
    evidence_query: str = ""


@dataclass(frozen=True)
class PlannedAssumption:
    text: str
    support_query: str
    contradict_query: str
    limitation_query: str
    null_result_query: str

    @property
    def retrieval_queries(self) -> list[str]:
        return [
            self.support_query,
            self.contradict_query,
            self.limitation_query,
            self.null_result_query,
        ]


@dataclass(frozen=True)
class ClaimPlan:
    claim: str
    claim_variants: list[PlannedClaimVariant] = field(default_factory=list)
    assumptions: list[PlannedAssumption] = field(default_factory=list)


class ClaimPlanner(Protocol):
    def build(self, query: str) -> ClaimPlan:
        ...


@dataclass
class HeuristicClaimPlanner:
    def build(self, query: str) -> ClaimPlan:
        claim = normalize_claim(query)
        topic = _readable_topic(claim)
        return ClaimPlan(
            claim=claim,
            claim_variants=[
                PlannedClaimVariant(
                    text=claim,
                    rationale=(
                        "Original user claim; serves as the anchor for lineage tracking."
                    ),
                    evidence_query=claim,
                ),
                PlannedClaimVariant(
                    text=(
                        "Boundary condition: the claim holds only under specific task, "
                        "dataset, and retrieval-quality conditions"
                    ),
                    rationale="Boundary-condition variant; useful for avoiding overclaiming.",
                    evidence_query="condition quality dataset task",
                ),
                PlannedClaimVariant(
                    text=(
                        "Failure variant: the claim can break when evidence is noisy, "
                        "mismatched, or under-specified"
                    ),
                    rationale="Negative-evidence variant; surfaces limitations before ideation.",
                    evidence_query="fail noisy mismatch limitation",
                ),
            ],
            assumptions=[
                PlannedAssumption(
                    text=f"The claimed improvement is real for the target setting: {topic}.",
                    support_query=f"{topic} improvement performance evidence",
                    contradict_query=f"{topic} no improvement worse contradict",
                    limitation_query=f"{topic} limitation boundary condition failure",
                    null_result_query=f"{topic} null result no consistent improvement",
                ),
                PlannedAssumption(
                    text="The required evidence or context is available and high-quality enough.",
                    support_query=f"{topic} relevant evidence context quality",
                    contradict_query=f"{topic} irrelevant evidence noisy retrieval mismatch",
                    limitation_query=f"{topic} evidence quality limitation noise",
                    null_result_query=f"{topic} retrieval noise no improvement",
                ),
                PlannedAssumption(
                    text="The evaluation metric faithfully measures the claimed improvement.",
                    support_query=f"{topic} metric evaluation benchmark validity",
                    contradict_query=f"{topic} metric mismatch unreliable evaluation",
                    limitation_query=f"{topic} brittle metrics citation mismatch limitation",
                    null_result_query=f"{topic} evaluation no consistent improvement",
                ),
                PlannedAssumption(
                    text=(
                        "The effect generalizes beyond the narrow datasets and domains used "
                        "in prior work."
                    ),
                    support_query=f"{topic} generalization multiple datasets domains",
                    contradict_query=f"{topic} domain shift fails robustness",
                    limitation_query=f"{topic} narrow dataset domain limitation",
                    null_result_query=f"{topic} domain shift no improvement",
                ),
            ],
        )


@dataclass
class LLMClaimPlanner:
    llm_client: object
    fallback: ClaimPlanner = field(default_factory=HeuristicClaimPlanner)
    max_variants: int = 6
    max_assumptions: int = 7
    used_planner: str = field(default="llm", init=False)
    fallback_reason: str = field(default="", init=False)

    def build(self, query: str) -> ClaimPlan:
        fallback_plan = self.fallback.build(query)
        try:
            content = self.llm_client.chat(_planner_messages(query), temperature=0.1)
            payload = _extract_json_object(content)
            plan = _payload_to_plan(
                payload=payload,
                fallback_plan=fallback_plan,
                max_variants=self.max_variants,
                max_assumptions=self.max_assumptions,
            )
            self.used_planner = "llm"
            self.fallback_reason = ""
            return plan
        except PlannerShapeError as exc:
            self.used_planner = "heuristic"
            self.fallback_reason = f"Invalid LLM plan shape: {exc}"
            return fallback_plan
        except ValueError as exc:
            self.used_planner = "heuristic"
            self.fallback_reason = f"Invalid LLM planner response: {exc}"
            return fallback_plan
        except Exception:
            self.used_planner = "heuristic"
            self.fallback_reason = "LLM planner failed; using heuristic planner."
            return fallback_plan


def normalize_claim(query: str) -> str:
    return query.strip().rstrip(".?。")


class PlannerShapeError(ValueError):
    pass


def _planner_messages(query: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are ClaimScope's research-planning module. Convert a fuzzy "
                "research direction into a literature-search plan before ideation. "
                "Return JSON only. Do not include markdown, citations, or prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "Input direction:\n"
                f"{query}\n\n"
                "Return this exact JSON shape:\n"
                "{\n"
                '  "normalized_claim": "specific, testable research claim",\n'
                '  "claim_variants": [\n'
                '    {"text": "historical or boundary variant", "rationale": "why this variant matters"}\n'
                "  ],\n"
                '  "assumptions": [\n'
                "    {\n"
                '      "text": "implicit assumption that must hold",\n'
                '      "support_query": "paper search query for supporting evidence",\n'
                '      "contradict_query": "paper search query for contradictory evidence",\n'
                '      "limitation_query": "paper search query for limitations or boundary conditions",\n'
                '      "null_result_query": "paper search query for null or negative results"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Make 3-6 variants and 4-7 assumptions. Queries should be short, "
                "domain-specific, and optimized for arXiv or Semantic Scholar search."
            ),
        },
    ]


def _extract_json_object(content: str) -> dict:
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM planner response did not contain a JSON object.")
    payload = json.loads(content[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM planner response must be a JSON object.")
    return payload


def _payload_to_plan(
    payload: dict,
    fallback_plan: ClaimPlan,
    max_variants: int,
    max_assumptions: int,
) -> ClaimPlan:
    claim = _coerce_text(payload.get("normalized_claim")) or fallback_plan.claim
    claim = normalize_claim(claim)

    variants = _parse_variants(payload.get("claim_variants"), claim, max_variants)
    assumptions = _parse_assumptions(payload.get("assumptions"), max_assumptions)

    if len(variants) < 3:
        raise PlannerShapeError("expected at least 3 claim variants")
    if len(assumptions) < 4:
        raise PlannerShapeError("expected at least 4 assumptions")

    return ClaimPlan(claim=claim, claim_variants=variants, assumptions=assumptions)


def _parse_variants(raw_variants: object, claim: str, max_variants: int) -> list[PlannedClaimVariant]:
    variants: list[PlannedClaimVariant] = [
        PlannedClaimVariant(
            text=claim,
            rationale="Normalized claim; serves as the anchor for lineage tracking.",
        )
    ]
    seen = {claim.lower()}
    if not isinstance(raw_variants, list):
        return variants
    for item in raw_variants:
        if not isinstance(item, dict):
            continue
        text = _coerce_text(item.get("text"))
        if not text or text.lower() in seen:
            continue
        rationale = _coerce_text(item.get("rationale")) or (
            "LLM-generated claim variant for targeted literature search."
        )
        variants.append(PlannedClaimVariant(text=text, rationale=rationale))
        seen.add(text.lower())
        if len(variants) >= max_variants:
            break
    return variants


def _parse_assumptions(raw_assumptions: object, max_assumptions: int) -> list[PlannedAssumption]:
    assumptions: list[PlannedAssumption] = []
    if not isinstance(raw_assumptions, list):
        return assumptions
    for item in raw_assumptions:
        if not isinstance(item, dict):
            continue
        text = _coerce_text(item.get("text"))
        if not text:
            continue
        required_queries = [
            _coerce_text(item.get("support_query")),
            _coerce_text(item.get("contradict_query")),
            _coerce_text(item.get("limitation_query")),
            _coerce_text(item.get("null_result_query")),
        ]
        if not all(required_queries):
            continue
        assumptions.append(
            PlannedAssumption(
                text=text,
                support_query=required_queries[0],
                contradict_query=required_queries[1],
                limitation_query=required_queries[2],
                null_result_query=required_queries[3],
            )
        )
        if len(assumptions) >= max_assumptions:
            break
    return assumptions


def _coerce_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _readable_topic(claim: str) -> str:
    compact = claim.strip().rstrip(".?")
    if len(compact) <= 110:
        return compact
    claim_terms = keywords(compact, 6)
    return " ".join(claim_terms[:4]) if claim_terms else compact[:110]

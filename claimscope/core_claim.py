from __future__ import annotations

import re
from dataclasses import replace
from time import perf_counter

from .models import AgentEvent, ClaimCandidate, CoreClaimResult

_EFFECT_VERBS = (
    "improve",
    "improves",
    "improved",
    "reduce",
    "reduces",
    "reduced",
    "increase",
    "increases",
    "increased",
    "decrease",
    "decreases",
    "decreased",
    "boost",
    "boosts",
    "boosted",
    "help",
    "helps",
    "helped",
)
_CONDITION_MARKERS = ("with", "under", "when", "if", "across", "during")


class HeuristicCoreClaimEngine:
    def run(self, direction: str) -> CoreClaimResult:
        from .planner import normalize_claim

        started = perf_counter()
        claim = normalize_claim(direction)
        candidate = _build_candidate(claim)
        event = AgentEvent(
            stage="core_claim",
            role="heuristic_core_claim_engine",
            status="complete",
            public_summary=(
                "Built a deterministic baseline claim candidate without private reasoning."
            ),
            duration_ms=max(0, round((perf_counter() - started) * 1000)),
            artifacts={
                "candidate_count": 1,
                "selected_claim": candidate.claim,
            },
            scores={"selected_candidate": candidate.score()},
        )
        return CoreClaimResult(
            direction=direction,
            selected_candidate=candidate,
            candidates=[candidate],
            events=[event],
            unresolved_ambiguities=list(candidate.missing_information),
            mode="heuristic",
            degraded=False,
        )


def with_selected_claim(result: CoreClaimResult, claim: str, *, mode: str) -> CoreClaimResult:
    refreshed = HeuristicCoreClaimEngine().run(claim)
    selected_candidate = refreshed.selected_candidate
    if selected_candidate is not None:
        selected_candidate = replace(selected_candidate, proposer=mode)
    return replace(
        refreshed,
        direction=result.direction,
        selected_claim=claim,
        selected_candidate=selected_candidate,
        candidates=[selected_candidate] if selected_candidate is not None else [],
        mode=mode,
    )


def _build_candidate(claim: str) -> ClaimCandidate:
    method_or_mechanism = _extract_method_or_mechanism(claim)
    target_or_task = _extract_target_or_task(claim)
    expected_effect = _extract_expected_effect(claim)
    conditions = _extract_conditions(claim)
    missing_information: list[str] = []
    if not method_or_mechanism:
        missing_information.append("method or mechanism")
    if not target_or_task:
        missing_information.append("target task or object")
    if not expected_effect:
        missing_information.append("measurable expected effect")
    confidence = max(0.0, min(1.0, _candidate_confidence(missing_information)))
    return ClaimCandidate(
        claim=claim,
        method_or_mechanism=method_or_mechanism,
        target_or_task=target_or_task,
        expected_effect=expected_effect,
        conditions=conditions,
        missing_information=missing_information,
        confidence=confidence,
        proposer="heuristic",
    )


def _candidate_confidence(missing_information: list[str]) -> float:
    covered_slots = 3 - len(missing_information)
    return round(max(0, covered_slots) / 3, 2)


def _extract_method_or_mechanism(claim: str) -> str:
    cleaned = claim.strip()
    match = re.match(r"(.+?)\b(for|on|in)\b\s+(.+)", cleaned, re.IGNORECASE)
    if match and _looks_like_method(match.group(1)):
        return match.group(1).strip()
    effect_match = _effect_match(cleaned)
    if effect_match:
        return effect_match.group(1).strip()
    return cleaned if _looks_like_method(cleaned) else ""


def _extract_target_or_task(claim: str) -> str:
    cleaned = claim.strip()
    match = re.match(r"(.+?)\b(for|on|in)\b\s+(.+)", cleaned, re.IGNORECASE)
    if match and _looks_like_method(match.group(1)):
        return _strip_conditions(match.group(3))
    effect_match = _effect_match(cleaned)
    if effect_match:
        return _strip_conditions(effect_match.group(3))
    return ""


def _extract_expected_effect(claim: str) -> str:
    effect_match = _effect_match(claim)
    if not effect_match:
        return ""
    verb = effect_match.group(2).strip()
    target = _strip_conditions(effect_match.group(3))
    return " ".join(part for part in (verb, target) if part).strip()


def _extract_conditions(claim: str) -> list[str]:
    conditions: list[str] = []
    for marker in _CONDITION_MARKERS:
        match = _condition_marker_match(claim, marker)
        if match is None:
            continue
        prefix = claim[: match.start()]
        suffix = claim[match.end() :]
        if _looks_like_method(prefix):
            conditions.append(suffix.strip())
    return conditions


def _effect_match(claim: str) -> re.Match[str] | None:
    verbs = "|".join(_EFFECT_VERBS)
    pattern = rf"(.+?)\b({verbs})\b\s+(.+)"
    return re.match(pattern, claim.strip(), re.IGNORECASE)


def _strip_conditions(text: str) -> str:
    stripped = text.strip()
    for marker in _CONDITION_MARKERS:
        match = _condition_marker_match(stripped, marker)
        if match is None:
            continue
        return stripped[: match.start()].strip()
    return stripped


def _condition_marker_match(text: str, marker: str) -> re.Match[str] | None:
    return re.search(rf"\s{re.escape(marker)}\s", text, re.IGNORECASE)


def _looks_like_method(text: str) -> bool:
    lowered = text.strip().lower()
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "use ",
            "using ",
            "apply ",
            "applying ",
            "add ",
            "adding ",
        )
    ) or bool(_effect_match(text))

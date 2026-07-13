from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from json import JSONDecodeError
from pathlib import Path
import re
from statistics import mean

from .models import CoreClaimResult


VALID_DOMAINS = {
    "ai",
    "medical_ai",
    "biology",
    "social_science",
    "climate",
    "materials",
    "hci",
    "education",
    "systems",
}
COMPONENT_WEIGHTS = {
    "slot_coverage": 0.12,
    "declarative_form": 0.12,
    "comparison_language": 0.16,
    "measurable_outcomes": 0.18,
    "falsifiability": 0.22,
    "specificity": 0.20,
}
COMPARISON_MARKERS = (
    "versus",
    "compared with",
    "compared to",
    "relative to",
    "baseline",
    "control",
    "without",
    "against",
    "before and after",
    "ablation",
    "randomized",
    "held-out",
    "than",
)
MEASUREMENT_MARKERS = (
    "accuracy",
    "auc",
    "f1",
    "precision",
    "recall",
    "rate",
    "rates",
    "score",
    "scores",
    "error",
    "errors",
    "false negative",
    "false negatives",
    "false positive",
    "false positives",
    "mortality",
    "latency",
    "throughput",
    "yield",
    "cost",
    "emissions",
    "retention",
    "completion",
    "variance",
    "temperature",
    "strength",
    "toughness",
    "energy",
    "power use",
    "water use",
    "electricity use",
    "attendance",
    "recommendations",
    "abstention",
    "validation",
    "calibration",
    "unsupported factual answers",
    "unsupported answers",
    "factuality",
    "time",
    "percent",
    "%",
)
BOUNDARY_MARKERS = (
    "on ",
    "under ",
    "when ",
    "with ",
    "across ",
    "during ",
    "held-out",
    "in ",
    "on ",
    "after ",
    "while ",
    "for ",
)
OVERCLAIM_MARKERS = (
    "guarantees",
    "proves",
    "eliminates all",
    "solves",
    "cures",
    "perfect",
    "universal",
    "universally",
    "never fails",
    "no risk",
    "all clinician errors",
    "all errors",
)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    direction: str
    domain: str
    required_concepts: list[str]
    forbidden_overclaims: list[str]
    expected_slots: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimScore:
    total: float
    slot_coverage: float
    declarative_form: float
    comparison_language: float
    measurable_outcomes: float
    falsifiability: float
    specificity: float
    overclaim_penalty: float
    components: dict[str, float]
    required_concept_coverage: float = 1.0
    required_concept_penalty: float = 0.0
    forbidden_overclaim_penalty: float = 0.0
    cue_stuffing_penalty: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str
    domain: str
    direction: str
    claim: str
    score: ClaimScore
    required_concepts_found: list[str]
    required_concepts_missing: list[str]
    forbidden_overclaims_found: list[str]
    mode: str
    slot_grounding: dict[str, float]
    target_comparator_contaminated: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["score"] = self.score.to_dict()
        return payload


@dataclass(frozen=True)
class BenchmarkReport:
    summary: dict[str, object]
    case_results: list[BenchmarkCaseResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": dict(self.summary),
            "case_results": [item.to_dict() for item in self.case_results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    case_path = Path(path)
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    with case_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON in benchmark case"
                ) from exc
            case = _payload_to_case(payload, line_number=line_number)
            if case.id in seen_ids:
                raise ValueError(f"duplicate benchmark id: {case.id}")
            seen_ids.add(case.id)
            cases.append(case)
    return cases


def score_claim(
    direction: str, result: CoreClaimResult, *, case: BenchmarkCase | None = None
) -> ClaimScore:
    claim = result.selected_claim.strip()
    lower_claim = claim.lower()
    required_coverage = _required_concept_coverage(claim, case)
    forbidden_overclaim_penalty = _case_forbidden_overclaim_penalty(claim, case)
    components = {
        "slot_coverage": _slot_coverage(result),
        "declarative_form": _declarative_form(claim),
        "comparison_language": _marker_score(lower_claim, COMPARISON_MARKERS),
        "measurable_outcomes": _marker_score(lower_claim, MEASUREMENT_MARKERS),
        "falsifiability": _falsifiability(claim, result),
        "specificity": _specificity(direction, claim),
        "required_concept_coverage": required_coverage,
    }
    overclaim_penalty = _overclaim_penalty(lower_claim)
    required_concept_penalty = (1.0 - required_coverage) * 20.0
    cue_stuffing_penalty = _cue_stuffing_penalty(lower_claim)
    weighted = sum(
        components[name] * weight for name, weight in COMPONENT_WEIGHTS.items()
    )
    total = max(
        0.0,
        min(
            100.0,
            weighted * 100
            - overclaim_penalty
            - required_concept_penalty
            - forbidden_overclaim_penalty
            - cue_stuffing_penalty,
        ),
    )
    rounded_components = {
        name: round(value, 3) for name, value in components.items()
    }
    rounded_components["overclaim_penalty"] = round(overclaim_penalty, 2)
    rounded_components["required_concept_penalty"] = round(
        required_concept_penalty, 2
    )
    rounded_components["forbidden_overclaim_penalty"] = round(
        forbidden_overclaim_penalty, 2
    )
    rounded_components["cue_stuffing_penalty"] = round(cue_stuffing_penalty, 2)
    return ClaimScore(
        total=round(total, 2),
        slot_coverage=rounded_components["slot_coverage"],
        declarative_form=rounded_components["declarative_form"],
        comparison_language=rounded_components["comparison_language"],
        measurable_outcomes=rounded_components["measurable_outcomes"],
        falsifiability=rounded_components["falsifiability"],
        specificity=rounded_components["specificity"],
        overclaim_penalty=round(overclaim_penalty, 2),
        required_concept_coverage=rounded_components["required_concept_coverage"],
        required_concept_penalty=round(required_concept_penalty, 2),
        forbidden_overclaim_penalty=round(forbidden_overclaim_penalty, 2),
        cue_stuffing_penalty=round(cue_stuffing_penalty, 2),
        components=rounded_components,
    )


def run_benchmark(cases: list[BenchmarkCase], engine: object) -> BenchmarkReport:
    case_results: list[BenchmarkCaseResult] = []
    for case in cases:
        result = engine.run(case.direction)
        score = score_claim(case.direction, result, case=case)
        claim = result.selected_claim
        found, missing = _concept_matches(claim, case.required_concepts)
        forbidden = _found_terms(claim, case.forbidden_overclaims)
        slot_grounding = _slot_grounding(result, case)
        target_comparator_contaminated = bool(
            result.selected_candidate
            and re.search(
                _comparison_regex(),
                result.selected_candidate.target_or_task,
                re.IGNORECASE,
            )
        )
        case_results.append(
            BenchmarkCaseResult(
                case_id=case.id,
                domain=case.domain,
                direction=case.direction,
                claim=claim,
                score=score,
                required_concepts_found=found,
                required_concepts_missing=missing,
                forbidden_overclaims_found=forbidden,
                mode=getattr(result, "mode", ""),
                slot_grounding=slot_grounding,
                target_comparator_contaminated=target_comparator_contaminated,
            )
        )
    scores = [item.score.total for item in case_results]
    pass_threshold = 70.0
    passed = sum(score >= pass_threshold for score in scores)
    component_names = [*COMPONENT_WEIGHTS, "required_concept_coverage"]
    component_means = {
        name: round(
            mean(item.score.components.get(name, 0.0) for item in case_results),
            3,
        )
        if case_results
        else 0.0
        for name in component_names
    }
    domain_scores = {}
    for domain in sorted({item.domain for item in case_results}):
        domain_values = [
            item.score.total for item in case_results if item.domain == domain
        ]
        domain_scores[domain] = {
            "case_count": len(domain_values),
            "mean": round(mean(domain_values), 2),
            "min": round(min(domain_values), 2),
            "max": round(max(domain_values), 2),
        }
    lowest_cases = sorted(case_results, key=lambda item: item.score.total)[:5]
    exact_copies = sum(
        _normalized_copy(item.claim) == _normalized_copy(item.direction)
        for item in case_results
    )
    input_concept_leakage = [
        _required_concept_coverage(case.direction, case) for case in cases
    ]
    grounded_results = [item for item in case_results if item.slot_grounding]
    slot_names = sorted(
        {
            name
            for item in grounded_results
            for name in item.slot_grounding
            if name != "overall"
        }
    )
    annotated_slot_scores = {
        name: round(mean(item.slot_grounding[name] for item in grounded_results if name in item.slot_grounding), 3)
        for name in slot_names
    }
    target_comparator_contamination = sum(
        item.target_comparator_contaminated for item in case_results
    )
    summary = {
        "metric_name": "structural_quality_score",
        "evaluation_scope": "deterministic_parser_smoke_test",
        "scoring_note": (
            "A deterministic claim-structure quality score; not scientific truth "
            "accuracy or evidence correctness."
        ),
        "case_count": len(case_results),
        "aggregate_score": round(mean(scores), 2) if scores else 0.0,
        "min_score": round(min(scores), 2) if scores else 0.0,
        "max_score": round(max(scores), 2) if scores else 0.0,
        "domains": sorted({item.domain for item in case_results}),
        "domain_scores": domain_scores,
        "component_means": component_means,
        "structural_pass_threshold": pass_threshold,
        "structural_pass_count": passed,
        "structural_pass_rate": round(passed / len(scores), 3) if scores else 0.0,
        "exact_copy_rate": (
            round(exact_copies / len(case_results), 3) if case_results else 0.0
        ),
        "input_concept_leakage_rate": (
            round(mean(input_concept_leakage), 3)
            if input_concept_leakage
            else 0.0
        ),
        "annotated_slot_case_count": len(grounded_results),
        "annotated_slot_concept_accuracy": (
            round(mean(item.slot_grounding["overall"] for item in grounded_results), 3)
            if grounded_results
            else None
        ),
        "annotated_slot_scores": annotated_slot_scores,
        "target_comparator_contamination_rate": (
            round(target_comparator_contamination / len(cases), 3) if cases else 0.0
        ),
        "lowest_case_ids": [item.case_id for item in lowest_cases],
        "engine": engine.__class__.__name__,
    }
    return BenchmarkReport(summary=summary, case_results=case_results)


def _payload_to_case(payload: object, *, line_number: int) -> BenchmarkCase:
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number}: benchmark case must be an object")
    case_id = _required_text(payload, "id", line_number)
    direction = _required_text(payload, "direction", line_number)
    domain = _required_text(payload, "domain", line_number)
    if domain not in VALID_DOMAINS:
        raise ValueError(f"line {line_number}: invalid domain {domain!r}")
    required_concepts = _required_text_list(
        payload, "required_concepts", line_number
    )
    forbidden_overclaims = _required_text_list(
        payload, "forbidden_overclaims", line_number
    )
    expected_slots = _optional_slot_map(payload, line_number)
    return BenchmarkCase(
        id=case_id,
        direction=direction,
        domain=domain,
        required_concepts=required_concepts,
        forbidden_overclaims=forbidden_overclaims,
        expected_slots=expected_slots,
    )


def _required_text(payload: dict[str, object], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {key} must be non-empty text")
    return " ".join(value.split())


def _required_text_list(
    payload: dict[str, object], key: str, line_number: int
) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"line {line_number}: {key} must be a non-empty list")
    items = [
        " ".join(item.split())
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    if not items:
        raise ValueError(f"line {line_number}: {key} must be a non-empty list")
    return items


def _optional_slot_map(
    payload: dict[str, object], line_number: int
) -> dict[str, list[str]]:
    raw = payload.get("expected_slots")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_number}: expected_slots must be an object")
    allowed = {"method_or_mechanism", "target_or_task", "conditions"}
    slots: dict[str, list[str]] = {}
    for name, values in raw.items():
        if name not in allowed:
            raise ValueError(f"line {line_number}: unsupported expected slot {name!r}")
        if not isinstance(values, list):
            raise ValueError(f"line {line_number}: expected slot {name!r} must be a list")
        terms = [
            " ".join(value.split())
            for value in values
            if isinstance(value, str) and value.strip()
        ]
        if not terms:
            raise ValueError(f"line {line_number}: expected slot {name!r} must not be empty")
        slots[name] = terms
    return slots


def _slot_grounding(
    result: CoreClaimResult,
    case: BenchmarkCase,
) -> dict[str, float]:
    candidate = result.selected_candidate
    if candidate is None or not case.expected_slots:
        return {}
    predicted = {
        "method_or_mechanism": candidate.method_or_mechanism,
        "target_or_task": candidate.target_or_task,
        "conditions": " ".join(candidate.conditions),
    }
    scores = {
        name: len(_found_terms(predicted.get(name, ""), terms)) / len(terms)
        for name, terms in case.expected_slots.items()
    }
    scores["overall"] = mean(scores.values()) if scores else 0.0
    return {name: round(value, 3) for name, value in scores.items()}


def _comparison_regex() -> str:
    return "|".join(
        re.escape(marker.strip()).replace(r"\ ", r"\s+")
        for marker in COMPARISON_MARKERS
        if marker.strip()
    )


def _slot_coverage(result: CoreClaimResult) -> float:
    candidate = result.selected_candidate
    if candidate is None:
        return 1.0 if result.selected_claim else 0.0
    slots = [
        candidate.method_or_mechanism,
        candidate.target_or_task,
        candidate.expected_effect,
        candidate.falsification_test,
    ]
    if candidate.conditions:
        slots.append(" ".join(candidate.conditions))
    return sum(bool(slot) for slot in slots) / len(slots)


def _declarative_form(claim: str) -> float:
    if not claim or claim.endswith("?"):
        return 0.0
    token_count = len(_tokens(claim))
    if token_count < 4:
        return 0.25
    effect_pattern = (
        r"\b(improves?|reduces?|increases?|decreases?|predicts?|detects?|"
        r"lowers?|raises?|outperforms?)\b"
    )
    if re.search(effect_pattern, claim, re.IGNORECASE):
        return 1.0
    return 0.65


def _marker_score(text: str, markers: tuple[str, ...]) -> float:
    matches = sum(1 for marker in markers if _contains_marker(text, marker))
    if not matches:
        return 0.0
    return min(1.0, 0.55 + 0.15 * matches)


def _falsifiability(claim: str, result: CoreClaimResult) -> float:
    lower_claim = claim.lower()
    score = 0.0
    score += 0.30 * _marker_score(lower_claim, COMPARISON_MARKERS)
    score += 0.30 * _marker_score(lower_claim, MEASUREMENT_MARKERS)
    if any(_contains_marker(lower_claim, marker) for marker in BOUNDARY_MARKERS):
        score += 0.20
    candidate = result.selected_candidate
    falsification_test = candidate.falsification_test if candidate else ""
    if falsification_test and "compare" in falsification_test.lower():
        score += 0.20
    elif falsification_test:
        score += 0.10
    return min(1.0, score)


def _specificity(direction: str, claim: str) -> float:
    claim_tokens = _tokens(claim)
    direction_tokens = set(_tokens(direction))
    if not claim_tokens:
        return 0.0
    non_direction_tokens = [
        token
        for token in claim_tokens
        if token not in direction_tokens and len(token) > 3
    ]
    score = 0.0
    score += min(0.35, len(claim_tokens) / 45)
    score += min(0.35, len(non_direction_tokens) / 18)
    lower_claim = claim.lower()
    if any(_contains_marker(lower_claim, marker) for marker in BOUNDARY_MARKERS):
        score += 0.15
    if _marker_score(lower_claim, MEASUREMENT_MARKERS):
        score += 0.15
    return min(1.0, score)


def _overclaim_penalty(lower_claim: str) -> float:
    matches = [
        marker for marker in OVERCLAIM_MARKERS if _contains_marker(lower_claim, marker)
    ]
    if re.search(
        r"\balways\s+(?:works?|improves?|reduces?|increases?|decreases?|"
        r"outperforms?|succeeds?|accurate|correct|safe|effective|stable|factual)\b",
        lower_claim,
    ):
        matches.append("always-claim")
    return min(35.0, 12.0 * len(matches))


def _required_concept_coverage(claim: str, case: BenchmarkCase | None) -> float:
    if case is None:
        return 1.0
    found = _found_terms(claim, case.required_concepts)
    return len(found) / len(case.required_concepts)


def _case_forbidden_overclaim_penalty(
    claim: str, case: BenchmarkCase | None
) -> float:
    if case is None:
        return 0.0
    found = _found_terms(claim, case.forbidden_overclaims)
    return min(30.0, 18.0 * len(found))


def _cue_stuffing_penalty(lower_claim: str) -> float:
    tokens = _tokens(lower_claim)
    cue_matches = sum(
        1
        for marker in COMPARISON_MARKERS + MEASUREMENT_MARKERS + BOUNDARY_MARKERS
        if _contains_marker(lower_claim, marker)
    )
    if len(tokens) <= 24 or cue_matches < 8:
        return 0.0
    unsupported_length = max(0, len(tokens) - 28)
    return min(18.0, (cue_matches - 7) * 1.4 + unsupported_length * 0.25)


def _concept_matches(claim: str, concepts: list[str]) -> tuple[list[str], list[str]]:
    found = _found_terms(claim, concepts)
    found_set = set(found)
    missing = [concept for concept in concepts if concept not in found_set]
    return found, missing


def _found_terms(claim: str, terms: list[str]) -> list[str]:
    lower_claim = claim.lower()
    return [term for term in terms if _contains_marker(lower_claim, term.lower())]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _contains_marker(text: str, marker: str) -> bool:
    normalized = marker.strip()
    if not normalized:
        return False
    if normalized == "%":
        return "%" in text
    pattern = re.escape(normalized).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text))


def _normalized_copy(text: str) -> str:
    return " ".join(_tokens(text))

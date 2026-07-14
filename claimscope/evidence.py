from __future__ import annotations

import re

from .text_utils import retrieval_tokens


SUPPORT_MARKERS = (
    "improve",
    "improves",
    "improved",
    "increase",
    "increased",
    "effective",
    "outperform",
    "outperformed",
    "benefit",
    "supports",
    "associated with",
    "significantly greater",
    "significantly higher",
)

REFUTE_MARKERS = (
    "does not",
    "did not",
    "do not",
    "fails to",
    "failed to",
    "contradict",
    "contrary to",
    "not associated with",
    "worse than",
    "lower performance",
    "reduced performance",
)

LIMITATION_MARKERS = (
    "however",
    "limitation",
    "limitations",
    "limited to",
    "failure mode",
    "brittle",
    "under domain shift",
    "only evaluated",
    "small sample",
    "single dataset",
    "lack of",
    "insufficient",
    "risk of bias",
)

NULL_RESULT_MARKERS = (
    "no consistent",
    "no improvement",
    "null result",
    "no significant",
    "not significant",
    "non-significant",
    "nonsignificant",
    "does not improve",
    "marginal gain",
    "negative result",
    "no statistically significant",
    "no difference",
    "did not differ",
    "comparable between",
)

INCREASE_MARKERS = (
    "increase",
    "increased",
    "higher",
    "greater",
    "more than",
    "doubled",
    "rose",
    "improved",
)

DECREASE_MARKERS = (
    "decrease",
    "decreased",
    "reduction",
    "reduced",
    "lower",
    "less than",
    "fewer",
    "declined",
)


def classify_evidence_stance(text: str) -> str:
    """Return one public evidence bucket without claiming full NLI accuracy."""

    lower = (text or "").lower()
    if _contains_any(lower, NULL_RESULT_MARKERS) or _has_non_significant_p_value(lower):
        return "null_result"
    if _contains_any(lower, LIMITATION_MARKERS):
        return "limit"
    if _contains_any(lower, REFUTE_MARKERS):
        return "contradict"
    if _contains_any(lower, SUPPORT_MARKERS):
        return "support"
    return "mention"


def infer_effect_label(text: str, outcome: str = "") -> str:
    """Infer Evidence Inference's increase/decrease/null label or abstain."""

    lower = (text or "").lower()
    if _contains_any(lower, NULL_RESULT_MARKERS) or _has_non_significant_p_value(lower):
        return "0"
    increased = _contains_any(lower, INCREASE_MARKERS)
    decreased = _contains_any(lower, DECREASE_MARKERS)
    if increased and not decreased:
        return "1"
    if decreased and not increased:
        return "-1"
    if "superior to" in lower:
        negative_outcome_terms = {
            "mortality",
            "error",
            "loss",
            "pain",
            "adverse",
            "duration",
            "symptom",
            "incidence",
            "risk",
        }
        outcome_terms = set(retrieval_tokens(outcome))
        return "-1" if outcome_terms & negative_outcome_terms else "1"
    return "unknown"


def infer_claim_label(claim: str, evidence: str) -> str:
    """Conservative lexical NLI baseline for gold evidence snippets."""

    claim_tokens = set(retrieval_tokens(claim))
    evidence_tokens = set(retrieval_tokens(evidence))
    if not claim_tokens or not evidence_tokens:
        return "unknown"
    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    claim_negative = _has_negation(claim)
    evidence_negative = _has_negation(evidence)
    number_conflict = _numbers_conflict(claim, evidence)
    if overlap >= 0.2 and (claim_negative != evidence_negative or number_conflict):
        return "contradiction"
    if overlap >= 0.32 and not number_conflict:
        return "entailment"
    return "unknown"


def is_null_result(text: str) -> bool:
    lower = (text or "").lower()
    return _contains_any(lower, NULL_RESULT_MARKERS) or _has_non_significant_p_value(lower)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _has_negation(text: str) -> bool:
    lower = (text or "").lower()
    return bool(
        re.search(
            r"\b(?:no|not|never|neither|without|none|cannot|can't|didn't|doesn't)\b",
            lower,
        )
    )


def _has_non_significant_p_value(text: str) -> bool:
    for value in re.findall(r"\bp\s*[=>]\s*(0?\.\d+)", text):
        try:
            if float(value) >= 0.05:
                return True
        except ValueError:
            continue
    return False


def _numbers_conflict(claim: str, evidence: str) -> bool:
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?", claim))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?", evidence))
    return bool(claim_numbers and evidence_numbers and not claim_numbers <= evidence_numbers)

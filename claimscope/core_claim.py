from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from time import perf_counter
from typing import Callable

from .models import AgentEvent, ClaimCandidate, ClaimCritique, CoreClaimResult

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
PROPOSER_ROLES = {
    "operationalizer": (
        "Turn the direction into measurable variables and a controlled comparison."
    ),
    "mechanism_analyst": (
        "State the mechanism, target, expected effect, and boundary conditions."
    ),
    "skeptical_empiricist": (
        "Write the narrowest claim that an adverse result could falsify."
    ),
}
CRITIC_ROLES = {
    "falsifiability_critic": (
        "Check whether each candidate can be falsified by an observable result."
    ),
    "scope_critic": "Check whether each candidate has an appropriately bounded scope.",
}
RUBRIC_KEYS = (
    "specificity",
    "falsifiability",
    "mechanism",
    "scope",
    "measurability",
    "risk_awareness",
)
_CANDIDATE_SCHEMA = (
    '{"claim": "testable claim", "method_or_mechanism": "mechanism", '
    '"target_or_task": "task", "expected_effect": "measurable effect", '
    '"conditions": ["condition"], "falsification_test": "test", '
    '"missing_information": ["missing field"], "confidence": 0.0}'
)
_CRITIQUE_SCHEMA = (
    '{"critiques": [{"candidate_id": "proposer id", "rubric_scores": {'
    '"specificity": 0, "falsifiability": 0, "mechanism": 0, "scope": 0, '
    '"measurability": 0, "risk_awareness": 0}, "reason_codes": ["code"], '
    '"revision": "stronger public claim"}]}'
)
_JUDGE_SCHEMA = (
    '{"selected_candidate_id": "proposer id", "final_claim": "final claim", '
    '"falsification_test": "test", "unresolved_ambiguities": ["ambiguity"]}'
)

AgentEventCallback = Callable[[AgentEvent], None]


class CoreClaimArenaError(RuntimeError):
    pass


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


class CoreClaimArena:
    def __init__(
        self,
        llm_client: object,
        max_workers: int = 3,
        timeout: float = 60,
        allow_heuristic_fallback: bool = True,
    ):
        self.llm_client = llm_client
        self.max_workers = max(1, min(3, int(max_workers)))
        self.timeout = timeout
        self.allow_heuristic_fallback = allow_heuristic_fallback

    def run(
        self,
        direction: str,
        on_event: AgentEventCallback | None = None,
    ) -> CoreClaimResult:
        candidates, proposer_events = self._run_proposers(direction, on_event)
        if not candidates:
            if not self.allow_heuristic_fallback:
                raise CoreClaimArenaError(
                    "Multi-agent arena produced no valid proposer candidates."
                )
            fallback = HeuristicCoreClaimEngine().run(direction)
            return replace(
                fallback,
                events=proposer_events + fallback.events,
                mode="multi_agent",
                degraded=True,
            )
        if not self.allow_heuristic_fallback and (
            len(candidates) != len(PROPOSER_ROLES)
            or any(event.status != "complete" for event in proposer_events)
        ):
            raise CoreClaimArenaError(
                "Strict multi-agent execution requires every proposer artifact."
            )

        critiques: list[ClaimCritique] = []
        critic_events: list[AgentEvent] = []
        for role, rubric in CRITIC_ROLES.items():
            _notify(on_event, _running_event(role))
            critique, event = self._run_critic(direction, role, rubric, candidates)
            critiques.extend(critique)
            critic_events.append(event)
            _notify(on_event, event)
            if not self.allow_heuristic_fallback and (
                event.status != "complete" or len(critique) != len(candidates)
            ):
                raise CoreClaimArenaError(
                    "Strict multi-agent execution requires every critic artifact."
                )

        _notify(on_event, _running_event("judge"))
        selected_candidate, final_claim, ambiguities, judge_event, judge_degraded = (
            self._run_judge(direction, candidates, critiques)
        )
        _notify(on_event, judge_event)
        if not self.allow_heuristic_fallback and (
            judge_degraded or judge_event.status != "complete"
        ):
            raise CoreClaimArenaError(
                "Strict multi-agent execution requires a valid judge artifact."
            )
        events = proposer_events + critic_events + [judge_event]
        degraded = judge_degraded or any(
            event.status == "failed" for event in proposer_events + critic_events
        )
        return CoreClaimResult(
            direction=direction,
            selected_claim=final_claim,
            selected_candidate=selected_candidate,
            candidates=candidates,
            critiques=critiques,
            events=events,
            unresolved_ambiguities=ambiguities,
            mode="multi_agent",
            degraded=degraded,
        )

    def _run_proposers(
        self,
        direction: str,
        on_event: AgentEventCallback | None = None,
    ) -> tuple[list[ClaimCandidate], list[AgentEvent]]:
        candidates_by_role: dict[str, ClaimCandidate] = {}
        events_by_role: dict[str, AgentEvent] = {}
        for role in PROPOSER_ROLES:
            _notify(on_event, _running_event(role))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_proposer, direction, role, rubric): role
                for role, rubric in PROPOSER_ROLES.items()
            }
            for future in as_completed(futures):
                role = futures[future]
                try:
                    candidate, event = future.result()
                except Exception:
                    candidate = None
                    event = _event(
                        role=role,
                        status="failed",
                        public_summary="The proposer did not return a usable public claim.",
                        started=perf_counter(),
                        artifacts={},
                    )
                if candidate is not None:
                    candidates_by_role[role] = candidate
                events_by_role[role] = event
                _notify(on_event, event)
        return (
            [
                candidates_by_role[role]
                for role in PROPOSER_ROLES
                if role in candidates_by_role
            ],
            [events_by_role[role] for role in PROPOSER_ROLES if role in events_by_role],
        )

    def _run_proposer(
        self, direction: str, role: str, rubric: str
    ) -> tuple[ClaimCandidate | None, AgentEvent]:
        started = perf_counter()
        try:
            content = self.llm_client.chat(
                _proposer_messages(direction, role, rubric),
                temperature=0.2,
                timeout=self.timeout,
            )
            payload = _extract_json_object(content)
            candidate = _payload_to_candidate(payload, proposer=role)
            return candidate, _event(
                role=role,
                status="complete",
                public_summary="Proposed one public core-claim candidate.",
                started=started,
                artifacts={"candidate_id": role},
                scores={"candidate_score": candidate.score()},
            )
        except Exception:
            return None, _event(
                role=role,
                status="failed",
                public_summary="The proposer did not return a usable public claim.",
                started=started,
                artifacts={},
            )

    def _run_critic(
        self,
        direction: str,
        role: str,
        rubric: str,
        candidates: list[ClaimCandidate],
    ) -> tuple[list[ClaimCritique], AgentEvent]:
        started = perf_counter()
        try:
            content = self.llm_client.chat(
                _critic_messages(direction, role, rubric, candidates),
                temperature=0.1,
                timeout=self.timeout,
            )
            payload = _extract_json_object(content)
            critiques = _payload_to_critiques(
                payload,
                critic=role,
                candidate_ids={candidate.proposer for candidate in candidates},
            )
            return critiques, _event(
                role=role,
                status="complete",
                public_summary="Critiqued public candidates against the rubric.",
                started=started,
                artifacts={"critique_count": len(critiques)},
            )
        except Exception:
            return [], _event(
                role=role,
                status="failed",
                public_summary="The critic did not return usable public critique data.",
                started=started,
                artifacts={},
            )

    def _run_judge(
        self,
        direction: str,
        candidates: list[ClaimCandidate],
        critiques: list[ClaimCritique],
    ) -> tuple[ClaimCandidate, str, list[str], AgentEvent, bool]:
        started = perf_counter()
        try:
            content = self.llm_client.chat(
                _judge_messages(direction, candidates, critiques),
                temperature=0.0,
                timeout=self.timeout,
            )
            payload = _extract_json_object(content)
            candidate_id = _coerce_text(payload.get("selected_candidate_id"))
            final_claim = _coerce_text(payload.get("final_claim"))
            falsification_test = _coerce_text(payload.get("falsification_test"))
            if not candidate_id or not final_claim or not falsification_test:
                raise ValueError("judge response missing required fields")
            if not isinstance(payload.get("unresolved_ambiguities"), list):
                raise ValueError("judge unresolved_ambiguities must be a list")
            selected = _candidate_by_id(candidates, candidate_id)
            selected = replace(
                selected,
                claim=final_claim,
                falsification_test=falsification_test,
            )
            ambiguities = _coerce_string_list(payload.get("unresolved_ambiguities"))
            return (
                selected,
                final_claim,
                ambiguities,
                _event(
                    role="judge",
                    status="complete",
                    public_summary="Selected the final public core claim.",
                    started=started,
                    artifacts={"selected_candidate_id": selected.proposer},
                ),
                False,
            )
        except Exception:
            selected = _fallback_candidate(candidates)
            return (
                selected,
                selected.claim,
                list(selected.missing_information),
                _event(
                    role="judge",
                    status="degraded",
                    public_summary=(
                        "Judge selection failed; used deterministic fallback."
                    ),
                    started=started,
                    artifacts={"selected_candidate_id": selected.proposer},
                    scores={"selected_candidate": selected.score()},
                ),
                True,
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
    if not re.search(
        r"\b(versus|vs\.?|compared (?:with|to)|baseline|control|without|against|than)\b",
        claim,
        re.IGNORECASE,
    ):
        missing_information.append("comparison baseline")
    if not re.search(
        r"\b(accuracy|auc|f1|precision|recall|rate|score|error|latency|throughput|"
        r"mortality|yield|cost|time|percent|metric|change)\b|%",
        claim,
        re.IGNORECASE,
    ):
        missing_information.append("evaluation metric")
    if not conditions:
        missing_information.append("boundary conditions")
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
    total_slots = 6
    covered_slots = total_slots - len(missing_information)
    return round(max(0, covered_slots) / total_slots, 2)


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


def _proposer_messages(direction: str, role: str, rubric: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"You are the {role} proposer in ClaimScope's public core-claim review. "
                f"{rubric} Use the same language as the research direction for every "
                "human-readable JSON value. Return JSON only. Do not include chain-of-thought, private "
                "reasoning, credentials, markdown, citations, or prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Direction:\n{direction}\n\n"
                "Return exactly this JSON schema:\n"
                f"{_CANDIDATE_SCHEMA}"
            ),
        },
    ]


def _critic_messages(
    direction: str,
    role: str,
    rubric: str,
    candidates: list[ClaimCandidate],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"You are the {role} critic in ClaimScope's public core-claim review. "
                f"{rubric} Score each candidate from 0 to 5 on the six public rubric "
                "keys. Use the same language as the research direction for every "
                "human-readable JSON value. Return JSON only. Do not include chain-of-thought, private "
                "reasoning, credentials, markdown, citations, or prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Direction:\n{direction}\n\n"
                "Candidates:\n"
                f"{json.dumps(_candidate_payloads(candidates), sort_keys=True, ensure_ascii=False)}\n\n"
                "Return exactly this JSON schema:\n"
                f"{_CRITIQUE_SCHEMA}"
            ),
        },
    ]


def _judge_messages(
    direction: str,
    candidates: list[ClaimCandidate],
    critiques: list[ClaimCritique],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the judge in ClaimScope's public core-claim review. Select the "
                "best candidate using only the structured public candidates and public "
                "critiques. Use the same language as the research direction for every "
                "human-readable JSON value. Return JSON only. Do not include chain-of-thought, private "
                "reasoning, credentials, markdown, citations, or prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Direction:\n{direction}\n\n"
                "Candidates:\n"
                f"{json.dumps(_candidate_payloads(candidates), sort_keys=True, ensure_ascii=False)}\n\n"
                "Critiques:\n"
                f"{json.dumps([item.to_dict() for item in critiques], sort_keys=True, ensure_ascii=False)}\n\n"
                "Return exactly this JSON schema:\n"
                f"{_JUDGE_SCHEMA}"
            ),
        },
    ]


def _candidate_payloads(candidates: list[ClaimCandidate]) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate.proposer,
            "claim": candidate.claim,
            "method_or_mechanism": candidate.method_or_mechanism,
            "target_or_task": candidate.target_or_task,
            "expected_effect": candidate.expected_effect,
            "conditions": candidate.conditions,
            "falsification_test": candidate.falsification_test,
            "missing_information": candidate.missing_information,
            "confidence": candidate.confidence,
            "score": candidate.score(),
        }
        for candidate in candidates
    ]


def _payload_to_candidate(payload: dict, proposer: str) -> ClaimCandidate:
    claim = _coerce_text(payload.get("claim"))
    if not claim:
        raise ValueError("missing claim")
    required_text = {
        "method_or_mechanism": _coerce_text(payload.get("method_or_mechanism")),
        "target_or_task": _coerce_text(payload.get("target_or_task")),
        "expected_effect": _coerce_text(payload.get("expected_effect")),
        "falsification_test": _coerce_text(payload.get("falsification_test")),
    }
    missing = [name for name, value in required_text.items() if not value]
    if missing:
        raise ValueError("candidate response missing required fields")
    conditions = _coerce_string_list(payload.get("conditions"))
    if not conditions:
        raise ValueError("candidate conditions must be non-empty")
    if not isinstance(payload.get("missing_information"), list):
        raise ValueError("candidate missing_information must be a list")
    if not isinstance(payload.get("confidence"), (int, float)):
        raise ValueError("candidate confidence must be numeric")
    return ClaimCandidate(
        claim=claim,
        method_or_mechanism=required_text["method_or_mechanism"],
        target_or_task=required_text["target_or_task"],
        expected_effect=required_text["expected_effect"],
        conditions=conditions,
        falsification_test=required_text["falsification_test"],
        missing_information=_coerce_string_list(payload.get("missing_information")),
        confidence=_coerce_float(payload.get("confidence"), default=0.0),
        proposer=proposer,
    )


def _payload_to_critiques(
    payload: dict, critic: str, candidate_ids: set[str]
) -> list[ClaimCritique]:
    raw_critiques = payload.get("critiques")
    if not isinstance(raw_critiques, list) or not raw_critiques:
        raise ValueError("critic response must include critiques")
    critiques: list[ClaimCritique] = []
    seen_candidate_ids: set[str] = set()
    for item in raw_critiques:
        if not isinstance(item, dict):
            raise ValueError("critique item must be an object")
        candidate_id = _coerce_text(item.get("candidate_id"))
        if not candidate_id or candidate_id not in candidate_ids:
            raise ValueError("critique candidate_id is unknown")
        if candidate_id in seen_candidate_ids:
            raise ValueError("duplicate critique candidate_id")
        seen_candidate_ids.add(candidate_id)
        reason_codes = _coerce_string_list(item.get("reason_codes"))
        revision = _coerce_text(item.get("revision"))
        if not reason_codes:
            raise ValueError("critique reason_codes must be non-empty")
        if not revision:
            raise ValueError("critique revision must be non-empty")
        critiques.append(
            ClaimCritique(
                candidate_id=candidate_id,
                critic=critic,
                rubric_scores=_rubric_scores(item.get("rubric_scores")),
                reason_codes=reason_codes,
                revision=revision,
                public_summary=_coerce_text(item.get("public_summary")),
            )
        )
    return critiques


def _rubric_scores(raw_scores: object) -> dict[str, float]:
    if not isinstance(raw_scores, dict):
        raise ValueError("critique rubric_scores must be an object")
    missing_keys = [key for key in RUBRIC_KEYS if key not in raw_scores]
    if missing_keys:
        raise ValueError("critique rubric_scores missing required keys")
    for key in RUBRIC_KEYS:
        if not isinstance(raw_scores.get(key), (int, float)):
            raise ValueError("critique rubric_scores must be numeric")
    return {
        key: max(0.0, min(5.0, _coerce_float(raw_scores.get(key), default=0.0)))
        for key in RUBRIC_KEYS
    }


def _extract_json_object(content: str) -> dict:
    start = content.find("{")
    if start < 0:
        raise ValueError("response did not contain a JSON object")
    in_string = False
    escaped = False
    depth = 0
    for index in range(start, len(content)):
        char = content[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(content[start : index + 1])
                if not isinstance(payload, dict):
                    raise ValueError("response JSON must be an object")
                return payload
    raise ValueError("response did not contain a complete JSON object")


def _candidate_by_id(
    candidates: list[ClaimCandidate], candidate_id: str
) -> ClaimCandidate:
    for candidate in candidates:
        if candidate.proposer == candidate_id:
            return candidate
    raise ValueError("unknown selected_candidate_id")


def _fallback_candidate(candidates: list[ClaimCandidate]) -> ClaimCandidate:
    return max(candidates, key=lambda candidate: candidate.score())


def _event(
    *,
    role: str,
    status: str,
    public_summary: str,
    started: float,
    artifacts: dict[str, object],
    scores: dict[str, float] | None = None,
) -> AgentEvent:
    return AgentEvent(
        stage="core_claim",
        role=role,
        status=status,
        public_summary=public_summary,
        duration_ms=max(0, round((perf_counter() - started) * 1000)),
        artifacts=artifacts,
        scores=scores or {},
    )


def _running_event(role: str) -> AgentEvent:
    if role in PROPOSER_ROLES:
        summary = "Generating one structured public claim candidate."
    elif role in CRITIC_ROLES:
        summary = "Reviewing public candidates against the assigned rubric."
    else:
        summary = "Comparing public candidates and critiques for final selection."
    return AgentEvent(
        stage="core_claim",
        role=role,
        status="running",
        public_summary=summary,
        artifacts={},
    )


def _notify(
    callback: AgentEventCallback | None,
    event: AgentEvent,
) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        return


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _coerce_text(item))]


def _coerce_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _coerce_float(value: object, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default

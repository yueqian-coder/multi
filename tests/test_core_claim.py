import importlib
import io
import json
import urllib.error

import pytest


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response

    def chat(self, messages, temperature=0.2):
        return self.response


class RoleAwareFakeLLM:
    expected_judgment = (
        "Retrieval-augmented medical QA reduces unsupported answers when source "
        "quality is audited against clinician-reviewed references."
    )

    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=0.2, timeout=None):
        content = "\n".join(message["content"] for message in messages)
        self.calls.append(content)
        if "judge" in content:
            return json.dumps(
                {
                    "selected_candidate_id": "mechanism_analyst",
                    "final_claim": self.expected_judgment,
                    "falsification_test": (
                        "Randomize QA cases to retrieval and no-retrieval settings."
                    ),
                    "unresolved_ambiguities": ["source quality threshold"],
                }
            )
        if "falsifiability_critic" in content:
            return json.dumps(
                {
                    "critiques": [
                        {
                            "candidate_id": "mechanism_analyst",
                            "rubric_scores": {
                                "specificity": 4,
                                "falsifiability": 5,
                                "mechanism": 5,
                                "scope": 4,
                                "measurability": 4,
                                "risk_awareness": 3,
                            },
                            "reason_codes": ["testable_outcome"],
                            "revision": self.expected_judgment,
                        }
                    ]
                }
            )
        if "scope_critic" in content:
            return json.dumps(
                {
                    "critiques": [
                        {
                            "candidate_id": "mechanism_analyst",
                            "rubric_scores": {
                                "specificity": 5,
                                "falsifiability": 4,
                                "mechanism": 4,
                                "scope": 5,
                                "measurability": 4,
                                "risk_awareness": 4,
                            },
                            "reason_codes": ["bounded_context"],
                            "revision": self.expected_judgment,
                        }
                    ]
                }
            )
        if "operationalizer" in content:
            return json.dumps(
                {
                    "claim": (
                        "Retrieval-augmented medical QA lowers unsupported answer "
                        "rates versus closed-book QA on clinician-reviewed cases."
                    ),
                    "method_or_mechanism": "retrieval-augmented answer grounding",
                    "target_or_task": "medical question answering",
                    "expected_effect": "lowers unsupported answer rates",
                    "conditions": ["clinician-reviewed cases"],
                    "falsification_test": (
                        "Compare unsupported answer rates against closed-book QA."
                    ),
                    "missing_information": [],
                    "confidence": 0.72,
                }
            )
        if "mechanism_analyst" in content:
            return json.dumps(
                {
                    "claim": (
                        "Curated retrieval evidence improves medical QA safety by "
                        "forcing answers to cite relevant clinical references."
                    ),
                    "method_or_mechanism": "curated retrieval evidence",
                    "target_or_task": "medical QA safety",
                    "expected_effect": "improves answer grounding",
                    "conditions": ["relevant clinical references are available"],
                    "falsification_test": (
                        "Measure citation support and harmful unsupported answers."
                    ),
                    "missing_information": [],
                    "confidence": 0.81,
                }
            )
        if "skeptical_empiricist" in content:
            return json.dumps(
                {
                    "claim": (
                        "Retrieval does not make medical QA safer unless retrieved "
                        "sources match the patient context."
                    ),
                    "method_or_mechanism": "patient-context-matched retrieval",
                    "target_or_task": "medical QA",
                    "expected_effect": "reduces unsafe unsupported recommendations",
                    "conditions": ["patient context is present in retrieved sources"],
                    "falsification_test": (
                        "Audit unsafe recommendations under context-mismatched "
                        "retrieval."
                    ),
                    "missing_information": ["retrieval quality threshold"],
                    "confidence": 0.64,
                }
            )
        raise AssertionError(f"unexpected prompt: {content}")


class JudgeFailingFakeLLM(RoleAwareFakeLLM):
    def chat(self, messages, temperature=0.2, timeout=None):
        content = "\n".join(message["content"] for message in messages)
        if "judge" in content:
            raise RuntimeError("judge unavailable")
        return super().chat(messages, temperature=temperature, timeout=timeout)


def _load_core_claim_engine():
    try:
        module = importlib.import_module("claimscope.core_claim")
    except ModuleNotFoundError as exc:
        pytest.fail(str(exc))
    assert hasattr(module, "HeuristicCoreClaimEngine")
    return module.HeuristicCoreClaimEngine


def _load_claim_candidate():
    module = importlib.import_module("claimscope.models")
    assert hasattr(module, "ClaimCandidate")
    return module.ClaimCandidate


def _load_planners():
    module = importlib.import_module("claimscope.planner")
    assert hasattr(module, "HeuristicClaimPlanner")
    assert hasattr(module, "LLMClaimPlanner")
    return module.HeuristicClaimPlanner, module.LLMClaimPlanner


def test_heuristic_core_claim_reports_missing_slots_and_trace():
    HeuristicCoreClaimEngine = _load_core_claim_engine()
    result = HeuristicCoreClaimEngine().run("use an error map loop for segmentation")

    assert result.selected_claim
    assert result.selected_candidate
    assert result.candidates
    assert "measurable expected effect" in result.unresolved_ambiguities
    assert result.events[-1].status == "complete"
    assert result.mode == "heuristic"


def test_core_claim_result_serializes_without_private_reasoning():
    HeuristicCoreClaimEngine = _load_core_claim_engine()
    result = HeuristicCoreClaimEngine().run("use an error map loop for segmentation")

    payload = result.to_dict()

    assert "chain_of_thought" not in json.dumps(payload).lower()
    assert payload["selected_candidate"]["falsification_test"]


def test_claim_candidate_clamps_confidence_and_scores_deterministically():
    ClaimCandidate = _load_claim_candidate()
    candidate = ClaimCandidate(
        claim="Error maps guide a segmentation refinement loop",
        method_or_mechanism="error map refinement loop",
        target_or_task="segmentation",
        missing_information=["measurable expected effect"],
        confidence=1.7,
    )

    assert candidate.confidence == 1.0
    assert candidate.score() == 70.0


def test_heuristic_core_claim_extracts_mixed_case_condition_markers():
    HeuristicCoreClaimEngine = _load_core_claim_engine()
    result = HeuristicCoreClaimEngine().run(
        "use an error map loop to improve segmentation With expert prompts"
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.conditions == ["expert prompts"]
    assert result.selected_candidate.target_or_task == "segmentation"
    assert result.selected_candidate.expected_effect == "improve segmentation"


def test_planners_expose_typed_core_claim_results_without_breaking_string_api():
    HeuristicClaimPlanner, LLMClaimPlanner = _load_planners()
    heuristic_planner = HeuristicClaimPlanner()
    heuristic_result = heuristic_planner.extract_core_claim_result(
        "use an error map loop for segmentation"
    )

    assert heuristic_result.selected_claim == heuristic_planner.extract_core_claim(
        "use an error map loop for segmentation"
    )

    llm_planner = LLMClaimPlanner(
        llm_client=FakeLLMClient(
            '{"normalized_claim": '
            '"Error maps guide promptable medical segmentation refinement"}'
        ),
        fallback=HeuristicClaimPlanner(),
    )
    llm_result = llm_planner.extract_core_claim_result(
        "use an error map loop for segmentation"
    )

    assert llm_result.selected_claim == (
        "Error maps guide promptable medical segmentation refinement"
    )
    assert llm_result.selected_claim == llm_planner.extract_core_claim(
        "use an error map loop for segmentation"
    )


def test_llm_planner_core_claim_result_uses_multi_agent_arena():
    _, LLMClaimPlanner = _load_planners()
    client = RoleAwareFakeLLM()
    planner = LLMClaimPlanner(llm_client=client)

    result = planner.extract_core_claim_result("Can retrieval make medical QA safer?")

    assert result.mode == "multi_agent"
    assert result.selected_claim == client.expected_judgment
    assert {candidate.proposer for candidate in result.candidates} == {
        "operationalizer",
        "mechanism_analyst",
        "skeptical_empiricist",
    }


def test_arena_runs_three_proposers_two_critics_and_judge():
    module = importlib.import_module("claimscope.core_claim")
    client = RoleAwareFakeLLM()

    result = module.CoreClaimArena(client).run("Can retrieval make medical QA safer?")

    assert {candidate.proposer for candidate in result.candidates} == {
        "operationalizer",
        "mechanism_analyst",
        "skeptical_empiricist",
    }
    assert {item.critic for item in result.critiques} == {
        "falsifiability_critic",
        "scope_critic",
    }
    assert result.mode == "multi_agent"
    assert result.selected_claim == client.expected_judgment
    assert result.selected_candidate.claim == client.expected_judgment
    assert len(result.events) == 6


def test_arena_falls_back_to_weighted_candidate_when_judge_fails():
    module = importlib.import_module("claimscope.core_claim")

    result = module.CoreClaimArena(JudgeFailingFakeLLM()).run(
        "Can retrieval make medical QA safer?"
    )

    assert result.degraded is True
    assert result.selected_candidate == max(
        result.candidates, key=lambda item: item.score()
    )


def test_llm_client_retries_transient_http_error_with_timeout(monkeypatch):
    module = importlib.import_module("claimscope.llm")
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "retry succeeded"}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                {},
                io.BytesIO(b'{"error":"raw provider body"}'),
            )
        return FakeResponse()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    client = module.OpenAICompatibleClient(
        api_key="test-secret", base_url="https://llm.example/v1", model="fake-model"
    )

    result = client.chat(
        [{"role": "user", "content": "private research prompt"}], timeout=7
    )

    assert result == "retry succeeded"
    assert [timeout for _, timeout in calls] == [7, 7]


def test_llm_client_exhausted_retry_error_is_sanitized(monkeypatch):
    module = importlib.import_module("claimscope.llm")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "server failed",
            {},
            io.BytesIO(
                b"raw provider body with test-secret and private research prompt"
            ),
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    client = module.OpenAICompatibleClient(
        api_key="test-secret", base_url="https://llm.example/v1", model="fake-model"
    )

    with pytest.raises(Exception) as exc_info:
        client.chat(
            [{"role": "user", "content": "private research prompt"}], timeout=3
        )

    message = f"{exc_info.value!r} {exc_info.value}"
    assert len(calls) == 2
    assert "HTTP 500" in message
    assert "test-secret" not in message
    assert "private research prompt" not in message
    assert "raw provider body" not in message

import importlib
import json

import pytest


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response

    def chat(self, messages, temperature=0.2):
        return self.response


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
        "use an error map loop for segmentation With expert prompts"
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.conditions == ["expert prompts"]


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
            '{"normalized_claim": "Error maps guide promptable medical segmentation refinement"}'
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

import json
import subprocess
import sys
from pathlib import Path

import pytest

from claimscope.core_claim import HeuristicCoreClaimEngine
from claimscope.models import ClaimCandidate, CoreClaimResult


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "benchmarks" / "claimbench.jsonl"


def _load_benchmark_module():
    try:
        import claimscope.benchmark as benchmark
    except ModuleNotFoundError as exc:
        pytest.fail(str(exc))
    return benchmark


def make_result(claim: str) -> CoreClaimResult:
    candidate = ClaimCandidate(
        claim=claim,
        method_or_mechanism=claim,
        target_or_task="benchmark target",
        expected_effect=claim,
        falsification_test="Compare the claim against a baseline on held-out cases.",
        proposer="test",
    )
    return CoreClaimResult(
        direction="test direction",
        selected_claim=claim,
        selected_candidate=candidate,
        candidates=[candidate],
    )


def test_specific_falsifiable_claim_outscores_vague_direction():
    benchmark = _load_benchmark_module()

    vague = benchmark.score_claim("AI for health", make_result("AI helps healthcare"))
    specific = benchmark.score_claim(
        "AI for health",
        make_result(
            "On held-out hospitals, uncertainty-calibrated triage reduces false "
            "negatives versus the same classifier without calibration"
        ),
    )

    assert specific.total > vague.total
    assert specific.falsifiability > vague.falsifiability
    assert (
        specific.components["measurable_outcomes"]
        > vague.components["measurable_outcomes"]
    )


def test_unsupported_certainty_receives_inspectable_overclaim_penalty():
    benchmark = _load_benchmark_module()

    cautious = benchmark.score_claim(
        "AI diagnosis",
        make_result("AI triage may reduce missed sepsis alerts versus rule-based triage"),
    )
    overclaimed = benchmark.score_claim(
        "AI diagnosis",
        make_result(
            "AI diagnosis always proves sepsis and eliminates all clinician errors"
        ),
    )

    assert overclaimed.overclaim_penalty > cautious.overclaim_penalty
    assert overclaimed.total < cautious.total
    assert "overclaim_penalty" in overclaimed.to_dict()


def test_claimbench_dataset_has_stable_cross_domain_case_shape():
    benchmark = _load_benchmark_module()

    cases = benchmark.load_cases(DATA_PATH)

    assert len(cases) >= 24
    assert {case.domain for case in cases} >= {
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
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.required_concepts for case in cases)
    assert all(case.forbidden_overclaims for case in cases)


def test_benchmark_report_contains_reproducible_case_results():
    benchmark = _load_benchmark_module()

    report = benchmark.run_benchmark(
        benchmark.load_cases(DATA_PATH), HeuristicCoreClaimEngine()
    )
    payload = report.to_dict()

    assert report.case_results
    assert report.summary["case_count"] == len(report.case_results)
    assert report.summary["aggregate_score"] > 0
    assert payload["case_results"][0]["score"]["components"]
    assert json.loads(json.dumps(payload)) == payload


def test_benchmark_cli_writes_json_report(tmp_path):
    output_path = tmp_path / "claimbench-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claimscope.cli",
            "benchmark",
            "--engine",
            "heuristic",
            "--data",
            str(DATA_PATH),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ClaimBench" in completed.stdout
    assert "cases" in completed.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["case_count"] >= 24
    assert payload["summary"]["aggregate_score"] > 0


def test_legacy_cli_still_accepts_research_direction():
    completed = subprocess.run(
        [sys.executable, "-m", "claimscope.cli", "retrieval improves factual QA"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ClaimScope Report" in completed.stdout
    assert "Normalized claim" in completed.stdout

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


def test_case_required_concepts_and_forbidden_overclaims_change_total():
    benchmark = _load_benchmark_module()
    result = make_result(
        "Uncertainty-calibrated triage reduces false negative alerts versus an "
        "uncalibrated classifier on held-out hospitals"
    )
    matching_case = benchmark.BenchmarkCase(
        id="matching",
        direction="AI sepsis triage",
        domain="medical_ai",
        required_concepts=[
            "uncertainty-calibrated triage",
            "false negative alerts",
            "held-out hospitals",
        ],
        forbidden_overclaims=["replaces clinicians"],
    )
    missing_case = benchmark.BenchmarkCase(
        id="missing",
        direction="AI sepsis triage",
        domain="medical_ai",
        required_concepts=[
            "mortality reduction",
            "pediatric emergency departments",
            "prospective trial",
        ],
        forbidden_overclaims=["replaces clinicians"],
    )
    forbidden_case = benchmark.BenchmarkCase(
        id="forbidden",
        direction="AI sepsis triage",
        domain="medical_ai",
        required_concepts=["uncertainty-calibrated triage"],
        forbidden_overclaims=["held-out hospitals"],
    )

    matching = benchmark.score_claim(matching_case.direction, result, case=matching_case)
    missing = benchmark.score_claim(missing_case.direction, result, case=missing_case)
    forbidden = benchmark.score_claim(
        forbidden_case.direction, result, case=forbidden_case
    )

    assert matching.total > missing.total
    assert matching.components["required_concept_coverage"] == 1.0
    assert missing.components["required_concept_coverage"] == 0.0
    assert forbidden.total < matching.total
    assert forbidden.components["forbidden_overclaim_penalty"] > 0
    assert 0 <= forbidden.total <= 100


def test_verbose_cue_stuffing_does_not_outrank_concise_mechanistic_claim():
    benchmark = _load_benchmark_module()
    concise = benchmark.score_claim(
        "AI sepsis triage",
        make_result(
            "Uncertainty-calibrated triage reduces false negative alerts versus "
            "uncalibrated triage on held-out hospitals"
        ),
    )
    stuffed = benchmark.score_claim(
        "AI sepsis triage",
        make_result(
            "AI improves healthcare with accuracy score rate baseline control "
            "versus compared with held-out under when with across during randomized "
            "ablation precision recall latency throughput yield retention completion "
            "for every broad clinical workflow and many important outcomes"
        ),
    )

    assert concise.total > stuffed.total
    assert stuffed.components["cue_stuffing_penalty"] > 0


def test_malformed_jsonl_error_names_line_without_echoing_payload(tmp_path):
    benchmark = _load_benchmark_module()
    data_path = tmp_path / "bad.jsonl"
    data_path.write_text(
        '{"id":"ok","direction":"x","domain":"ai","required_concepts":["x"],'
        '"forbidden_overclaims":["secret"]}\n'
        '{"id":"bad","direction":"contains test-secret",',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        benchmark.load_cases(data_path)

    message = str(exc_info.value)
    assert "line 2" in message
    assert "invalid JSON" in message
    assert "test-secret" not in message
    assert "contains" not in message


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


def test_benchmark_cli_default_dataset_works_outside_repository(tmp_path):
    output_path = tmp_path / "claimbench-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claimscope.cli",
            "benchmark",
            "--engine",
            "heuristic",
            "--output",
            str(output_path),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ClaimBench: 27 cases" in completed.stdout
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"][
        "case_count"
    ] == 27


def test_service_uses_packaged_claimbench_dataset():
    from claimscope.service import ClaimScopeService

    service = ClaimScopeService()

    assert service.benchmark_path.parent.name == "data"
    assert service.benchmark_path.is_file()
    assert service.evaluate_claim_benchmark()["summary"]["case_count"] == 27


def test_repository_and_packaged_claimbench_copies_stay_identical():
    packaged = REPO_ROOT / "claimscope" / "data" / "claimbench.jsonl"

    assert packaged.read_bytes() == DATA_PATH.read_bytes()


@pytest.mark.parametrize(
    "option_builder",
    [
        lambda output_path: ["--output", str(output_path)],
        lambda output_path: [f"--output={output_path}"],
        lambda output_path: ["--engine", "heuristic", "--output", str(output_path)],
        lambda output_path: [
            "--data",
            str(DATA_PATH),
            "--output",
            str(output_path),
        ],
    ],
)
def test_any_explicit_benchmark_only_option_selects_benchmark_command(
    tmp_path, option_builder
):
    output_path = tmp_path / "output-only-claimbench.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claimscope.cli",
            "benchmark",
            *option_builder(output_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ClaimBench" in completed.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["case_count"] >= 24
    assert "case_results" in payload


def test_literal_benchmark_is_preserved_as_legacy_research_direction():
    completed = subprocess.run(
        [sys.executable, "-m", "claimscope.cli", "benchmark"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ClaimScope Report" in completed.stdout
    assert "**Input direction / claim:** benchmark" in completed.stdout


def test_explicit_benchmark_options_still_select_benchmark_command():
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
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ClaimBench" in completed.stdout


@pytest.mark.parametrize(
    "benchmark_args",
    [
        ["--engine=heuristic"],
        [f"--data={DATA_PATH}"],
    ],
)
def test_explicit_equals_style_engine_and_data_options_select_benchmark_command(
    benchmark_args,
):
    completed = subprocess.run(
        [sys.executable, "-m", "claimscope.cli", "benchmark", *benchmark_args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ClaimBench" in completed.stdout


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

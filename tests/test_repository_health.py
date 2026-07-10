from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_open_source_release_surface_is_present_and_truthful():
    required = [
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "CITATION.cff",
        "CHANGELOG.md",
        ".github/workflows/ci.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/pull_request_template.md",
        "examples/core_claim_result.json",
        "examples/claimscope_report.md",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    assert not missing, f"missing release files: {missing}"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "Core Claim Arena",
        "ClaimBench",
        "MCP",
        "30-second",
        "Limitations",
        "Roadmap",
    ):
        assert phrase.lower() in readme.lower()

    example = (ROOT / "examples/claimscope_report.md").read_text(encoding="utf-8")
    assert "synthetic" in example.lower()
    assert "concept" not in example.lower()

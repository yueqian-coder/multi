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


def test_release_metadata_targets_the_real_repository_and_real_ui():
    repository_url = "https://github.com/yueqian-coder/multi"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert repository_url in readme
    assert repository_url in citation
    assert repository_url in pyproject
    assert "github.com/claimscope/claimscope" not in readme + citation
    assert "docs/assets/claimscope-v02-desktop.png" in readme
    assert 'license = "MIT"' in pyproject
    assert "License :: OSI Approved" not in pyproject


def test_release_files_are_complete_and_ci_installs_runtime_test_extras():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    conduct = (ROOT / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    example = (ROOT / "examples/claimscope_report.md").read_text(encoding="utf-8")

    assert "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE" in license_text
    assert "Enforcement Responsibilities" in conduct
    assert '.[web,mcp,dev]' in ci
    assert "## Negative Evidence" in example
    assert "## Idea Opportunities" in example
    assert "python -m build --wheel" in ci
    assert "gh[pousr]_" in ci
    assert "github_pat_" in ci
    assert "xox[baprs]-" in ci
    assert "AKIA" in ci
    assert "ClientSession" in ci

# ClaimScope v0.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a release-ready, evidence-first ClaimScope with a real Core Claim multi-agent arena, auditable task-chain events, an offline benchmark, a polished research workbench UI, and an MCP server.

**Architecture:** Typed artifacts in `claimscope.models` form the contract between planners, the new Core Claim Arena, the discovery pipeline, the UI, CLI, benchmark, and MCP transports. LLM agents return bounded JSON artifacts; deterministic scoring and heuristic fallbacks keep tests and no-key demos functional. Every workflow stage emits public trace events while private reasoning and secrets remain outside report objects.

**Tech Stack:** Python 3.10+, standard-library HTTP/concurrency, Streamlit 1.36+, pytest 7.4+, official Python MCP SDK with FastMCP, GitHub Actions.

## Global Constraints

- API credentials are read only from environment variables or an in-memory password field; never write, log, cache, export, or commit them.
- Public traces contain role, status, duration, artifacts, scores, and concise reasons; never expose hidden chain-of-thought.
- The demo works without an API key and labels fixture papers as synthetic.
- `unknown` and `unsupported` remain distinct evidence states.
- A retrieval miss is not proof that evidence does not exist.
- Importing the MCP module must not start the server.
- All user-facing flows must work at 1280x720 desktop and 390x844 mobile sizes without horizontal clipping.

---

### Task 1: Core Claim Typed Contracts And Deterministic Baseline

**Files:**
- Modify: `claimscope/models.py`
- Create: `claimscope/core_claim.py`
- Modify: `claimscope/planner.py`
- Create: `tests/test_core_claim.py`

**Interfaces:**
- Produces: `ClaimCandidate`, `ClaimCritique`, `AgentEvent`, `CoreClaimResult` dataclasses.
- Produces: `HeuristicCoreClaimEngine.run(direction: str) -> CoreClaimResult`.
- Consumes: existing `normalize_claim(query: str) -> str`.

- [ ] **Step 1: Write failing contract and baseline tests**

```python
def test_heuristic_core_claim_reports_missing_slots_and_trace():
    result = HeuristicCoreClaimEngine().run("use an error map loop for segmentation")
    assert result.selected_claim
    assert result.candidates
    assert "measurable expected effect" in result.unresolved_ambiguities
    assert result.events[-1].status == "complete"
    assert result.mode == "heuristic"

def test_core_claim_result_serializes_without_private_reasoning():
    payload = result.to_dict()
    assert "chain_of_thought" not in json.dumps(payload).lower()
    assert payload["selected_candidate"]["falsification_test"]
```

- [ ] **Step 2: Run the focused tests**

Run: `python -m pytest tests/test_core_claim.py -q`

Expected: FAIL because the new contracts and engine do not exist.

- [ ] **Step 3: Implement frozen dataclasses and deterministic scoring**

```python
@dataclass(frozen=True)
class ClaimCandidate:
    claim: str
    method_or_mechanism: str = ""
    target_or_task: str = ""
    expected_effect: str = ""
    conditions: list[str] = field(default_factory=list)
    falsification_test: str = "Compare the stated outcome against a baseline under the stated condition."
    missing_information: list[str] = field(default_factory=list)
    confidence: float = 0.0
    proposer: str = "heuristic"

    def score(self) -> float:
        covered = sum(bool(value) for value in (
            self.method_or_mechanism,
            self.target_or_task,
            self.expected_effect,
            self.falsification_test,
        ))
        return round(covered / 4 * 100 - len(self.missing_information) * 5, 1)
```

Implement `to_dict()` on report artifacts with `dataclasses.asdict`, clamp confidence to `[0, 1]`, and make the heuristic engine return one honest baseline candidate plus a complete trace event.

- [ ] **Step 4: Make the planner adapter use the typed result**

`HeuristicClaimPlanner.extract_core_claim()` calls `HeuristicCoreClaimEngine().run(query).selected_claim`. Add `extract_core_claim_result()` to both planner implementations while preserving the existing string-returning API.

- [ ] **Step 5: Run focused and regression tests**

Run: `python -m pytest tests/test_core_claim.py tests/test_pipeline.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add claimscope/models.py claimscope/core_claim.py claimscope/planner.py tests/test_core_claim.py
git commit -m "Add typed core claim baseline"
```

### Task 2: Multi-Agent Core Claim Arena

**Files:**
- Modify: `claimscope/core_claim.py`
- Modify: `claimscope/llm.py`
- Modify: `claimscope/planner.py`
- Modify: `tests/test_core_claim.py`

**Interfaces:**
- Consumes: `llm_client.chat(messages, temperature, timeout=None) -> str`.
- Produces: `CoreClaimArena(llm_client, max_workers=3).run(direction) -> CoreClaimResult`.
- Produces: `LLMClaimPlanner.extract_core_claim_result(query) -> CoreClaimResult`.

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_arena_runs_three_proposers_two_critics_and_judge():
    client = RoleAwareFakeLLM()
    result = CoreClaimArena(client).run("Can retrieval make medical QA safer?")
    assert {candidate.proposer for candidate in result.candidates} == {
        "operationalizer", "mechanism_analyst", "skeptical_empiricist"
    }
    assert {item.critic for item in result.critiques} == {
        "falsifiability_critic", "scope_critic"
    }
    assert result.mode == "multi_agent"
    assert result.selected_claim == client.expected_judgment
    assert len(result.events) == 6

def test_arena_falls_back_to_weighted_candidate_when_judge_fails():
    result = CoreClaimArena(JudgeFailingFakeLLM()).run("...")
    assert result.degraded is True
    assert result.selected_candidate == max(result.candidates, key=lambda item: item.score())
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_core_claim.py -q`

Expected: FAIL because `CoreClaimArena` is missing.

- [ ] **Step 3: Implement bounded proposer concurrency and JSON parsing**

Use `ThreadPoolExecutor(max_workers=min(3, max_workers))`. Each proposer receives a role-specific public rubric and the same exact JSON schema. Parse the first complete JSON object, reject missing `claim`, clamp numeric values, and emit one event per proposer.

```python
PROPOSER_ROLES = {
    "operationalizer": "Turn the direction into measurable variables and a controlled comparison.",
    "mechanism_analyst": "State the mechanism, target, expected effect, and boundary conditions.",
    "skeptical_empiricist": "Write the narrowest claim that an adverse result could falsify.",
}
```

- [ ] **Step 4: Implement batch critics and judge**

Each critic receives all candidates in one call and returns `candidate_id`, six rubric scores from 0-5, reason codes, and one revision. The judge receives only structured candidates and critiques. It returns `selected_candidate_id`, `final_claim`, `falsification_test`, and `unresolved_ambiguities`.

- [ ] **Step 5: Harden the LLM client**

Add a configurable timeout and one retry for HTTP 429/5xx while preserving sanitized exceptions. Do not include authorization headers, raw response bodies, or user prompts in exception messages or repr output.

- [ ] **Step 6: Run tests and compile check**

Run: `python -m pytest tests/test_core_claim.py tests/test_pipeline.py -q`

Run: `python -m compileall claimscope`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add claimscope/core_claim.py claimscope/llm.py claimscope/planner.py tests/test_core_claim.py
git commit -m "Build multi-agent core claim arena"
```

### Task 3: Auditable Discovery Task Chain And Evidence Integrity

**Files:**
- Modify: `claimscope/models.py`
- Modify: `claimscope/pipeline.py`
- Modify: `claimscope/retrievers.py`
- Modify: `claimscope/text_utils.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `AnalysisReport.events: list[AgentEvent]` and `AnalysisReport.warnings: list[str]`.
- Produces: `Paper.external_id`, `Paper.is_fixture`, `EvidenceItem.method`.
- Preserves: `ClaimScopePipeline.analyze(query, limit=20) -> AnalysisReport`.

- [ ] **Step 1: Add failing provenance and state tests**

```python
def test_report_exposes_stage_events_and_retrieval_limitations():
    report = build_report()
    assert [event.stage for event in report.events] == [
        "research_direction", "core_claim", "claim_boundaries",
        "hidden_assumptions", "evidence_queries", "evidence_retrieval",
        "evidence_adjudication", "opportunity_synthesis", "quality_review",
    ]
    assert all(event.public_summary for event in report.events)
    assert "abstract-only" in " ".join(report.warnings).lower()

def test_retrieval_miss_stays_unknown_not_unsupported():
    report = ClaimScopePipeline(StaticPaperRetriever([])).analyze("...")
    assert all(item.status == "unknown" for item in report.assumptions)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: FAIL on missing events and provenance fields.

- [ ] **Step 3: Repair multilingual markers and normalize evidence labels**

Replace corrupted marker text with valid UTF-8 Chinese markers and add English scientific null-result phrases. Keep the classifier conservative: neutral sentences remain `mention` and are not attached as evidence.

- [ ] **Step 4: Emit explicit stage events and warnings**

Wrap each pipeline stage with a small timer helper that emits `complete`, `degraded`, `failed`, or `skipped`. Retriever exceptions become sanitized warnings naming the source, while combined retrieval continues.

- [ ] **Step 5: Improve paper identity and fixture labeling**

Parse arXiv IDs and Semantic Scholar paper IDs where available. Deduplicate by external ID before normalized title. Mark bundled demo records with `is_fixture=True` and render them as synthetic.

- [ ] **Step 6: Add quality review rules**

Return warnings for zero papers, zero direct evidence, all-unknown assumptions, abstract-only evidence, and heuristic fallback. Never convert an unknown assumption to unsupported solely due to no retrieval hit.

- [ ] **Step 7: Run regression tests**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add claimscope/models.py claimscope/pipeline.py claimscope/retrievers.py claimscope/text_utils.py tests/test_pipeline.py
git commit -m "Add auditable discovery task chain"
```

### Task 4: ClaimBench Evaluation Loop

**Files:**
- Create: `claimscope/benchmark.py`
- Create: `benchmarks/claimbench.jsonl`
- Create: `tests/test_benchmark.py`
- Modify: `claimscope/cli.py`

**Interfaces:**
- Produces: `score_claim(direction: str, result: CoreClaimResult) -> ClaimScore`.
- Produces: `run_benchmark(cases, engine) -> BenchmarkReport`.
- Produces CLI: `claimscope benchmark --engine heuristic --data benchmarks/claimbench.jsonl`.

- [ ] **Step 1: Add failing rubric tests**

```python
def test_specific_falsifiable_claim_outscores_vague_direction():
    vague = score_claim("AI for health", make_result("AI helps healthcare"))
    specific = score_claim(
        "AI for health",
        make_result("On held-out hospitals, uncertainty-calibrated triage reduces false negatives versus the same classifier without calibration"),
    )
    assert specific.total > vague.total
    assert specific.falsifiability > vague.falsifiability

def test_benchmark_report_contains_reproducible_case_results():
    report = run_benchmark(load_cases(path), HeuristicCoreClaimEngine())
    assert report.case_results
    assert report.summary["case_count"] == len(report.case_results)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_benchmark.py -q`

Expected: FAIL because benchmark APIs are missing.

- [ ] **Step 3: Implement deterministic rubric**

Score slot coverage, declarative form, comparison language, measurable outcomes, falsification test quality, specificity, and overclaim penalties. Store component scores so users can inspect why a result scored as it did.

- [ ] **Step 4: Add cross-domain seed dataset**

Add at least 24 cases across AI, medical AI, biology, social science, climate, materials, HCI, education, and systems. Each JSONL row contains `id`, `direction`, `domain`, `required_concepts`, and `forbidden_overclaims`.

- [ ] **Step 5: Add CLI benchmark command and JSON output**

The command prints a compact summary and optionally writes JSON with `--output`. Live LLM benchmark mode is opt-in; no-key CI runs only the heuristic engine.

- [ ] **Step 6: Run tests and a benchmark smoke test**

Run: `python -m pytest tests/test_benchmark.py -q`

Run: `python -m claimscope.cli benchmark --engine heuristic --data benchmarks/claimbench.jsonl`

Expected: 24+ cases complete with a nonzero aggregate score.

- [ ] **Step 7: Commit**

```bash
git add claimscope/benchmark.py benchmarks/claimbench.jsonl tests/test_benchmark.py claimscope/cli.py
git commit -m "Add ClaimBench evaluation loop"
```

### Task 5: MCP Server And Package Entry Points

**Files:**
- Create: `claimscope/service.py`
- Create: `claimscope/mcp_server.py`
- Create: `tests/test_mcp_server.py`
- Create: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `.env.example`

**Interfaces:**
- Produces: `ClaimScopeService.extract_core_claim(direction, mode="auto") -> dict`.
- Produces: `ClaimScopeService.analyze_research_direction(direction, online=False, limit=12) -> dict`.
- Produces: FastMCP tools `extract_core_claim`, `analyze_research_direction`, `build_evidence_queries`, `evaluate_claim_benchmark`, and `get_demo_report`.
- Produces console scripts: `claimscope`, `claimscope-mcp`.

- [ ] **Step 1: Confirm the current official FastMCP API**

Use the official Model Context Protocol Python SDK documentation. Pin a compatible lower bound in `pyproject.toml`; do not depend on an undocumented import path.

- [ ] **Step 2: Write failing service and MCP tests**

```python
def test_mcp_module_import_has_no_server_side_effect():
    module = importlib.import_module("claimscope.mcp_server")
    assert module.mcp.name == "ClaimScope"

def test_extract_core_claim_tool_returns_json_compatible_trace():
    payload = extract_core_claim_payload("Can retrieval make medical QA safer?", mode="heuristic")
    json.dumps(payload)
    assert payload["selected_claim"]
    assert payload["events"]
```

- [ ] **Step 3: Verify failure**

Run: `python -m pytest tests/test_mcp_server.py -q`

Expected: FAIL because service and MCP modules do not exist.

- [ ] **Step 4: Implement the service boundary**

Move retriever and planner construction out of Streamlit and CLI into `ClaimScopeService`. Keep return values JSON-compatible and provide dependency injection for tests.

- [ ] **Step 5: Implement FastMCP tools**

```python
mcp = FastMCP("ClaimScope")

@mcp.tool()
def extract_core_claim(direction: str, mode: str = "auto") -> dict:
    """Turn a fuzzy research direction into an auditable falsifiable claim."""
    return build_service().extract_core_claim(direction, mode=mode)

def main() -> None:
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add packaging and MCP configuration example**

Define optional dependency groups `web`, `mcp`, and `dev`; console scripts point to `claimscope.cli:main` and `claimscope.mcp_server:main`. Add a README configuration example using `python -m claimscope.mcp_server` and environment variables with placeholders only.

- [ ] **Step 7: Install and test**

Run: `python -m pip install -e ".[mcp,dev]"`

Run: `python -m pytest tests/test_mcp_server.py -q`

Run: `python -m claimscope.mcp_server --help`

Expected: imports and tests pass; no server starts during import.

- [ ] **Step 8: Commit**

```bash
git add claimscope/service.py claimscope/mcp_server.py tests/test_mcp_server.py pyproject.toml requirements.txt .env.example
git commit -m "Expose ClaimScope through MCP"
```

### Task 6: Research Workbench UI

**Files:**
- Create: `claimscope/ui_components.py`
- Modify: `app.py`
- Modify: `.streamlit/config.toml`
- Create: `tests/test_ui_contract.py`

**Interfaces:**
- Consumes: `ClaimScopeService` JSON-compatible results.
- Produces: Core Claim Test and Full Discovery modes with persistent session results.
- Produces: downloadable feedback JSONL and report Markdown.

- [ ] **Step 1: Write failing source-level UI contract tests**

```python
def test_ui_has_agent_arena_and_no_internal_thought_copy():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "Core Claim Arena" in source
    assert "Agent activity" in source
    assert "chain of thought" not in source.lower()

def test_ui_uses_session_state_for_results():
    source = Path("app.py").read_text(encoding="utf-8")
    assert 'st.session_state["core_claim_result"]' in source
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_ui_contract.py -q`

Expected: FAIL on missing arena and session-state contracts.

- [ ] **Step 3: Split rendering helpers from app orchestration**

Create focused renderers for top bar, mode switcher, task stepper, candidate comparison, rubric meters, selected claim, agent activity, assumption ledger, evidence drawer, warnings, and feedback export.

- [ ] **Step 4: Rebuild the Core Claim Test flow**

Use a compact configuration drawer instead of a permanently wide sidebar. Keep input, primary action, selected claim, confidence, missing information, and candidates within the first meaningful viewport. Disable retrieval controls in Core Claim mode.

- [ ] **Step 5: Rebuild Full Discovery views**

Use tabs for Overview, Assumption Ledger, Evidence, Opportunities, Trace, and Export. Evidence cards show stance, provenance method, paper metadata, and abstract-only warnings.

- [ ] **Step 6: Add in-memory provider configuration and feedback**

The password field may initialize a session-only client. Display the base URL host and model, never the key. Feedback produces downloadable JSONL containing input, observed claim, expected claim, rating, and notes.

- [ ] **Step 7: Run tests and browser QA**

Run: `python -m pytest tests/test_ui_contract.py -q`

Open the app in the in-app browser. Verify Core Claim and Full Discovery at 1280x720 and 390x844, capture screenshots, confirm no horizontal overflow, and inspect console errors.

- [ ] **Step 8: Commit**

```bash
git add claimscope/ui_components.py app.py .streamlit/config.toml tests/test_ui_contract.py
git commit -m "Redesign ClaimScope research workbench"
```

### Task 7: Open-Source Release Surface

**Files:**
- Modify: `README.md`
- Create: `LICENSE`
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `CITATION.cff`
- Create: `CHANGELOG.md`
- Create: `.github/workflows/ci.yml`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/pull_request_template.md`
- Create: `examples/core_claim_result.json`
- Create: `examples/claimscope_report.md`

**Interfaces:**
- Produces: a 30-second quick start, truthful feature matrix, benchmark section, MCP setup, architecture, limitations, roadmap, contribution path, and release metadata.

- [ ] **Step 1: Add failing repository health test**

```python
def test_release_files_exist_and_readme_links_resolve():
    required = ["LICENSE", "CONTRIBUTING.md", "SECURITY.md", "CITATION.cff"]
    assert all(Path(item).exists() for item in required)
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Core Claim Arena" in readme
    assert "ClaimBench" in readme
    assert "MCP" in readme
```

- [ ] **Step 2: Create release metadata and community files**

Use MIT licensing, Contributor Covenant 2.1, private vulnerability reporting guidance, citation metadata for ClaimScope software, and issue templates with reproducible-environment fields.

- [ ] **Step 3: Rewrite README around the differentiator**

Lead with one sentence, one screenshot, and one runnable command. Follow with why it is different, workflow, sample output, quick starts for web/CLI/Python/MCP, benchmark, architecture, limitations, roadmap, and contribution links. Clearly label demo data as synthetic.

- [ ] **Step 4: Add CI and secret scan**

Test Python 3.10, 3.11, and 3.12. Run pytest, compileall, CLI benchmark smoke, and a repository regex scan for key-like values and the private gateway hostname.

- [ ] **Step 5: Generate truthful example artifacts from the code**

Run the deterministic demo and save only generated JSON/Markdown with synthetic-paper labels. Do not handwrite claims that the current code cannot produce.

- [ ] **Step 6: Run repository health tests**

Run: `python -m pytest -q`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add README.md LICENSE CONTRIBUTING.md SECURITY.md CODE_OF_CONDUCT.md CITATION.cff CHANGELOG.md .github examples tests
git commit -m "Prepare ClaimScope v0.2 open source release"
```

### Task 8: Final Adversarial Review And Release Verification

**Files:**
- Modify only files implicated by review findings.

**Interfaces:**
- Consumes every prior task deliverable.
- Produces a clean worktree, passing verification evidence, release notes, and a runnable local server.

- [ ] **Step 1: Run independent adversarial reviews**

Dispatch focused reviewers for scientific validity, Python/API correctness, MCP interoperability, security/privacy, UI/accessibility, and open-source onboarding. Require file/line evidence for every finding.

- [ ] **Step 2: Triage findings by severity**

Fix all correctness, secret exposure, misleading scientific claim, broken install, broken MCP, and visible layout findings. Record lower-priority enhancements in the roadmap only when they are genuinely non-blocking.

- [ ] **Step 3: Run the complete verification matrix**

Run: `python -m pytest -q`

Run: `python -m compileall claimscope app.py`

Run: `python -m claimscope.cli benchmark --engine heuristic --data benchmarks/claimbench.jsonl`

Run: `python -m claimscope.cli "RAG can reliably reduce hallucination in LLM-generated answers"`

Run: `python -c "import claimscope.mcp_server; print(claimscope.mcp_server.mcp.name)"`

Run: `rg -n "sk-[A-Za-z0-9_-]{32,}" . -g '!outputs/**'`

Expected: tests and smoke commands pass; secret scan returns no matches.

- [ ] **Step 4: Perform final browser verification**

Run Streamlit on an available port. Capture accepted desktop and mobile screenshots for the README. Verify both modes, the API-not-configured state, feedback download, Markdown export, and no console errors.

- [ ] **Step 5: Commit review fixes and report release status**

```bash
git add <reviewed-files>
git commit -m "Harden ClaimScope v0.2 release"
```

Report exact test counts, benchmark case count, MCP tool count, the local URL, residual limitations, and the final commit SHA.

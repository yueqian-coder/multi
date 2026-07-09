# ClaimScope v0.2 Design

## Product Decision

ClaimScope v0.2 is an evidence-first, assumption-centric research workbench for
the stage before idea generation. It is not a general research operating system
and does not claim to automate science. Its primary promise is:

> Turn a fuzzy research direction into a falsifiable claim, expose what must be
> true for that claim to hold, and preserve an auditable trail from agent
> judgments to evidence and opportunity signals.

The first memorable workflow is **Core Claim Arena**. Later stages use the same
auditable contract, so every module can be tested independently or run as a
complete chain.

## Success Criteria

1. A new user can run a useful demo without an API key in under two minutes.
2. With an OpenAI-compatible endpoint configured, Core Claim Arena uses multiple
   independent agent calls and returns a structured, falsifiable claim rather
   than a cosmetic rewrite.
3. The UI exposes agent roles, statuses, scores, artifacts, and concise reasons,
   but never exposes private chain-of-thought.
4. Full Discovery maps assumptions to traceable evidence, distinguishes
   support, contradiction, limitations, and null results, and communicates
   uncertainty honestly.
5. A public benchmark compares heuristic, single-agent, and multi-agent claim
   extraction using deterministic scoring plus optional human review.
6. The same workflow is available through the web app, Python API, CLI, and an
   MCP server.
7. The repository presents a credible open-source project: license, package
   metadata, CI, security guidance, contribution workflow, screenshots, sample
   outputs, and a release-ready README.

## Architecture

```text
Web / CLI / Python / MCP
          |
          v
  ClaimScope Service Layer
          |
          +--> Core Claim Arena
          |      +--> Claim Proposer A
          |      +--> Claim Proposer B
          |      +--> Claim Proposer C
          |      +--> Falsifiability Critic
          |      +--> Scope Critic
          |      `--> Judge
          |
          +--> Assumption Planner
          +--> Evidence Query Planner
          +--> Academic Retrievers
          +--> Evidence Adjudicator
          `--> Opportunity Synthesizer
                 |
                 v
       Structured Analysis Report
```

The orchestration layer owns retries, timeouts, bounded parallelism, fallback,
and trace events. Agents return typed JSON artifacts. UI and transport layers
consume those artifacts and do not depend on prompt internals.

## Core Claim Arena

### Input

- Free-form research direction in English or Chinese.
- Optional domain lens.
- Optional constraints supplied by the researcher.

### Candidate Contract

Each proposer returns:

- `claim`: one declarative, testable statement.
- `method_or_mechanism`.
- `target_or_task`.
- `expected_effect`.
- `conditions`.
- `falsification_test`.
- `missing_information`.
- `confidence` from 0 to 1.

The heuristic proposer creates a safe baseline and flags missing fields. In LLM
mode, three independent proposer calls use different roles: operationalizer,
mechanism analyst, and skeptical empiricist.

### Critic Contract

Critics score each candidate on:

- specificity;
- falsifiability;
- scope discipline;
- mechanism clarity;
- measurable outcome;
- unsupported certainty.

Critics return short reason codes and revision suggestions, not hidden reasoning.

### Judge Contract

The judge receives candidates and critic artifacts, then selects or revises one
claim. If the judge fails, a deterministic weighted scorer selects the strongest
candidate. The output includes the final claim, runner-up candidates, aggregate
scores, unresolved ambiguities, and the complete structured trace.

## Full Task Chain

The complete analysis is a state machine with independently runnable stages:

1. `research_direction`: validate and normalize input.
2. `core_claim`: run Core Claim Arena.
3. `claim_boundaries`: produce narrower, conditional, historical, and failure
   variants.
4. `hidden_assumptions`: decompose causal, data, measurement, implementation,
   and generalization assumptions.
5. `evidence_queries`: generate support, contradiction, limitation, and
   null-result searches per assumption.
6. `evidence_retrieval`: retrieve and deduplicate papers with stable identifiers.
7. `evidence_adjudication`: classify evidence with an LLM when available and a
   conservative deterministic fallback otherwise.
8. `opportunity_synthesis`: rank only gaps that are traceable to assumptions or
   negative evidence.
9. `quality_review`: run consistency, citation, coverage, and overclaim checks.

Every stage emits `TaskEvent` records with role, state, duration, artifact count,
and a concise public summary. Failed stages are explicit; downstream stages may
degrade but may not silently pretend completion.

## Evidence Integrity

- Preserve paper URL, source, authors, year, external ID, and abstract provenance.
- Deduplicate by DOI, arXiv ID, Semantic Scholar ID, or normalized title.
- Never call an absence of retrieved evidence proof of absence.
- Separate `unknown` from `unsupported`.
- Record whether a stance came from the LLM adjudicator or deterministic fallback.
- Show retrieval warnings, source failures, and abstract-only limitations.
- Demo fixtures are clearly labeled synthetic and never presented as real papers.

## API And Secret Handling

`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `MODEL_NAME` remain environment-based.
The web app may also accept a password-type key for the current Streamlit session;
it is held in memory only and is never logged, exported, cached, or written to
disk. Agent configuration includes model, timeout, retry count, concurrency,
temperature, and maximum estimated calls.

Provider failures produce sanitized diagnostics. Raw responses are never logged
when they may contain user content. Agent traces contain structured artifacts,
not credentials or chain-of-thought.

## MCP Surface

The stdio MCP server exposes focused tools:

- `extract_core_claim`
- `analyze_research_direction`
- `build_evidence_queries`
- `evaluate_claim_benchmark`
- `get_demo_report`

Tool inputs and outputs use JSON-compatible typed schemas. Importing the MCP
module must work without starting a server, enabling ordinary unit tests. The
server is launched with `claimscope-mcp` or `python -m claimscope.mcp_server`.

## Web Experience

The UI is a research workbench rather than a marketing page.

- Compact top bar with product name, mode, provider health, and export action.
- Main input panel with examples and one primary action.
- A horizontal task-chain stepper that shows running, complete, degraded, and
  skipped states.
- Core Claim Arena displays candidate comparison, rubric scores, selected claim,
  missing information, and agent activity.
- Full Discovery uses an assumption ledger with filters and evidence drawers.
- Warnings and provenance stay near the claims they qualify.
- Results persist in Streamlit session state while users change tabs.
- Desktop and mobile layouts avoid horizontal clipping and oversized sidebars.

The visual language is restrained and research-oriented: white and cool-gray
surfaces, teal for primary actions, indigo for agent activity, amber for
uncertainty, and red only for contradiction or failure. Cards are used only for
individual artifacts, not as nested page sections.

## Evaluation Loop

`ClaimBench` contains cross-domain fuzzy directions and reference criteria. It
scores:

- slot coverage;
- declarative form;
- falsifiability cues;
- specificity;
- overclaim penalties;
- stability across repeated runs.

The benchmark supports three systems: heuristic, single-agent, and multi-agent.
Offline fixtures make CI deterministic. Live evaluation is opt-in and reports
model, endpoint class, latency, and estimated call count without exposing keys.
The UI includes a feedback form for Pass, Partial, or Fail plus an expected claim;
feedback can be downloaded as JSONL for future benchmark curation.

The implementation loop is complete only when unit tests, integration tests,
benchmark smoke tests, secret scans, CLI smoke tests, MCP import/tool tests, and
desktop/mobile UI checks pass.

## Open-Source Release Surface

The repository will include:

- `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and
  `CITATION.cff`;
- `pyproject.toml` with console scripts and optional web/MCP dependencies;
- GitHub Actions for tests and secret-pattern checks;
- an improved README with a concise differentiator, workflow visual, screenshots,
  demo commands, architecture, benchmark, roadmap, and limitations;
- sample JSON and Markdown reports;
- issue templates and a pull request template;
- a changelog and release checklist.

## Out Of Scope For v0.2

- Autonomous experiment execution.
- Full PDF acquisition behind paywalls.
- Claims of exhaustive literature coverage.
- Replacing peer review or expert judgment.
- A general-purpose chat assistant.

These exclusions keep the contribution recognizable and the quality measurable.

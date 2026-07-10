# Task 3 Report: Auditable Discovery Task Chain And Evidence Integrity

## Summary

Implemented an auditable downstream discovery chain with ordered public stage events, explicit quality warnings, paper/evidence provenance fields, stable retrieval deduplication, conservative stance handling, and clean UTF-8 multilingual/null-result markers.

## TDD Evidence

- Added RED tests in `tests/test_pipeline.py` for the nine ordered stage events, retrieval limitations, unknown retrieval misses, paper/evidence provenance defaults, external-ID deduplication, sanitized combined-retriever failures, and multilingual/null-result stance classification.
- Verified RED with `python -m pytest tests/test_pipeline.py -q`: 6 failures for missing `events`, `warnings`, `external_id`, external-ID dedupe, sanitized warnings, and multilingual evidence behavior.
- Implemented the minimal production changes to satisfy those tests.
- Verified GREEN with `python -m pytest tests/test_pipeline.py -q`: 23 passed.

## Implementation Notes

- `AnalysisReport` now exposes `events: list[AgentEvent]` and `warnings: list[str]` while preserving default construction compatibility.
- `ClaimScopePipeline.analyze(query, limit=20)` emits nine ordered events:
  `research_direction`, `core_claim`, `claim_boundaries`, `hidden_assumptions`, `evidence_queries`, `evidence_retrieval`, `evidence_adjudication`, `opportunity_synthesis`, `quality_review`.
- Event summaries are public and count-oriented; they do not include hidden reasoning, exception text, API keys, or raw secrets.
- Retriever failures are converted into sanitized warnings that name the failing retriever class while continuing retrieval.
- `CombinedRetriever` records source-level warnings and deduplicates by external ID before normalized title.
- `Paper` now carries `external_id` and `is_fixture`; arXiv IDs and Semantic Scholar IDs are parsed when available, with URL fallback identity for other URL-backed papers.
- Fixture/demo papers are labeled as synthetic fixtures in markdown output.
- `EvidenceItem` now carries `method="abstract_heuristic"`.
- Retrieval misses leave assumptions as `unknown`; no-hit assumptions are not converted to `unsupported`.
- Quality review warnings cover zero papers, zero direct evidence, all-unknown assumptions, abstract-only evidence, and heuristic fallback.
- UTF-8 Chinese support/negative/null markers and English scientific null-result phrases were added. Neutral mentions remain excluded from evidence.
- `split_sentences` now handles Chinese sentence punctuation before the legacy fallback expression.

## Verification

- `python -m pytest tests/test_pipeline.py -q` -> 23 passed.
- `python -m pytest -q` -> 44 passed.
- `python -m compileall claimscope tests` -> exited 0.
- `git diff --check` -> exited 0; Git reported line-ending normalization warnings only.

## Self-Review

- Scope stayed within the requested implementation files plus this required report.
- Existing report APIs remain intact; new fields have defaults.
- Warnings are explicit and conservative rather than silently swallowing failures.
- Remaining limitation: evidence remains abstract-only and heuristic, which is now surfaced as a quality warning rather than hidden.

## Task 3 Review Fix Evidence

### RED

- Added regression tests in `tests/test_pipeline.py` for:
  - planner exceptions producing exactly one sanitized `failed` `core_claim` event while continuing with a heuristic plan;
  - downstream adjudication helper exceptions producing exactly one sanitized `failed` `evidence_adjudication` event while continuing with typed fallbacks;
  - normal no-key `HeuristicClaimPlanner` not warning about heuristic fallback;
  - actual LLM planner fallback still warning about heuristic fallback;
  - mixed Chinese/English sentence punctuation splitting in one pass.
- Verified RED with `python -m pytest tests/test_pipeline.py -q`:
  - `test_planner_failure_emits_sanitized_failed_event_and_continues` failed because `RuntimeError: secret-token-123 planner private chain-of-thought leaked` escaped from `self.planner.build(query)`.
  - `test_failed_adjudication_stage_uses_typed_fallbacks_and_continues` failed because `RuntimeError: secret patient identifier from private notes` escaped from `_build_assumptions`.
  - `test_split_sentences_handles_mixed_chinese_and_english_punctuation` failed because the English sentence and following Chinese sentence stayed joined.
  - `test_normal_heuristic_planner_does_not_warn_about_fallback` failed because a plain heuristic planner emitted a heuristic fallback warning.
  - Result: `4 failed, 24 passed`.

### GREEN

- Implemented sanitized per-stage exception handling in `ClaimScopePipeline.analyze`.
  - The nine logical stages still emit in fixed order, exactly once each.
  - Planner exceptions recover through `HeuristicClaimPlanner` and mark `core_claim` as `failed`.
  - Stage-helper exceptions use conservative typed fallbacks and mark the owning stage as `failed`.
  - Public summaries and warnings do not include raw exception messages, secrets, or private reasoning text.
- Changed heuristic fallback warnings so a normal `HeuristicClaimPlanner` is not treated as fallback; LLM fallback and explicit planner recovery still warn.
- Replaced `split_sentences()` with one mixed punctuation regex using escaped Chinese punctuation code points.
- Verified focused GREEN with `python -m pytest tests/test_pipeline.py -q`: `28 passed in 0.20s`.
- Verified full suite with `python -m pytest -q`: `49 passed in 0.31s`.
- Verified bytecode compilation with `python -m compileall claimscope tests`: exited 0 after listing `claimscope` and `tests`.
- Verified whitespace with `git diff --check`: exited 0; Git reported line-ending normalization warnings only.

### Self-Review

- Scope stayed within `claimscope/pipeline.py`, `claimscope/text_utils.py`, `tests/test_pipeline.py`, and this report.
- The new warnings are intentionally generic and stage-named; they preserve public auditability without leaking exception payloads.
- `_targeted_search` now returns a failure flag so retriever exceptions can make the retrieval stage `failed` instead of merely degraded.
- No model/planner API changes were needed.

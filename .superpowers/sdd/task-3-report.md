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

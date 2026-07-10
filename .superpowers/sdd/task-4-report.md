# Task 4 Report: ClaimBench Evaluation Loop

## Status

Complete.

Commit: `6139c25` (`Add ClaimBench evaluation loop`)

## TDD Evidence

- Added `tests/test_benchmark.py` first.
- Verified RED with `python -m pytest tests/test_benchmark.py -q`.
- Initial failures were caused by the missing `claimscope.benchmark` module and missing `claimscope benchmark` CLI command.
- Implemented the deterministic benchmark module, dataset, and CLI command after the red run.

## Implemented

- `claimscope/benchmark.py`
  - `BenchmarkCase`, `ClaimScore`, `BenchmarkCaseResult`, and `BenchmarkReport`.
  - `load_cases(path)`.
  - `score_claim(direction, result)`.
  - `run_benchmark(cases, engine)`.
  - Inspectable component scores for slot coverage, declarative form, comparison language, measurable outcomes, falsifiability, specificity, and overclaim penalty.
- `benchmarks/claimbench.jsonl`
  - 27 deterministic seed cases.
  - Covers AI, medical AI, biology, social science, climate, materials, HCI, education, and systems.
  - Every row has stable `id`, valid `domain`, non-empty `required_concepts`, and non-empty `forbidden_overclaims`.
- `claimscope/cli.py`
  - Added `claimscope benchmark --engine heuristic --data benchmarks/claimbench.jsonl`.
  - Optional `--output` writes reproducible JSON.
  - Preserved the legacy positional research-direction invocation.
  - Heuristic benchmark mode is offline and requires no key or network.

## Verification

- `python -m pytest tests/test_benchmark.py -q`
  - `6 passed`
- `python -m claimscope.cli benchmark --engine heuristic --data benchmarks/claimbench.jsonl`
  - `ClaimBench: 27 cases, aggregate score 70.09 (HeuristicCoreClaimEngine)`
- `python -m claimscope.cli "retrieval improves factual QA"`
  - Produced a normal ClaimScope markdown report.
- `python -m pytest -q`
  - `55 passed`
- `python -m compileall claimscope tests`
  - Passed.

These checks were rerun after commit `6139c25`.

## Self-Review

- Scope stayed limited to the requested owned files plus this required report.
- No live LLM path was added to the benchmark CLI, so offline CI remains keyless and network-free.
- JSON output contains public directions, public claims, scores, and case metadata only.
- The scoring rubric is deterministic and intentionally heuristic; it should be treated as a reproducible baseline rather than a model-quality oracle.

## Concerns

- The heuristic engine currently echoes already-specific benchmark directions. This is appropriate for the offline deterministic seed benchmark, but future live/multi-agent benchmarking should compare engines on the same cases under an explicit opt-in mode.

## Review Fixes

### Requested Follow-Up

- Case `required_concepts` coverage and `forbidden_overclaims` hits now affect case totals through inspectable `required_concept_coverage`, `required_concept_penalty`, and `forbidden_overclaim_penalty` components.
- Literal `claimscope benchmark` now remains a legacy research direction, while `claimscope benchmark --engine heuristic --data benchmarks/claimbench.jsonl` still selects the benchmark command.
- Malformed JSONL decode failures are wrapped in `ValueError` with the source line number and without echoing the raw line payload.
- Verbose cue-word stuffing now receives an inspectable `cue_stuffing_penalty`.

### TDD Evidence

- Added failing regression tests in `tests/test_benchmark.py`.
- RED command: `python -m pytest tests/test_benchmark.py -q`
  - Failed 4 tests for missing `case=` scoring support, missing `cue_stuffing_penalty`, unsanitized JSON decode errors, and incorrect literal `benchmark` CLI dispatch.
- GREEN command: `python -m pytest tests/test_benchmark.py -q`
  - `11 passed`

### Verification After Fixes

- `python -m pytest tests/test_benchmark.py -q`
  - `11 passed`
- `python -m claimscope.cli benchmark --engine heuristic --data benchmarks/claimbench.jsonl`
  - `ClaimBench: 27 cases, aggregate score 70.09 (HeuristicCoreClaimEngine)`
- `python -m claimscope.cli benchmark`
  - Produced a normal ClaimScope markdown report for the literal research direction `benchmark`.
- `python -m pytest -q`
  - `60 passed`
- `python -m compileall claimscope tests`
  - Passed.

### Self-Review

- Totals are still clamped to `[0, 100]`.
- Score penalties are public and inspectable through `ClaimScore.to_dict()` and report JSON.
- The benchmark CLI remains offline and keyless.
- The `.superpowers/` report path is ignored by repo policy unless explicitly force-added.

## Final Dispatch Fix

### Requested Follow-Up

- `_has_explicit_benchmark_option()` now recognizes every benchmark-only option:
  - `--engine`
  - `--engine=value`
  - `--data`
  - `--data=value`
  - `--output`
  - `--output=value`
- Unadorned literal `claimscope benchmark` remains a legacy research-direction query.

### TDD Evidence

- Added focused CLI regression tests for:
  - output-only separated form: `benchmark --output path`
  - output-only equals form: `benchmark --output=path`
  - explicit engine/data separated routes
  - explicit engine/data equals routes
- RED command: `python -m pytest tests/test_benchmark.py -q`
  - `4 failed, 13 passed`
  - Failures showed output-only forms fell through to analysis mode and equals-style engine/data options were not recognized as benchmark dispatch.
- GREEN command: `python -m pytest tests/test_benchmark.py -q`
  - `17 passed in 2.66s`

### Verification

- `python -m pytest tests/test_benchmark.py -q`
  - `17 passed in 2.66s`
- Output-only CLI smoke with report written under `%TEMP%`:
  - `python -m claimscope.cli benchmark --output <temp path>`
  - `ClaimBench: 27 cases, aggregate score 70.09 (HeuristicCoreClaimEngine)`
  - JSON check: `case_count=27 aggregate=70.09`
- Literal legacy CLI smoke:
  - `python -m claimscope.cli benchmark`
  - Produced a normal ClaimScope markdown report for input direction `benchmark`.
- `python -m pytest -q`
  - `66 passed in 3.50s`

### Self-Review

- Dispatch stays conservative: only `benchmark` followed by a known benchmark-only option selects the benchmark command.
- Output-only benchmark mode uses the default heuristic engine and default JSONL data, so it remains keyless and offline.

# Task 1 Report: Core Claim Typed Contracts And Deterministic Baseline

## Scope

Implemented Task 1 inside `C:\Users\13555\Desktop\gkj\nlp\.worktrees\claimscope-v0.2` without touching unrelated project files. The change adds typed core-claim artifacts, a deterministic heuristic baseline engine, planner adapters that preserve the existing string API, and focused tests for the new public contract.

## TDD Record

### Red

Added `tests/test_core_claim.py` first with contract checks for:

- heuristic engine output shape and trace completion
- serialization without private reasoning
- deterministic `ClaimCandidate.score()` behavior and confidence clamping
- planner adapters exposing typed results while preserving `extract_core_claim()`

Focused red run:

```powershell
python -m pytest tests/test_core_claim.py -q
```

Observed failure reasons before implementation:

- `claimscope.core_claim` module did not exist
- `ClaimCandidate` was missing from `claimscope.models`
- `HeuristicClaimPlanner.extract_core_claim_result()` did not exist

### Green

Implemented:

- frozen dataclasses `ClaimCandidate`, `ClaimCritique`, `AgentEvent`, and `CoreClaimResult`
- `to_dict()` on the new public artifacts via `dataclasses.asdict`
- confidence clamping to `[0, 1]`
- deterministic `ClaimCandidate.score()` matching the task brief
- `HeuristicCoreClaimEngine.run(direction: str) -> CoreClaimResult`
- planner adapter methods `extract_core_claim_result()` on both planners while preserving the string-returning API

Focused green run:

```powershell
python -m pytest tests/test_core_claim.py -q
```

Result: `4 passed`

## Implementation Notes

### `claimscope/models.py`

Added the typed public artifacts required by the brief:

- `ClaimCandidate`
- `ClaimCritique`
- `AgentEvent`
- `CoreClaimResult`

`ClaimCandidate` clamps confidence in `__post_init__()`, keeps the exact default falsification test from the brief, and computes deterministic coverage-based scores.

### `claimscope/core_claim.py`

Created the deterministic baseline engine:

- normalizes the incoming direction through the existing `normalize_claim()`
- derives a single honest baseline candidate from lightweight string heuristics
- records missing slots instead of inventing unsupported specificity
- emits one complete `AgentEvent`
- returns a `CoreClaimResult` in `heuristic` mode

The baseline deliberately leaves unresolved ambiguity visible when a direction is underspecified, e.g. missing a measurable expected effect.

### `claimscope/planner.py`

Updated planners to use the typed result contract:

- `HeuristicClaimPlanner.extract_core_claim_result()` now delegates to `HeuristicCoreClaimEngine`
- `HeuristicClaimPlanner.extract_core_claim()` now preserves the old string API by returning `.selected_claim`
- `LLMClaimPlanner.extract_core_claim_result()` now returns a typed result and preserves fallback behavior
- existing `build()` and pipeline-facing behavior remain intact

## Verification

Required focused regression run:

```powershell
python -m pytest tests/test_core_claim.py tests/test_pipeline.py -q
```

Result: `21 passed`

Full suite run:

```powershell
python -m pytest -q
```

Result: `21 passed`

## Self-Review

Checked the diff against the brief and verified:

- only the task-owned source files were changed for implementation
- the new public artifacts expose structured data only and no private reasoning
- the planner string API still works
- the 17 pre-existing pipeline tests remain green
- the deterministic baseline stays conservative instead of fabricating missing slots

No follow-up fixups were needed after the review pass.

## Files Changed

- `claimscope/models.py`
- `claimscope/core_claim.py`
- `claimscope/planner.py`
- `tests/test_core_claim.py`

## Residual Notes

- `LLMClaimPlanner.extract_core_claim_result()` currently wraps the heuristic baseline contract around the selected claim text. That is intentional for Task 1; richer multi-agent trace semantics belong to Task 2.

## Review Follow-Up: Mixed-Case Condition Markers

Review feedback identified that `_extract_conditions()` checked lowercase markers but split the original claim with lowercase tokens, which broke mixed-case markers like `With` and `Under`.

### Follow-Up TDD Evidence

Added regression test:

- `test_heuristic_core_claim_extracts_mixed_case_condition_markers`

Red run:

```powershell
python -m pytest tests/test_core_claim.py -q
```

Observed failure:

```text
FAILED tests/test_core_claim.py::test_heuristic_core_claim_extracts_mixed_case_condition_markers
E   ValueError: not enough values to unpack (expected 2, got 1)
```

Green run:

```powershell
python -m pytest tests/test_core_claim.py -q
```

Result: `5 passed`

Required regression run:

```powershell
python -m pytest tests/test_core_claim.py tests/test_pipeline.py -q
```

Result: `22 passed`

Full suite rerun:

```powershell
python -m pytest -q
```

Result: `22 passed`

### Follow-Up Implementation

- `_extract_conditions()` now finds condition markers with a case-insensitive regex and slices the original claim text, preserving the original condition casing in the returned condition text.
- `LLMClaimPlanner.extract_core_claim_result()` now threads the fallback claim into `_extract_core_claim()` so fallback computation is not duplicated during core-claim extraction.

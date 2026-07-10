# Task 6 Report: Research Workbench UI

## Status

Implemented the scoped ClaimScope research workbench UI using the supplied desktop and mobile concepts as the visual spec.

## Delivered

- Rebuilt `app.py` around compact Core Claim Arena and Full Discovery modes.
- Added `claimscope/ui_components.py` for public event rows, candidate comparison, critique summaries, evidence rendering, feedback JSONL, provider host display, and Markdown export.
- Persisted `core_claim_result` and `discovery_result` in Streamlit session state.
- Kept provider key session-only and excluded it from exports and public result rendering.
- Disabled retrieval controls in Core Claim mode and preserved Full Discovery tabs and Markdown export.
- Added focused UI source/helper contract tests.
- Updated Streamlit theme colors to the white/cool-gray, charcoal, teal, indigo, amber, and red palette.
- Included the accepted desktop and mobile concept assets under `docs/assets/`.

## Verification

- `python -m pytest tests/test_ui_contract.py -q` -> 5 passed.
- `python -m compileall -q app.py claimscope tests` -> passed.
- `python -m pytest -q` -> 72 passed, 7 failed in pre-existing MCP/package-environment tests because the active system interpreter does not expose the optional `mcp` package or installed console entry points.
- Streamlit smoke started on `http://127.0.0.1:8503`, loaded the initial Core Claim screen, and successfully ran the heuristic Core Claim flow. The initial DOM/screenshot showed no horizontal overflow in the exercised viewport and no app console errors.

## Concerns For Controller QA

- Browser fidelity review should validate the supplied 1280x720 and 390x844 viewports, especially the Streamlit 1.32 radio fallback used where newer segmented-control support is unavailable.
- Full MCP suite should be rerun in the project environment after installing the optional MCP dependency and package entry points.

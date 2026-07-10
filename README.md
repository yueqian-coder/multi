# ClaimScope

ClaimScope turns a fuzzy research direction into a falsifiable claim, an assumption ledger, adversarial evidence queries, and bounded experiment opportunities before you commit to an idea.

> Open-source pre-ideation research tooling: deterministic by default, evidence-first, and explicit about uncertainty.

## 30-second Quick Start

```bash
python -m pip install -e ".[web,dev]"
python -m claimscope.cli "RAG can reliably reduce hallucination in LLM-generated answers"
streamlit run app.py
```

The offline demo uses bundled **synthetic fixture papers**, requires no API key, and is reproducible apart from report timestamps. The images in `docs/assets/*-concept.png` are design concepts, not product screenshots or generated report output.

## Core Claim Arena

Core Claim Arena extracts a candidate claim, exposes the method, target, expected effect, missing information, falsification test, confidence, and public activity trace. The heuristic engine is deterministic and works offline; optional multi-agent LLM mode is available through the service and MCP surfaces when configured.

## Full Assumption / Evidence Workflow

1. Normalize the direction into a core claim and boundary variants.
2. Decompose the claim into hidden, testable assumptions.
3. Generate support, contradiction, limitation, and null-result queries for each assumption.
4. Retrieve papers, map snippets to assumptions, and keep `unknown` distinct from `unsupported`.
5. Surface negative evidence, failure modes, quality warnings, and experiment-sized opportunity slots.
6. Export a Markdown report with workflow trace, provenance, evidence matrix, and limitations.

## ClaimBench

ClaimBench is a deterministic, no-network smoke benchmark with 27 cross-domain cases. It scores slot coverage, specificity, comparison language, measurable outcomes, falsification quality, and overclaim penalties.

```bash
python -m claimscope.cli benchmark --engine heuristic --data benchmarks/claimbench.jsonl
```

## MCP

```bash
python -m pip install -e ".[mcp]"
python -m claimscope.mcp_server
```

The five tools are `extract_core_claim`, `analyze_research_direction`, `build_evidence_queries`, `evaluate_claim_benchmark`, and `get_demo_report`. Optional LLM configuration uses placeholders only: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `MODEL_NAME`. Keys are read from environment variables or an in-memory UI field and are never written to reports.

## Architecture

`claimscope.models` defines typed artifacts. `core_claim` and `planner` create claims; `pipeline` coordinates the assumption/evidence workflow; `retrievers` provide offline fixtures or optional public sources; `service` provides JSON-compatible boundaries; `cli`, Streamlit, and `mcp_server` are transports. Public trace events contain role, status, duration, artifacts, and scores, not private reasoning.

## Limitations

- Bundled papers are synthetic fixtures, not research findings.
- Online retrieval is abstract/snippet-oriented and is not a substitute for full papers.
- Heuristic scores are transparent signals, not scientific validity judgments.
- LLM mode depends on a configured endpoint and can degrade to heuristics.
- ClaimBench is a smoke benchmark, not a claim of general research-agent quality.

## Roadmap

- Add opt-in full-text connectors with stronger provenance and license-aware caching.
- Expand ClaimBench annotations and publish reproducible score reports.
- Improve configurable agent rubrics and reviewer feedback loops.
- Add more export formats while preserving the public artifact contract.

## Community

- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Issue tracker](https://github.com/claimscope/claimscope/issues)
- [Discussions](https://github.com/claimscope/claimscope/discussions)

ClaimScope is released under the [MIT License](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff), and release notes are in [CHANGELOG.md](CHANGELOG.md).

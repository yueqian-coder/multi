from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import load_cases, run_benchmark
from .core_claim import HeuristicCoreClaimEngine
from .pipeline import ClaimScopePipeline
from .planner import HeuristicClaimPlanner, LLMClaimPlanner
from .llm import OpenAICompatibleClient
from .retrievers import ArxivRetriever, CombinedRetriever, SemanticScholarRetriever, StaticPaperRetriever
from .service import DEMO_PAPERS


def main() -> None:
    if (
        len(sys.argv) > 1
        and sys.argv[1] == "benchmark"
        and _has_explicit_benchmark_option(sys.argv[2:])
    ):
        _run_benchmark_cli(sys.argv[2:])
        return
    _run_analysis_cli(sys.argv[1:])


def _has_explicit_benchmark_option(argv: list[str]) -> bool:
    benchmark_options = {"--engine", "--data", "--output"}
    return any(
        arg in benchmark_options
        or any(arg.startswith(f"{option}=") for option in benchmark_options)
        for arg in argv
    )


def _run_analysis_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run a ClaimScope analysis.")
    parser.add_argument("claim", help="Research direction or claim to inspect")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use arXiv and Semantic Scholar instead of bundled demo papers.",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--planner",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Use the deterministic planner or an env-configured OpenAI-compatible LLM planner.",
    )
    args = parser.parse_args(argv)

    if args.online:
        retriever = CombinedRetriever([SemanticScholarRetriever(), ArxivRetriever()])
    else:
        retriever = StaticPaperRetriever(DEMO_PAPERS)
    planner = HeuristicClaimPlanner()
    if args.planner == "llm":
        llm_client = OpenAICompatibleClient.from_env()
        if llm_client:
            planner = LLMClaimPlanner(llm_client=llm_client)
        else:
            print(
                "LLM planner requested but OPENAI_API_KEY or OPENAI_BASE_URL is not set; "
                "falling back to heuristic planner.",
                file=sys.stderr,
            )
    report = ClaimScopePipeline(retriever=retriever, planner=planner).analyze(
        args.claim, limit=args.limit
    )
    if isinstance(planner, LLMClaimPlanner) and planner.used_planner == "heuristic":
        print(planner.fallback_reason, file=sys.stderr)
    markdown = report.to_markdown()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


def _run_benchmark_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic ClaimBench benchmark."
    )
    parser.add_argument(
        "--engine",
        choices=["heuristic"],
        default="heuristic",
        help="Benchmark engine. CI-safe heuristic mode does not require keys or network.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("benchmarks") / "claimbench.jsonl",
        help="Path to ClaimBench JSONL cases.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    engine = HeuristicCoreClaimEngine()
    cases = load_cases(args.data)
    report = run_benchmark(cases, engine)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.to_json(), encoding="utf-8")
    summary = report.summary
    print(
        "ClaimBench: "
        f"{summary['case_count']} cases, "
        f"aggregate score {summary['aggregate_score']} "
        f"({summary['engine']})"
    )


if __name__ == "__main__":
    main()

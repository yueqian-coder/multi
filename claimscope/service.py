from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import load_cases, run_benchmark
from .core_claim import HeuristicCoreClaimEngine
from .llm import OpenAICompatibleClient
from .models import AnalysisReport, Paper
from .pipeline import ClaimScopePipeline
from .planner import ClaimPlan, HeuristicClaimPlanner, LLMClaimPlanner
from .retrievers import (
    ArxivRetriever,
    CombinedRetriever,
    PaperRetriever,
    SemanticScholarRetriever,
    StaticPaperRetriever,
)


DEMO_PAPERS = [
    Paper(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        year=2020,
        authors=["Lewis et al."],
        abstract=(
            "Retrieval-augmented generation improves factuality for knowledge-intensive "
            "question answering by conditioning answers on retrieved evidence. However, "
            "performance depends strongly on retrieval quality, and noisy passages can "
            "hurt generation."
        ),
        source="demo",
        url="https://arxiv.org/abs/2005.11401",
    ),
    Paper(
        title="Evaluating Retrieval-Augmented Language Models for Factuality",
        year=2023,
        authors=["Chen et al."],
        abstract=(
            "RAG can reduce hallucination on open-domain QA when evidence is relevant. "
            "The gains are smaller under domain shift, and citation mismatch remains a "
            "common failure mode."
        ),
        source="demo",
    ),
    Paper(
        title="On the Limits of Retrieval-Augmented Generation",
        year=2024,
        authors=["Patel et al."],
        abstract=(
            "We find no consistent improvement for long-form generation when retrieved "
            "documents are irrelevant. Limitations include retrieval noise, unsupported "
            "claims, and brittle evaluation metrics."
        ),
        source="demo",
    ),
]


class ClaimScopeService:
    def __init__(
        self,
        *,
        offline_retriever: PaperRetriever | None = None,
        online_retriever: PaperRetriever | None = None,
        heuristic_planner: object | None = None,
        llm_client: object | None = None,
        benchmark_path: str | Path | None = None,
    ) -> None:
        self.offline_retriever = offline_retriever or StaticPaperRetriever(DEMO_PAPERS)
        self.online_retriever = online_retriever or CombinedRetriever(
            [SemanticScholarRetriever(), ArxivRetriever()]
        )
        self.heuristic_planner = heuristic_planner or HeuristicClaimPlanner()
        self.llm_client = llm_client
        self.benchmark_path = Path(benchmark_path) if benchmark_path else _default_benchmark_path()

    @classmethod
    def from_env(cls) -> "ClaimScopeService":
        return cls(llm_client=OpenAICompatibleClient.from_env())

    def extract_core_claim(self, direction: str, mode: str = "auto") -> dict:
        planner = self._planner_for_mode(mode)
        extractor = getattr(planner, "extract_core_claim_result", None)
        if extractor:
            result = extractor(direction)
        else:
            result = HeuristicCoreClaimEngine().run(planner.extract_core_claim(direction))
        return _jsonable(result)

    def analyze_research_direction(
        self, direction: str, online: bool = False, limit: int = 12
    ) -> dict:
        pipeline = ClaimScopePipeline(
            retriever=self.online_retriever if online else self.offline_retriever,
            planner=self._planner_for_mode("auto"),
        )
        return _analysis_payload(pipeline.analyze(direction, limit=limit))

    def build_evidence_queries(self, direction: str, mode: str = "auto") -> dict:
        plan = self._planner_for_mode(mode).build(direction)
        return _plan_payload(direction, plan)

    def evaluate_claim_benchmark(self) -> dict:
        cases = load_cases(self.benchmark_path)
        return run_benchmark(cases, HeuristicCoreClaimEngine()).to_dict()

    def get_demo_report(self) -> dict:
        report = ClaimScopePipeline(
            retriever=self.offline_retriever,
            planner=self._planner_for_mode("heuristic"),
        ).analyze("RAG can reliably reduce hallucination in LLM-generated answers", limit=12)
        return {
            "report": _analysis_payload(report),
            "markdown": report.to_markdown(),
        }

    def _planner_for_mode(self, mode: str) -> object:
        normalized = mode.strip().lower()
        if normalized == "heuristic":
            return self.heuristic_planner
        if normalized not in {"auto", "llm"}:
            raise ValueError("mode must be one of: auto, heuristic, llm")
        llm_client = self.llm_client or OpenAICompatibleClient.from_env()
        if llm_client:
            return LLMClaimPlanner(llm_client=llm_client)
        return self.heuristic_planner


def _analysis_payload(report: AnalysisReport) -> dict:
    payload = _jsonable(report)
    payload["workflow_steps"] = _jsonable(report.workflow_steps())
    payload["assumption_matrix"] = report.assumption_matrix()
    return payload


def _plan_payload(direction: str, plan: ClaimPlan) -> dict:
    queries = []
    for assumption in plan.assumptions:
        queries.extend(
            [
                {
                    "assumption": assumption.text,
                    "kind": "support",
                    "query": assumption.support_query,
                },
                {
                    "assumption": assumption.text,
                    "kind": "contradict",
                    "query": assumption.contradict_query,
                },
                {
                    "assumption": assumption.text,
                    "kind": "limitation",
                    "query": assumption.limitation_query,
                },
                {
                    "assumption": assumption.text,
                    "kind": "null_result",
                    "query": assumption.null_result_query,
                },
            ]
        )
    return {
        "direction": direction,
        "claim": plan.claim,
        "claim_variants": _jsonable(plan.claim_variants),
        "assumptions": _jsonable(plan.assumptions),
        "queries": queries,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _default_benchmark_path() -> Path:
    return Path(__file__).resolve().parents[1] / "benchmarks" / "claimbench.jsonl"

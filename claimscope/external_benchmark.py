from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

from .evidence import infer_claim_label, infer_effect_label
from .models import Paper
from .pipeline import _mine_negative_evidence
from .planner import ClaimPlanner, HeuristicClaimPlanner
from .retrievers import (
    BM25PaperRetriever,
    ReciprocalRankFusionRetriever,
    SentenceTransformerPaperRetriever,
    TfidfPaperRetriever,
)
from .text_utils import retrieval_tokens


DEFAULT_DATASETS = (
    "scifact_open",
    "evidence_inference",
    "nli4ct",
    "limitgen",
    "claimdecomp",
    "litsearch",
)


@dataclass
class DatasetEvaluation:
    dataset: str
    task: str
    status: str
    scope: str
    case_count: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ExternalBenchmarkReport:
    engine: str
    datasets: list[DatasetEvaluation]
    recommendations: list[str]
    protocol: dict[str, object] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "engine": self.engine,
            "protocol": self.protocol,
            "summary": {
                "requested": len(self.datasets),
                "evaluated": sum(item.status == "evaluated" for item in self.datasets),
                "partial": sum(item.status == "partial" for item in self.datasets),
                "unavailable": sum(
                    item.status == "unavailable" for item in self.datasets
                ),
                "aggregate_score": None,
                "aggregate_score_note": (
                    "No aggregate score: retrieval, NLI, effect inference, limitation "
                    "coverage, and claim decomposition are different tasks."
                ),
            },
            "datasets": [item.to_dict() for item in self.datasets],
            "recommendations": self.recommendations,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class LLMBenchmarkAdjudicator:
    """Batched, public-output-only classifier for benchmark adjudication."""

    llm_client: object
    batch_size: int = 10
    failed_batches: int = field(default=0, init=False)

    def predict(
        self,
        task: str,
        examples: list[dict[str, object]],
        labels: tuple[str, ...],
    ) -> list[str]:
        predictions: dict[str, str] = {}
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start : start + self.batch_size]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific evidence classifier, not a conversational "
                        "assistant. Classify every example using only the supplied evidence. "
                        "Return JSON only with shape "
                        "{\"predictions\":[{\"id\":\"...\",\"label\":\"...\"}]}. "
                        "Do not provide reasoning, prose, citations, or additional keys."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task,
                            "allowed_labels": list(labels),
                            "examples": batch,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            try:
                content = self.llm_client.chat(messages, temperature=0.0, timeout=120)
                payload = _extract_json_object(content)
                for item in payload.get("predictions", []):
                    identifier = str(item.get("id", ""))
                    label = str(item.get("label", "")).lower()
                    if identifier and label in labels:
                        predictions[identifier] = label
            except Exception:
                self.failed_batches += 1
        return [predictions.get(str(item["id"]), "unknown") for item in examples]


def run_external_benchmark(
    data_dir: str | Path,
    *,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    max_cases: int | None = None,
    seed: int = 13,
    planner: ClaimPlanner | None = None,
    adjudicator: LLMBenchmarkAdjudicator | None = None,
    retrieval_engine: str = "bm25",
    engine: str = "heuristic+bm25",
) -> ExternalBenchmarkReport:
    root = Path(data_dir)
    selected = list(dict.fromkeys(datasets))
    unknown = sorted(set(selected) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"unknown external benchmark dataset(s): {', '.join(unknown)}")
    planner = planner or HeuristicClaimPlanner()
    runners = {
        "scifact_open": _evaluate_scifact_open,
        "evidence_inference": _evaluate_evidence_inference,
        "nli4ct": _evaluate_nli4ct,
        "limitgen": _evaluate_limitgen,
        "claimdecomp": _evaluate_claimdecomp,
        "litsearch": _evaluate_litsearch,
    }
    evaluations = [
        runners[name](
            root,
            max_cases=max_cases,
            seed=seed,
            planner=planner,
            adjudicator=adjudicator,
            retrieval_engine=retrieval_engine,
        )
        for name in selected
    ]
    return ExternalBenchmarkReport(
        engine=engine,
        datasets=evaluations,
        recommendations=_recommendations(evaluations),
        protocol={
            "datasets": selected,
            "max_cases_per_dataset": max_cases,
            "seed": seed,
            "retrieval_engine": retrieval_engine,
            "planner": planner.__class__.__name__,
            "adjudicator": (
                adjudicator.__class__.__name__ if adjudicator else "deterministic"
            ),
        },
    )


def _evaluate_scifact_open(
    root: Path,
    *,
    max_cases: int | None,
    seed: int,
    planner: ClaimPlanner,
    adjudicator: LLMBenchmarkAdjudicator | None,
    retrieval_engine: str,
) -> DatasetEvaluation:
    del planner
    claims_path = root / "scifact-open" / "data" / "claims.jsonl"
    corpus_path = root / "scifact-open" / "data" / "corpus_candidates.jsonl"
    if not claims_path.exists() or not corpus_path.exists():
        return _unavailable(
            "scifact_open",
            "open-domain scientific claim retrieval and verification",
            "Run scripts/prepare_external_benchmarks.py --datasets scifact_open.",
        )
    claims = _sample(_read_jsonl(claims_path), max_cases, seed)
    corpus_rows = _read_jsonl(corpus_path)
    corpus_by_id = {str(row.get("doc_id", "")): row for row in corpus_rows}
    papers = [
        Paper(
            title=str(row.get("title", "")),
            year=_coerce_int((row.get("metadata") or {}).get("year")),
            authors=[],
            abstract=" ".join(row.get("abstract") or []),
            source="SciFact-Open",
            external_id=str(row.get("doc_id", "")),
        )
        for row in corpus_rows
    ]
    by_id = {paper.external_id: paper for paper in papers}
    retriever = _build_retriever(
        papers,
        retrieval_engine,
        embedding_cache_path=root / ".index-cache" / "scifact-open-e5-small-v2.npz",
    )
    recalls = {5: [], 20: [], 100: []}
    reciprocal_ranks: list[float] = []
    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    adjudication_examples: list[dict[str, object]] = []
    for row in claims:
        claim = str(row.get("claim", ""))
        evidence = row.get("evidence") or {}
        relevant = {str(doc_id) for doc_id in evidence}
        ranked = retriever.search(claim, limit=100)
        ranked_ids = [paper.external_id for paper in ranked]
        _append_retrieval_metrics(ranked_ids, relevant, recalls, reciprocal_ranks)
        for doc_id, annotation in evidence.items():
            paper = by_id.get(str(doc_id))
            if not paper:
                continue
            sentence_indexes = annotation.get("sentences") or []
            abstract_sentences = corpus_by_id.get(str(doc_id), {}).get("abstract") or []
            snippet = " ".join(
                abstract_sentences[index]
                for index in sentence_indexes
                if isinstance(index, int) and 0 <= index < len(abstract_sentences)
            )
            gold_labels.append(
                "entailment"
                if str(annotation.get("label", "")).upper() == "SUPPORT"
                else "contradiction"
            )
            predicted_labels.append(infer_claim_label(claim, snippet))
            adjudication_examples.append(
                {
                    "id": str(len(adjudication_examples)),
                    "claim": claim,
                    "evidence": snippet[:4000],
                }
            )
    if adjudicator:
        predicted_labels = adjudicator.predict(
            "Decide whether the evidence entails or contradicts the scientific claim.",
            adjudication_examples,
            ("entailment", "contradiction"),
        )
    metrics = _retrieval_metric_dict(recalls, reciprocal_ranks)
    metrics.update(
        _classification_metrics(
            gold_labels,
            predicted_labels,
            labels=("entailment", "contradiction"),
            prefix="gold_evidence_stance_",
        )
    )
    return DatasetEvaluation(
        dataset="scifact_open",
        task="open-domain scientific claim retrieval and stance",
        status="evaluated",
        scope=f"{retrieval_engine} over the official ~12K candidate pool; stance uses gold evidence sentences.",
        case_count=len(claims),
        metrics=metrics,
        notes=[
            "This is candidate-pool retrieval, not the full 500K-corpus score.",
            "Pooling highlights may be model-generated according to the dataset authors.",
            *_adjudicator_notes(adjudicator),
        ],
    )


def _evaluate_evidence_inference(
    root: Path,
    *,
    max_cases: int | None,
    seed: int,
    planner: ClaimPlanner,
    adjudicator: LLMBenchmarkAdjudicator | None,
    retrieval_engine: str,
) -> DatasetEvaluation:
    del planner, retrieval_engine
    annotations_path = root / "evidence_annotations.csv"
    prompts_path = root / "evidence_prompts.csv"
    if not annotations_path.exists() or not prompts_path.exists():
        return _unavailable(
            "evidence_inference",
            "RCT comparative-effect inference",
            "Run scripts/prepare_external_benchmarks.py --datasets evidence_inference.",
        )
    with prompts_path.open(encoding="utf-8-sig", newline="") as handle:
        prompts = {row["PromptID"]: row for row in csv.DictReader(handle)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with annotations_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("Valid Label", "")).lower() not in {"true", "1"}:
                continue
            if not row.get("Annotations") or row.get("Label Code") not in {"-1", "0", "1"}:
                continue
            grouped[row["PromptID"]].append(row)
    prompt_ids = _sample(sorted(grouped), max_cases, seed)
    gold: list[str] = []
    predicted: list[str] = []
    adjudication_examples: list[dict[str, object]] = []
    for prompt_id in prompt_ids:
        rows = grouped[prompt_id]
        gold_label = Counter(row["Label Code"] for row in rows).most_common(1)[0][0]
        snippets = list(dict.fromkeys(row["Annotations"].strip() for row in rows))
        prompt = prompts.get(prompt_id, {})
        gold.append(gold_label)
        predicted.append(
            infer_effect_label(" ".join(snippets), str(prompt.get("Outcome", "")))
        )
        adjudication_examples.append(
            {
                "id": str(len(adjudication_examples)),
                "outcome": str(prompt.get("Outcome", "")),
                "intervention": str(prompt.get("Intervention", "")),
                "comparator": str(prompt.get("Comparator", "")),
                "evidence": " ".join(snippets)[:5000],
            }
        )
    if adjudicator:
        predicted = adjudicator.predict(
            (
                "Relative to the comparator, classify the reported outcome effect as "
                "-1 (significantly decreased), 0 (no significant difference), or "
                "1 (significantly increased)."
            ),
            adjudication_examples,
            ("-1", "0", "1"),
        )
    metrics = _classification_metrics(
        gold,
        predicted,
        labels=("-1", "0", "1"),
        prefix="effect_",
    )
    metrics["null_result_recall"] = _label_recall(gold, predicted, "0")
    return DatasetEvaluation(
        dataset="evidence_inference",
        task="increase/decrease/no-significant-difference inference",
        status="partial",
        scope="Gold evidence snippets only; full-text evidence retrieval is not measured.",
        case_count=len(prompt_ids),
        metrics=metrics,
        notes=[
            "The deterministic baseline abstains when direction cannot be inferred safely.",
            "A biomedical NLI/effect model is required for claims such as 'superior' whose direction depends on the outcome.",
            *_adjudicator_notes(adjudicator),
        ],
    )


def _evaluate_nli4ct(
    root: Path,
    *,
    max_cases: int | None,
    seed: int,
    planner: ClaimPlanner,
    adjudicator: LLMBenchmarkAdjudicator | None,
    retrieval_engine: str,
) -> DatasetEvaluation:
    del planner, retrieval_engine
    dataset_root = root / "nli4ct" / "Complete_dataset"
    cases_path = dataset_root / "Gold_test.json"
    trials_dir = dataset_root / "CT json"
    if not cases_path.exists() or not trials_dir.exists():
        return _unavailable(
            "nli4ct",
            "multi-evidence clinical-trial NLI",
            "Run scripts/prepare_external_benchmarks.py --datasets nli4ct.",
        )
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = _sample(list(payload.items()), max_cases, seed)
    gold: list[str] = []
    predicted: list[str] = []
    evidence_recalls: list[float] = []
    evidence_f1s: list[float] = []
    adjudication_examples: list[dict[str, object]] = []
    trial_cache: dict[str, dict[str, list[str]]] = {}
    for _, case in cases:
        section = str(case["Section_id"])
        candidates: list[tuple[str, str]] = []
        relevant: set[str] = set()
        gold_text: list[str] = []
        for prefix, id_key, evidence_key in (
            ("P", "Primary_id", "Primary_evidence_index"),
            ("S", "Secondary_id", "Secondary_evidence_index"),
        ):
            trial_id = case.get(id_key)
            if not trial_id:
                continue
            trial = trial_cache.get(str(trial_id))
            if trial is None:
                trial = json.loads(
                    (trials_dir / f"{trial_id}.json").read_text(encoding="utf-8")
                )
                trial_cache[str(trial_id)] = trial
            lines = trial.get(section) or []
            candidates.extend((f"{prefix}:{index}", text) for index, text in enumerate(lines))
            for index in case.get(evidence_key) or []:
                if isinstance(index, int) and 0 <= index < len(lines):
                    relevant.add(f"{prefix}:{index}")
                    gold_text.append(lines[index])
        ranked_ids = _rank_text_candidates(str(case["Statement"]), candidates, limit=5)
        retrieved = set(ranked_ids)
        true_positive = len(retrieved & relevant)
        evidence_recalls.append(true_positive / len(relevant) if relevant else 0.0)
        precision = true_positive / len(retrieved) if retrieved else 0.0
        recall = true_positive / len(relevant) if relevant else 0.0
        evidence_f1s.append(_f1(precision, recall))
        gold_label = (
            "entailment"
            if str(case.get("Label", "")).lower() == "entailment"
            else "contradiction"
        )
        gold.append(gold_label)
        predicted.append(
            infer_claim_label(str(case["Statement"]), " ".join(gold_text))
        )
        adjudication_examples.append(
            {
                "id": str(len(adjudication_examples)),
                "statement": str(case["Statement"]),
                "evidence": " ".join(gold_text)[:6000],
            }
        )
    if adjudicator:
        predicted = adjudicator.predict(
            "Decide whether the clinical-trial evidence entails or contradicts the statement.",
            adjudication_examples,
            ("entailment", "contradiction"),
        )
    metrics = _classification_metrics(
        gold,
        predicted,
        labels=("entailment", "contradiction"),
        prefix="gold_evidence_nli_",
    )
    metrics.update(
        {
            "evidence_recall@5": _rounded_mean(evidence_recalls),
            "evidence_f1@5": _rounded_mean(evidence_f1s),
        }
    )
    return DatasetEvaluation(
        dataset="nli4ct",
        task="clinical-trial evidence selection and entailment/contradiction",
        status="evaluated",
        scope="BM25 section-line selection plus lexical NLI on official gold evidence.",
        case_count=len(cases),
        metrics=metrics,
        notes=[
            "Gold-evidence NLI and evidence selection are reported separately.",
            "The current pipeline is abstract-oriented; this benchmark exposes the need for structured full-text evidence handling.",
            *_adjudicator_notes(adjudicator),
        ],
    )


def _evaluate_limitgen(
    root: Path,
    *,
    max_cases: int | None,
    seed: int,
    planner: ClaimPlanner,
    adjudicator: LLMBenchmarkAdjudicator | None,
    retrieval_engine: str,
) -> DatasetEvaluation:
    del adjudicator, retrieval_engine
    path = root / "limitgen_human.json"
    if not path.exists():
        return _unavailable(
            "limitgen",
            "scientific limitation identification",
            "Run scripts/prepare_external_benchmarks.py --datasets limitgen.",
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _sample(list(payload.values()), max_cases, seed)
    lexical_scores: list[float] = []
    category_scores: list[float] = []
    explicit_finding: list[float] = []
    cue_map = {
        "methodology": ("method", "mechanism", "assumption", "data quality"),
        "experimental design": (
            "dataset",
            "baseline",
            "generaliz",
            "ablation",
            "sample",
        ),
        "result analysis": ("metric", "evaluation", "significance", "analysis"),
        "literature review": ("citation", "prior work", "literature", "evidence"),
    }
    for row in rows:
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        plan = planner.build(f"{title}. {abstract[:1200]}")
        generated = [item.text for item in plan.assumptions]
        generated.extend(item.limitation_query for item in plan.assumptions)
        generated_text = " ".join(generated).lower()
        limitations = row.get("limitations") or {}
        gold_items = [
            text
            for values in limitations.values()
            for text in (values if isinstance(values, list) else [])
        ]
        lexical_scores.append(_set_coverage(gold_items, generated))
        present_categories = [
            category for category, values in limitations.items() if values
        ]
        category_scores.append(
            mean(
                float(any(cue in generated_text for cue in cue_map.get(category, ())))
                for category in present_categories
            )
            if present_categories
            else 0.0
        )
        paper = Paper(
            title=title,
            year=None,
            authors=[],
            abstract=abstract,
            source="LimitGen",
        )
        explicit_finding.append(float(bool(_mine_negative_evidence([paper]))))
    return DatasetEvaluation(
        dataset="limitgen",
        task="limitation coverage transfer test",
        status="partial",
        scope="LimitGen-Human title/abstract transfer; compares planned checks with review limitations.",
        case_count=len(rows),
        metrics={
            "limitation_lexical_coverage": _rounded_mean(lexical_scores),
            "limitation_category_cue_coverage": _rounded_mean(category_scores),
            "explicit_abstract_finding_rate": _rounded_mean(explicit_finding),
        },
        notes=[
            "LimitGen is an adjacent peer-review task, not a direct end-to-end ClaimScope benchmark.",
            "Low lexical coverage does not prove semantic failure; it is a conservative proxy pending expert or LLM matching.",
        ],
    )


def _evaluate_claimdecomp(
    root: Path,
    *,
    max_cases: int | None,
    seed: int,
    planner: ClaimPlanner,
    adjudicator: LLMBenchmarkAdjudicator | None,
    retrieval_engine: str,
) -> DatasetEvaluation:
    del adjudicator, retrieval_engine
    path = root / "claimdecomp_annotations.csv"
    if not path.exists():
        return _unavailable(
            "claimdecomp",
            "literal and implied claim decomposition",
            "Run scripts/prepare_external_benchmarks.py --datasets claimdecomp.",
        )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = _sample(list(csv.DictReader(handle)), max_cases, seed)
    all_scores: list[float] = []
    implicit_scores: list[float] = []
    generated_counts: list[float] = []
    implicit_case_count = 0
    for row in rows:
        plan = planner.build(row["claim"])
        generated = [item.text for item in plan.assumptions]
        generated.extend(item.text for item in plan.claim_variants)
        generated_counts.append(float(len(generated)))
        all_questions = _line_items(row.get("questions-all", ""))
        implied_questions = _line_items(row.get("questions-implied", ""))
        all_scores.append(_set_coverage(all_questions, generated))
        if implied_questions:
            implicit_case_count += 1
            implicit_scores.append(_set_coverage(implied_questions, generated))
    return DatasetEvaluation(
        dataset="claimdecomp",
        task="explicit proposition and implicit-facet decomposition",
        status="partial",
        scope="Author-released 100-row annotation sheet; lexical matching proxy.",
        case_count=len(rows),
        metrics={
            "question_lexical_coverage": _rounded_mean(all_scores),
            "implicit_question_lexical_coverage": _rounded_mean(implicit_scores),
            "implicit_case_count": float(implicit_case_count),
            "generated_items_per_claim": _rounded_mean(generated_counts),
        },
        notes=[
            "CLAIMDECOMP is political fact-checking, so this measures decomposition transfer rather than scientific validity.",
            "The complete 1,200-row download URLs are no longer live; the official annotation sheet contains 100 rows.",
        ],
    )


def _evaluate_litsearch(
    root: Path,
    *,
    max_cases: int | None,
    seed: int,
    planner: ClaimPlanner,
    adjudicator: LLMBenchmarkAdjudicator | None,
    retrieval_engine: str,
) -> DatasetEvaluation:
    del planner, adjudicator
    queries_path = root / "litsearch_queries.jsonl"
    corpus_path = root / "litsearch_gold_corpus.jsonl"
    if not queries_path.exists() or not corpus_path.exists():
        return _unavailable(
            "litsearch",
            "scientific literature retrieval",
            "Run scripts/prepare_external_benchmarks.py --datasets litsearch.",
        )
    queries = _sample(_read_jsonl(queries_path), max_cases, seed)
    corpus = _read_jsonl(corpus_path)
    papers = [
        Paper(
            title=str(row.get("title", "")),
            year=None,
            authors=[],
            abstract=str(row.get("abstract", "")),
            source="LitSearch",
            external_id=str(row.get("corpusid", "")),
        )
        for row in corpus
    ]
    retriever = _build_retriever(
        papers,
        retrieval_engine,
        embedding_cache_path=root / ".index-cache" / "litsearch-gold-e5-small-v2.npz",
    )
    recalls = {5: [], 20: [], 100: []}
    reciprocal_ranks: list[float] = []
    for row in queries:
        relevant = {str(item) for item in row.get("corpusids") or []}
        ranked = retriever.search(str(row.get("query", "")), limit=100)
        _append_retrieval_metrics(
            [paper.external_id for paper in ranked],
            relevant,
            recalls,
            reciprocal_ranks,
        )
    return DatasetEvaluation(
        dataset="litsearch",
        task="realistic scientific literature query retrieval",
        status="partial",
        scope=f"{retrieval_engine} over the union of 574 gold papers for all 597 official queries.",
        case_count=len(queries),
        metrics=_retrieval_metric_dict(recalls, reciprocal_ranks),
        notes=[
            "This gold-union candidate test is diagnostic and easier than the official ~1.6GB corpus.",
            "Do not compare these numbers directly with the LitSearch leaderboard.",
        ],
    )


def _recommendations(evaluations: list[DatasetEvaluation]) -> list[str]:
    by_name = {item.dataset: item for item in evaluations}
    recommendations: list[str] = []
    scifact = by_name.get("scifact_open")
    litsearch = by_name.get("litsearch")
    nli4ct = by_name.get("nli4ct")
    evidence = by_name.get("evidence_inference")
    limitgen = by_name.get("limitgen")
    claimdecomp = by_name.get("claimdecomp")
    if scifact and scifact.metrics.get("recall@20", 1.0) < 0.75:
        if "dense" in scifact.scope:
            recommendations.append(
                "Evaluate the official full corpus, then tune rank fusion or add "
                "a cross-encoder; fine-tune retrieval only if held-out Recall@20 "
                "remains below target."
            )
        else:
            recommendations.append(
                "Add a pretrained scientific dense retriever and reciprocal-rank fusion after BM25."
            )
    if litsearch and litsearch.metrics.get("recall@20", 1.0) < 0.80:
        recommendations.append(
            "Add query expansion plus a cross-encoder reranker for natural-language literature questions."
        )
    if nli4ct and nli4ct.metrics.get("gold_evidence_nli_macro_f1", 1.0) < 0.65:
        recommendations.append(
            "Replace lexical stance rules with a pretrained scientific/clinical NLI model and calibrated abstention."
        )
    if evidence and evidence.metrics.get("effect_macro_f1", 1.0) < 0.65:
        recommendations.append(
            "Use an outcome-aware three-way effect classifier for increase/decrease/null-result evidence."
        )
    if limitgen and limitgen.metrics.get("limitation_lexical_coverage", 1.0) < 0.25:
        recommendations.append(
            "Ground limitation discovery in full text and retrieved comparison papers; abstract keyword mining is insufficient."
        )
    if claimdecomp and claimdecomp.metrics.get(
        "implicit_question_lexical_coverage", 1.0
    ) < 0.25:
        recommendations.append(
            "Train or fine-tune an explicit-vs-implied decomposition model only after collecting scientific-domain annotations."
        )
    return recommendations


def _classification_metrics(
    gold: list[str],
    predicted: list[str],
    *,
    labels: tuple[str, ...],
    prefix: str,
) -> dict[str, float]:
    if not gold:
        return {
            f"{prefix}accuracy": 0.0,
            f"{prefix}macro_f1": 0.0,
            f"{prefix}coverage": 0.0,
        }
    answered = [prediction != "unknown" for prediction in predicted]
    f1s = []
    for label in labels:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(gold, predicted)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(gold, predicted)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(gold, predicted)
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1s.append(_f1(precision, recall))
    return {
        f"{prefix}accuracy": round(
            sum(expected == actual for expected, actual in zip(gold, predicted))
            / len(gold),
            4,
        ),
        f"{prefix}macro_f1": round(mean(f1s), 4),
        f"{prefix}coverage": round(sum(answered) / len(answered), 4),
    }


def _rank_text_candidates(
    query: str,
    candidates: list[tuple[str, str]],
    *,
    limit: int,
) -> list[str]:
    papers: list[Paper] = []
    for index, (identifier, text) in enumerate(candidates):
        prefix = identifier.split(":", 1)[0]
        context = [
            candidates[neighbor][1]
            for neighbor in range(max(0, index - 1), min(len(candidates), index + 2))
            if candidates[neighbor][0].startswith(f"{prefix}:")
        ]
        papers.append(
            Paper(
                title=text,
                year=None,
                authors=[],
                abstract=" ".join(context),
                source="NLI4CT",
                external_id=identifier,
            )
        )
    return [
        paper.external_id for paper in BM25PaperRetriever(papers).search(query, limit=limit)
    ]


def _build_retriever(
    papers: list[Paper],
    engine: str,
    *,
    embedding_cache_path: str | Path | None = None,
):
    if engine == "bm25":
        return BM25PaperRetriever(papers)
    if engine == "tfidf":
        return TfidfPaperRetriever(papers)
    if engine == "hybrid-tfidf":
        return ReciprocalRankFusionRetriever(
            [BM25PaperRetriever(papers), TfidfPaperRetriever(papers)]
        )
    if engine == "dense":
        return SentenceTransformerPaperRetriever(
            papers,
            embedding_cache_path=embedding_cache_path,
        )
    if engine == "hybrid-dense":
        return ReciprocalRankFusionRetriever(
            [
                BM25PaperRetriever(papers),
                SentenceTransformerPaperRetriever(
                    papers,
                    embedding_cache_path=embedding_cache_path,
                ),
            ]
        )
    raise ValueError(f"unknown retrieval engine: {engine}")


def _append_retrieval_metrics(
    ranked_ids: list[str],
    relevant: set[str],
    recalls: dict[int, list[float]],
    reciprocal_ranks: list[float],
) -> None:
    for cutoff, values in recalls.items():
        found = len(set(ranked_ids[:cutoff]) & relevant)
        values.append(found / len(relevant) if relevant else 0.0)
    first_rank = next(
        (index for index, identifier in enumerate(ranked_ids, start=1) if identifier in relevant),
        None,
    )
    reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)


def _retrieval_metric_dict(
    recalls: dict[int, list[float]],
    reciprocal_ranks: list[float],
) -> dict[str, float]:
    metrics = {
        f"recall@{cutoff}": _rounded_mean(values)
        for cutoff, values in recalls.items()
    }
    metrics["mrr@100"] = _rounded_mean(reciprocal_ranks)
    return metrics


def _label_recall(gold: list[str], predicted: list[str], label: str) -> float:
    indexes = [index for index, value in enumerate(gold) if value == label]
    if not indexes:
        return 0.0
    return round(sum(predicted[index] == label for index in indexes) / len(indexes), 4)


def _set_coverage(gold_items: list[str], predicted_items: list[str]) -> float:
    if not gold_items or not predicted_items:
        return 0.0
    return mean(
        max(_token_f1(gold, predicted) for predicted in predicted_items)
        for gold in gold_items
    )


def _token_f1(left: str, right: str) -> float:
    left_tokens = Counter(retrieval_tokens(left))
    right_tokens = Counter(retrieval_tokens(right))
    overlap = sum((left_tokens & right_tokens).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(right_tokens.values())
    recall = overlap / sum(left_tokens.values())
    return _f1(precision, recall)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _line_items(value: str) -> list[str]:
    return [
        item.strip()
        for item in (value or "").splitlines()
        if item.strip() and item.strip().lower() not in {"n/a", "na"}
    ]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample(items: list, max_cases: int | None, seed: int) -> list:
    if max_cases is None or max_cases <= 0 or len(items) <= max_cases:
        return items
    return random.Random(seed).sample(items, max_cases)


def _rounded_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _coerce_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _unavailable(dataset: str, task: str, note: str) -> DatasetEvaluation:
    return DatasetEvaluation(
        dataset=dataset,
        task=task,
        status="unavailable",
        scope="Dataset files were not found in the configured data directory.",
        notes=[note],
    )


def _extract_json_object(content: str) -> dict:
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM benchmark response did not contain a JSON object.")
    payload = json.loads(content[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM benchmark response must be an object.")
    return payload


def _adjudicator_notes(
    adjudicator: LLMBenchmarkAdjudicator | None,
) -> list[str]:
    if adjudicator is None:
        return []
    if adjudicator.failed_batches:
        return [
            f"LLM adjudication failed for {adjudicator.failed_batches} batch(es); missing predictions count as abstentions."
        ]
    return ["Stance/effect labels were produced by batched LLM adjudication on gold evidence."]

import json
import sys
import types

import pytest

from claimscope.evidence import (
    classify_evidence_stance,
    infer_claim_label,
    infer_effect_label,
)
from claimscope.external_benchmark import LLMBenchmarkAdjudicator, run_external_benchmark
from claimscope.models import Paper
from claimscope.retrievers import BM25PaperRetriever, SentenceTransformerPaperRetriever


def test_bm25_retriever_prefers_rare_query_terms_and_abstains_on_no_match():
    papers = [
        Paper(
            title="General language modeling",
            year=2024,
            authors=[],
            abstract="A broad study of language models and evaluation.",
            source="test",
            external_id="general",
        ),
        Paper(
            title="Citrullinated proteins in neutrophil traps",
            year=2024,
            authors=[],
            abstract="We study citrullination in inflammatory cycles.",
            source="test",
            external_id="specific",
        ),
    ]
    retriever = BM25PaperRetriever(papers)

    ranked = retriever.search("citrullinated neutrophil inflammation", 2)
    assert ranked[0].external_id == "specific"
    assert retriever.search("zzzxxyy unmatched", 2) == []


def test_evidence_classifier_keeps_null_results_separate_from_refutation():
    assert (
        classify_evidence_stance("There was no significant difference between groups.")
        == "null_result"
    )
    assert (
        infer_effect_label("The intervention significantly reduced pain.", "pain")
        == "-1"
    )
    assert infer_effect_label("The groups did not differ (p = 0.61).", "pain") == "0"
    assert (
        infer_claim_label(
            "Treatment improves survival",
            "Treatment does not improve survival in the study cohort",
        )
        == "contradiction"
    )


def test_external_benchmark_reports_per_task_metrics_without_fake_aggregate(tmp_path):
    scifact = tmp_path / "scifact-open" / "data"
    scifact.mkdir(parents=True)
    (scifact / "claims.jsonl").write_text(
        json.dumps(
            {
                "id": 1,
                "claim": "Aspirin reduces headache duration",
                "evidence": {
                    "11": {
                        "label": "SUPPORT",
                        "sentences": [0],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (scifact / "corpus_candidates.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": 11,
                        "title": "Aspirin for headache duration",
                        "abstract": ["Aspirin reduces headache duration in adults."],
                        "metadata": {"year": 2024},
                    }
                ),
                json.dumps(
                    {
                        "doc_id": 12,
                        "title": "Unrelated protein study",
                        "abstract": ["We analyze protein folding."],
                        "metadata": {"year": 2023},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_external_benchmark(
        tmp_path,
        datasets=["scifact_open", "litsearch"],
    )
    payload = report.to_dict()

    assert payload["summary"]["aggregate_score"] is None
    assert payload["protocol"]["seed"] == 13
    assert payload["protocol"]["retrieval_engine"] == "bm25"
    assert payload["summary"]["evaluated"] == 1
    assert payload["summary"]["unavailable"] == 1
    assert report.datasets[0].metrics["recall@5"] == 1.0
    assert report.datasets[1].status == "unavailable"


def test_llm_adjudicator_batches_public_labels_and_abstains_on_missing_ids():
    class FakeClient:
        def chat(self, messages, temperature=0.0, timeout=120):
            assert "chain-of-thought" not in messages[0]["content"].lower()
            return '{"predictions":[{"id":"0","label":"entailment"}]}'

    adjudicator = LLMBenchmarkAdjudicator(FakeClient(), batch_size=2)

    assert adjudicator.predict(
        "classify",
        [{"id": "0", "evidence": "yes"}, {"id": "1", "evidence": "no"}],
        ("entailment", "contradiction"),
    ) == ["entailment", "unknown"]


def test_dense_retriever_reuses_validated_embedding_cache(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")

    class FakeSentenceTransformer:
        encode_calls = 0

        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, **kwargs):
            del kwargs
            type(self).encode_calls += 1
            return np.asarray(
                [[float("aspirin" in text), float("protein" in text)] for text in texts]
            )

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    papers = [
        Paper(
            title="Aspirin outcomes",
            year=2024,
            authors=[],
            abstract="aspirin reduces headache duration",
            source="test",
            external_id="1",
        ),
        Paper(
            title="Protein folding",
            year=2024,
            authors=[],
            abstract="protein structure analysis",
            source="test",
            external_id="2",
        ),
    ]
    cache_path = tmp_path / "index.npz"

    first = SentenceTransformerPaperRetriever(
        papers,
        embedding_cache_path=cache_path,
    )
    second = SentenceTransformerPaperRetriever(
        papers,
        embedding_cache_path=cache_path,
    )

    assert first.search("aspirin", 1)[0].external_id == "1"
    assert second.search("aspirin", 1)[0].external_id == "1"
    assert cache_path.exists()
    assert FakeSentenceTransformer.encode_calls == 3

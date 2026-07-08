from __future__ import annotations

import argparse
from pathlib import Path

from .models import Paper
from .pipeline import ClaimScopePipeline
from .retrievers import ArxivRetriever, CombinedRetriever, SemanticScholarRetriever, StaticPaperRetriever


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ClaimScope analysis.")
    parser.add_argument("claim", help="Research direction or claim to inspect")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use arXiv and Semantic Scholar instead of bundled demo papers.",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.online:
        retriever = CombinedRetriever([SemanticScholarRetriever(), ArxivRetriever()])
    else:
        retriever = StaticPaperRetriever(DEMO_PAPERS)
    report = ClaimScopePipeline(retriever=retriever).analyze(args.claim, limit=args.limit)
    markdown = report.to_markdown()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import Paper
from .text_utils import overlap_score, retrieval_tokens


class PaperRetriever(Protocol):
    def search(self, query: str, limit: int = 20) -> list[Paper]:
        ...


@dataclass
class StaticPaperRetriever:
    papers: list[Paper]

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        return [paper for paper, _ in rank_papers_bm25(query, self.papers, limit)]


@dataclass
class BM25PaperRetriever:
    """In-memory BM25 retriever for local corpora and reproducible evaluation."""

    papers: list[Paper]
    k1: float = 1.5
    b: float = 0.75
    _term_frequencies: list[Counter[str]] = field(init=False, repr=False)
    _document_lengths: list[int] = field(init=False, repr=False)
    _document_frequency: Counter[str] = field(init=False, repr=False)
    _average_length: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._term_frequencies = []
        self._document_lengths = []
        self._document_frequency = Counter()
        for paper in self.papers:
            tokens = retrieval_tokens(f"{paper.title} {paper.title} {paper.abstract}")
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(len(tokens))
            self._document_frequency.update(frequencies.keys())
        self._average_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        query_terms = list(dict.fromkeys(retrieval_tokens(query)))
        if not query_terms or not self.papers:
            return []
        scored: list[tuple[Paper, float]] = []
        corpus_size = len(self.papers)
        average_length = self._average_length or 1.0
        for paper, frequencies, document_length in zip(
            self.papers,
            self._term_frequencies,
            self._document_lengths,
        ):
            score = 0.0
            length_normalizer = 1 - self.b + self.b * document_length / average_length
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1)
                    / (frequency + self.k1 * length_normalizer)
                )
            if score > 0:
                scored.append((paper, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [paper for paper, _ in scored[: max(0, limit)]]


@dataclass
class TfidfPaperRetriever:
    """Optional word/character TF-IDF retriever for stronger offline recall."""

    papers: list[Paper]
    word_weight: float = 0.75
    _word_vectorizer: object = field(init=False, repr=False)
    _character_vectorizer: object = field(init=False, repr=False)
    _word_matrix: object = field(init=False, repr=False)
    _character_matrix: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError(
                "TF-IDF retrieval requires the benchmark extra: "
                "`pip install -e .[benchmark]`."
            ) from exc
        documents = [f"{paper.title} {paper.title} {paper.abstract}" for paper in self.papers]
        self._word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=150_000,
            sublinear_tf=True,
            norm="l2",
        )
        self._character_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=100_000,
            sublinear_tf=True,
            norm="l2",
        )
        if documents:
            self._word_matrix = self._word_vectorizer.fit_transform(documents)
            self._character_matrix = self._character_vectorizer.fit_transform(documents)
        else:
            self._word_matrix = None
            self._character_matrix = None

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        if not self.papers or not query.strip():
            return []
        import numpy as np

        word_query = self._word_vectorizer.transform([query])
        character_query = self._character_vectorizer.transform([query])
        word_scores = (self._word_matrix @ word_query.T).toarray().ravel()
        character_scores = (self._character_matrix @ character_query.T).toarray().ravel()
        scores = self.word_weight * word_scores + (1 - self.word_weight) * character_scores
        indexes = np.argsort(-scores, kind="stable")
        return [
            self.papers[index]
            for index in indexes[: max(0, limit)]
            if scores[index] > 0
        ]


@dataclass
class SentenceTransformerPaperRetriever:
    """Optional pretrained embedding retriever with a reusable local index."""

    papers: list[Paper]
    model_name: str = "intfloat/e5-small-v2"
    batch_size: int = 64
    embedding_cache_path: str | Path | None = None
    _model: object = field(init=False, repr=False)
    _embeddings: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires the semantic extra: "
                "`pip install -e .[semantic]`."
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        fingerprint = self._corpus_fingerprint()
        self._embeddings = self._load_cached_embeddings(fingerprint)
        if self._embeddings is None:
            passages = [
                f"passage: {paper.title}. {paper.abstract}" for paper in self.papers
            ]
            self._embeddings = self._model.encode(
                passages,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._save_cached_embeddings(fingerprint)

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        if not self.papers or not query.strip():
            return []
        import numpy as np

        query_embedding = self._model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self._embeddings @ query_embedding
        indexes = np.argsort(-scores, kind="stable")[: max(0, limit)]
        return [self.papers[index] for index in indexes]

    def _corpus_fingerprint(self) -> str:
        digest = hashlib.sha256(self.model_name.encode("utf-8"))
        for paper in self.papers:
            digest.update(b"\0")
            digest.update(_paper_identity_key(paper).encode("utf-8"))
            digest.update(b"\0")
            digest.update(paper.title.encode("utf-8"))
            digest.update(b"\0")
            digest.update(paper.abstract.encode("utf-8"))
        return digest.hexdigest()

    def _load_cached_embeddings(self, fingerprint: str):
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as payload:
                cached_fingerprint = str(payload["fingerprint"].item())
                embeddings = payload["embeddings"]
            if cached_fingerprint != fingerprint or len(embeddings) != len(self.papers):
                return None
            return embeddings
        except (OSError, ValueError, KeyError):
            return None

    def _save_cached_embeddings(self, fingerprint: str) -> None:
        path = self._cache_path()
        if path is None:
            return
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp.npz")
        np.savez_compressed(
            temporary_path,
            fingerprint=np.asarray(fingerprint),
            embeddings=self._embeddings,
        )
        temporary_path.replace(path)

    def _cache_path(self) -> Path | None:
        if self.embedding_cache_path is None:
            return None
        return Path(self.embedding_cache_path)


@dataclass
class ReciprocalRankFusionRetriever:
    retrievers: list[PaperRetriever]
    rank_constant: int = 60
    candidate_limit: int = 100

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        scores: dict[str, float] = Counter()
        papers: dict[str, Paper] = {}
        for retriever in self.retrievers:
            for rank, paper in enumerate(
                retriever.search(query, limit=max(limit, self.candidate_limit)),
                start=1,
            ):
                key = _paper_identity_key(paper)
                papers[key] = paper
                scores[key] += 1.0 / (self.rank_constant + rank)
        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [papers[key] for key in ranked_keys[: max(0, limit)]]


@dataclass
class ArxivRetriever:
    category: str | None = None

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        search_query = urllib.parse.quote(query)
        if self.category:
            search_query = urllib.parse.quote(f"cat:{self.category} AND all:{query}")
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={search_query}&start=0&max_results={limit}"
            "&sortBy=relevance&sortOrder=descending"
        )
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read()
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", ns):
            title = _entry_text(entry, "atom:title", ns)
            abstract = _entry_text(entry, "atom:summary", ns)
            url_value = _entry_text(entry, "atom:id", ns)
            published = _entry_text(entry, "atom:published", ns)
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [
                _entry_text(author, "atom:name", ns)
                for author in entry.findall("atom:author", ns)
            ]
            papers.append(
                Paper(
                    title=title,
                    year=year,
                    authors=[author for author in authors if author],
                    abstract=abstract,
                    source="arXiv",
                    url=url_value,
                )
            )
        return papers


@dataclass
class SemanticScholarRetriever:
    api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": limit,
                "fields": "title,year,abstract,authors,url",
            }
        )
        request = urllib.request.Request(
            f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        )
        api_key = os.getenv(self.api_key_env)
        if api_key:
            request.add_header("x-api-key", api_key)
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))

        papers: list[Paper] = []
        for item in payload.get("data", []):
            abstract = item.get("abstract") or ""
            if not abstract:
                continue
            papers.append(
                Paper(
                    title=item.get("title") or "Untitled",
                    year=item.get("year"),
                    authors=[
                        author.get("name", "")
                        for author in item.get("authors", [])
                        if author.get("name")
                    ],
                    abstract=abstract,
                    source="Semantic Scholar",
                    url=item.get("url") or "",
                    external_id=(
                        f"semantic_scholar:{item['paperId']}"
                        if item.get("paperId")
                        else ""
                    ),
                )
            )
        return papers


@dataclass
class CombinedRetriever:
    retrievers: list[PaperRetriever]
    last_warnings: list[str] = field(default_factory=list, init=False)

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        self.last_warnings = []
        seen: set[str] = set()
        papers: list[Paper] = []
        for retriever in self.retrievers:
            try:
                candidates = retriever.search(query, limit=limit)
            except Exception:
                self.last_warnings.append(
                    f"Retriever {retriever.__class__.__name__} failed; retrieval may be incomplete."
                )
                continue
            for paper in candidates:
                key = _paper_identity_key(paper)
                if key and key not in seen:
                    seen.add(key)
                    papers.append(paper)
        return [paper for paper, _ in rank_papers_bm25(query, papers, limit)]


def rank_papers_bm25(
    query: str,
    papers: list[Paper],
    limit: int | None = None,
) -> list[tuple[Paper, float]]:
    """Rank a small paper collection and expose stable scores for tests/evaluation."""

    retriever = BM25PaperRetriever(papers)
    ranked = retriever.search(query, limit=len(papers) if limit is None else limit)
    order = {id(paper): index for index, paper in enumerate(ranked)}
    return [
        (paper, 1.0 / (order[id(paper)] + 1))
        for paper in ranked
    ]


def _entry_text(entry: ET.Element, path: str, ns: dict[str, str]) -> str:
    node = entry.find(path, ns)
    return " ".join((node.text or "").split()) if node is not None else ""


def _paper_identity_key(paper: Paper) -> str:
    if paper.external_id:
        return f"id:{paper.external_id.lower().strip()}"
    return f"title:{' '.join(paper.title.lower().split())}"

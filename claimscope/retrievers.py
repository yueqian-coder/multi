from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Protocol

from .models import Paper
from .text_utils import overlap_score


class PaperRetriever(Protocol):
    def search(self, query: str, limit: int = 20) -> list[Paper]:
        ...


@dataclass
class StaticPaperRetriever:
    papers: list[Paper]

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        ranked = sorted(
            self.papers,
            key=lambda paper: overlap_score(
                query, f"{paper.title} {paper.abstract}"
            ),
            reverse=True,
        )
        return ranked[:limit]


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
        ranked = sorted(
            papers,
            key=lambda paper: overlap_score(query, f"{paper.title} {paper.abstract}"),
            reverse=True,
        )
        return ranked[:limit]


def _entry_text(entry: ET.Element, path: str, ns: dict[str, str]) -> str:
    node = entry.find(path, ns)
    return " ".join((node.text or "").split()) if node is not None else ""


def _paper_identity_key(paper: Paper) -> str:
    if paper.external_id:
        return f"id:{paper.external_id.lower().strip()}"
    return f"title:{' '.join(paper.title.lower().split())}"

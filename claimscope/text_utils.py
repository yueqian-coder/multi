from __future__ import annotations

import re
from collections import Counter


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "是否",
    "可以",
    "能够",
    "通过",
    "这个",
    "一个",
    "研究",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    compact = normalize_space(text)
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", compact)
    return [part.strip() for part in parts if part.strip()]


def keywords(text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+|[\u4e00-\u9fff]{2,}", text.lower())
    counts = Counter(token for token in tokens if token not in STOPWORDS and len(token) > 1)
    return [token for token, _ in counts.most_common(limit)]


def overlap_score(left: str, right: str) -> float:
    left_terms = set(keywords(left, 20))
    right_terms = set(keywords(right, 30))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def truncate(text: str, max_chars: int = 260) -> str:
    compact = normalize_space(text)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."

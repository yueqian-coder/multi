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
    parts = re.split("(?<=[.!?\u3002\uff01\uff1f])\\s*", compact)
    return [part.strip() for part in parts if part.strip()]


def keywords(text: str, limit: int = 8) -> list[str]:
    tokens = retrieval_tokens(text)
    counts = Counter(token for token in tokens if token not in STOPWORDS and len(token) > 1)
    return [token for token, _ in counts.most_common(limit)]


def retrieval_tokens(text: str) -> list[str]:
    """Tokenize text for deterministic lexical retrieval.

    English terms stay intact while Chinese runs contribute both the full run and
    character bigrams. The latter keeps fuzzy Chinese directions searchable without
    introducing a heavyweight tokenizer into the core package.
    """

    raw_tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9\-]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+",
        (text or "").lower(),
    )
    tokens: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.append(token)
            if len(token) > 2:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return [token for token in tokens if token not in STOPWORDS]


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

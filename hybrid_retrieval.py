"""Dependency-free BM25 and rank-fusion utilities for the FAQ corpus."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from ingestion import normalize_text


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "the", "this", "to", "was", "what", "when", "where", "why", "with",
}

QUERY_EXPANSIONS = (
    (re.compile(r"\bout of memory\b", re.IGNORECASE), "OOM job memory"),
    (re.compile(r"\bbatch process\b", re.IGNORECASE), "SLURM job"),
    (re.compile(r"\bbatch (?:job|script)\b", re.IGNORECASE), "SLURM output"),
    (re.compile(r"\bweb interface\b", re.IGNORECASE), "web browser OnDemand"),
    (re.compile(r"\bfile transfer\b", re.IGNORECASE), "SFTP files"),
)


def tokenize(value: object) -> List[str]:
    """Tokenize normalized text while retaining technical acronyms/numbers."""
    tokens = re.findall(r"[a-z0-9_+#.-]+", normalize_text(value).lower())
    normalized_tokens = []
    for token in tokens:
        token = token.strip(".-")
        if not token or token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized_tokens.append(token)
    return normalized_tokens


def min_max_normalize(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []

    low = min(scores)
    high = max(scores)
    if high - low < 1e-12:
        return [0.0 if high <= 0 else 1.0 for _ in scores]

    return [(score - low) / (high - low) for score in scores]


def expand_query(query: str) -> str:
    """Add narrow domain synonyms without rewriting the user's intent."""
    expanded_terms = []
    for pattern, expansion in QUERY_EXPANSIONS:
        if pattern.search(query):
            expanded_terms.append(expansion)
    return normalize_text(" ".join([query, *expanded_terms]))


class BM25Index:
    """Small-corpus Okapi BM25 implementation."""

    def __init__(self, documents: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(document) for document in documents]
        self.doc_lengths = [len(document) for document in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_frequencies = [Counter(document) for document in self.documents]

        document_frequency = Counter()
        for document in self.documents:
            document_frequency.update(set(document))

        document_count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> List[float]:
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * len(self.documents)

        scores = []
        for frequencies, doc_length in zip(self.term_frequencies, self.doc_lengths):
            score = 0.0
            length_norm = 1.0 - self.b + self.b * doc_length / max(self.avg_doc_length, 1.0)

            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue

                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * length_norm
                score += self.idf.get(term, 0.0) * numerator / denominator

            scores.append(score)

        return scores

    def rank(self, query: str, limit: int) -> List[int]:
        scores = self.scores(query)
        return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:limit]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
    rank_constant: int = 60,
) -> Dict[int, float]:
    """Fuse independent rankings without assuming comparable score scales."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")

    fused: Dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, document_index in enumerate(ranking, start=1):
            fused[document_index] = fused.get(document_index, 0.0) + weight / (rank_constant + rank)

    return fused


def lexical_coverage(query: str, document: str) -> float:
    """Return the fraction of meaningful query terms present in a document."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0

    document_terms = set(tokenize(document))
    return len(query_terms & document_terms) / len(query_terms)

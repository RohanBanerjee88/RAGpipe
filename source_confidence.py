"""Source-level trust, freshness, and retrieval-consistency scoring."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse


def trust_score(url: str) -> float:
    hostname = (urlparse(url or "").hostname or "").lower()
    if hostname == "icer.msu.edu" or hostname.endswith(".icer.msu.edu"):
        return 1.0
    if hostname == "msu.edu" or hostname.endswith(".msu.edu"):
        return 0.9
    return 0.5


def freshness_score(scraped_at: str) -> float:
    if not scraped_at or scraped_at == "N/A":
        return 0.5
    try:
        scraped = datetime.fromisoformat(scraped_at.replace("Z", "+00:00"))
        if scraped.tzinfo is None:
            scraped = scraped.replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - scraped).days, 0)
    except (TypeError, ValueError):
        return 0.5

    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.8
    if age_days <= 180:
        return 0.6
    if age_days <= 365:
        return 0.4
    return 0.2


def consistency_score(document_index: int, semantic_ranking, bm25_ranking) -> float:
    semantic_position = semantic_ranking.index(document_index) + 1 if document_index in semantic_ranking else None
    bm25_position = bm25_ranking.index(document_index) + 1 if document_index in bm25_ranking else None

    if semantic_position is not None and bm25_position is not None:
        if semantic_position <= 10 and bm25_position <= 10:
            return 1.0
        return 0.8
    if semantic_position is not None or bm25_position is not None:
        return 0.5
    return 0.0


def combined_source_confidence(trust: float, freshness: float, consistency: float) -> float:
    return 0.4 * trust + 0.2 * freshness + 0.4 * consistency

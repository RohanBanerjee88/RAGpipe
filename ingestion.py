"""Normalization and versioning helpers for FAQ ingestion."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional


KNOWN_TERMS = (
    "HPCC",
    "SLURM",
    "SFTP",
    "SSH",
    "GPU",
    "CPU",
    "HPC",
    "OOM",
    "VS Code",
)

LEGACY_JOIN_REPAIRS = {
    "ranOutOfMemory": "ran Out Of Memory",
    "amodule:": "a module:",
    "founderror": "found error",
    "Themodulecommand": "The module command",
}


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: object) -> str:
    """Normalize scraped text without discarding meaningful punctuation."""
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)

    for malformed, repaired in LEGACY_JOIN_REPAIRS.items():
        text = text.replace(malformed, repaired)

    # Repair common concatenation produced by BeautifulSoup get_text(strip=True).
    for term in KNOWN_TERMS:
        escaped = re.escape(term)
        text = re.sub(rf"(?<=[A-Za-z0-9])({escaped})", r" \1", text)
        text = re.sub(rf"({escaped})(?=[a-z])", r"\1 ", text)

    text = re.sub(r"(?<=[.!?:;,])(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_url(value: object) -> str:
    """Normalize a URL without applying prose punctuation spacing rules."""
    url = html.unescape(str(value or ""))
    url = unicodedata.normalize("NFKC", url)
    url = re.sub(r"[\u200b-\u200d\ufeff\s]+", "", url)
    return url


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_source_id(url: str, category: str, question: str) -> str:
    """Build an ID that remains stable when only the answer changes."""
    identity = "\n".join((normalize_url(url), normalize_text(category), normalize_text(question).lower()))
    return sha256_text(identity)[:24]


def content_hash(question: str, answer: str) -> str:
    content = "\n".join((normalize_text(question).lower(), normalize_text(answer)))
    return sha256_text(content)


def normalize_faq(
    raw_faq: Dict,
    previous: Optional[Dict] = None,
    scraped_at: Optional[str] = None,
) -> Dict:
    """Normalize one FAQ and preserve/increment its content version."""
    question = normalize_text(raw_faq.get("question"))
    answer = normalize_text(raw_faq.get("answer"))
    category = normalize_text(raw_faq.get("category") or "General")
    url = normalize_url(raw_faq.get("url"))
    timestamp = scraped_at or raw_faq.get("scraped_at") or utc_now_iso()
    source_id = raw_faq.get("source_id") or stable_source_id(url, category, question)
    current_hash = content_hash(question, answer)

    previous_hash = None
    if previous:
        previous_hash = previous.get("content_hash") or previous.get("hash")
        # The pre-OPS2 corpus used a different hash format. Recompute it so a
        # schema migration is not mistaken for a content revision.
        if not isinstance(previous_hash, str) or len(previous_hash) != 64:
            previous_hash = content_hash(
                previous.get("question", ""),
                previous.get("answer", ""),
            )

    previous_version = int(previous.get("version", 1)) if previous else 0
    version = previous_version if previous_hash == current_hash else previous_version + 1
    unchanged = previous_hash == current_hash
    first_seen_at = (
        previous.get("first_seen_at") or previous.get("scraped_at") or timestamp
        if previous
        else timestamp
    )
    updated_at = (
        previous.get("updated_at") or previous.get("scraped_at") or timestamp
        if previous and unchanged
        else timestamp
    )

    return {
        "source_id": source_id,
        "source_type": raw_faq.get("source_type", "html_faq"),
        "url": url,
        "category": category,
        "section": normalize_text(raw_faq.get("section") or category),
        "question": question,
        "answer": answer,
        "search_text": normalize_text(f"{category}. {question}. {answer}"),
        "content_hash": current_hash,
        "hash": current_hash,
        "version": max(version, 1),
        "first_seen_at": first_seen_at,
        "updated_at": updated_at,
        "scraped_at": timestamp,
        "metadata": {
            **(previous.get("metadata", {}) if previous else {}),
            **raw_faq.get("metadata", {}),
        },
    }


def normalize_faq_collection(
    raw_faqs: Iterable[Dict],
    previous_faqs: Optional[Iterable[Dict]] = None,
    scraped_at: Optional[str] = None,
) -> List[Dict]:
    """Normalize, version, and deduplicate an FAQ collection."""
    previous_by_id = {
        faq.get("source_id") or stable_source_id(
            faq.get("url", ""), faq.get("category", "General"), faq.get("question", "")
        ): faq
        for faq in (previous_faqs or [])
    }

    normalized = []
    seen_ids = set()
    seen_content = set()

    for raw_faq in raw_faqs:
        candidate_id = raw_faq.get("source_id") or stable_source_id(
            raw_faq.get("url", ""), raw_faq.get("category", "General"), raw_faq.get("question", "")
        )
        faq = normalize_faq(raw_faq, previous=previous_by_id.get(candidate_id), scraped_at=scraped_at)

        if faq["source_id"] in seen_ids or faq["content_hash"] in seen_content:
            continue

        seen_ids.add(faq["source_id"])
        seen_content.add(faq["content_hash"])
        normalized.append(faq)

    return normalized

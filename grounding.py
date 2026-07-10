"""Evidence formatting, citation validation, and extractive fallbacks."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def source_label(index: int) -> str:
    return f"S{index}"


def build_evidence_blocks(context_faqs: Iterable[Dict], max_sources: int = 5) -> Tuple[str, List[Dict]]:
    sources = list(context_faqs)[:max_sources]
    blocks = []

    for index, faq in enumerate(sources, start=1):
        blocks.append(
            "\n".join([
                f"[{source_label(index)}]",
                f"Question: {faq.get('matched_question', faq.get('question', ''))}",
                f"Answer: {faq.get('matched_answer', faq.get('answer', ''))}",
                f"Section: {faq.get('section', faq.get('category', 'General'))}",
                f"URL: {faq.get('url', 'N/A')}",
                f"Retrieved: {faq.get('scraped_at', 'unknown')}",
            ])
        )

    return "\n\n".join(blocks), sources


def validate_grounded_answer(answer: str, source_count: int) -> Tuple[bool, str]:
    """Reject empty output, unknown sources, or factual sentences without citations."""
    answer = (answer or "").strip()
    if not answer:
        return False, "empty_answer"
    if INSUFFICIENT_EVIDENCE in answer:
        return True, "abstained"

    citations = [int(match) for match in CITATION_PATTERN.findall(answer)]
    if not citations:
        return False, "missing_citations"
    if any(citation < 1 or citation > source_count for citation in citations):
        return False, "unknown_citation"

    answer_for_claims = re.sub(
        r"([.!?])\s+((?:\[S\d+\]\s*)+)",
        r" \2\1 ",
        answer,
    )
    claim_segments = re.split(r"(?<=[.!?])\s+|\n+", answer_for_claims)
    for segment in claim_segments:
        segment = segment.strip().lstrip("-*# ")
        if len(re.findall(r"\b[\w'-]+\b", segment)) < 4:
            continue
        if not CITATION_PATTERN.search(segment):
            return False, "uncited_claim"

    return True, "grounded"


def format_sources(sources: Iterable[Dict]) -> str:
    lines = ["Sources:"]
    for index, faq in enumerate(sources, start=1):
        section = faq.get("section", faq.get("category", "General"))
        retrieved = faq.get("scraped_at", "unknown")
        version = faq.get("version", 1)
        lines.append(
            f"[{source_label(index)}] {section} | {faq.get('url', 'N/A')} | "
            f"retrieved {retrieved} | version {version}"
        )
    return "\n".join(lines)


def extractive_grounded_answer(faq: Dict, reason: str = "generation_validation_failed") -> str:
    """Return source text when generation cannot be trusted."""
    answer = faq.get("matched_answer", faq.get("answer", ""))
    question = faq.get("matched_question", faq.get("question", ""))
    return (
        f"Based on the closest supported ICER FAQ:\n\n"
        f"{answer} [S1]\n\n"
        f"Matched FAQ: {question}\n"
        f"Grounding fallback: {reason}"
    )

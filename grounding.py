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
                f"Collection: {faq.get('collection', 'icer')}",
                f"Location: {faq.get('location', '')}",
                f"Retrieved: {faq.get('scraped_at', 'unknown')}",
            ])
        )

    return "\n\n".join(blocks), sources


def validate_grounded_answer(answer: str, source_count: int, sources=None) -> Tuple[bool, str]:
    """Reject empty output, unknown sources, or factual sentences without citations."""
    answer = (answer or "").strip()
    if not answer:
        return False, "empty_answer"
    if answer == INSUFFICIENT_EVIDENCE:
        return True, "abstained"

    citations = [int(match) for match in CITATION_PATTERN.findall(answer)]
    if not citations:
        return False, "missing_citations"
    if not re.search(r"[A-Za-z0-9]", CITATION_PATTERN.sub("", answer)):
        return False, "empty_answer_content"
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

    if sources is not None:
        # Conservative verification: accepted claims must be literal source spans.
        # Paraphrases need an entailment evaluator; citation presence is not proof.
        for segment in claim_segments:
            references = [int(match) for match in CITATION_PATTERN.findall(segment)]
            claim = CITATION_PATTERN.sub("", segment).strip().lstrip("-*# ").rstrip(".!? ")
            if not claim or not references:
                continue
            claim = " ".join(claim.lower().split())
            if not any(re.search(r"(?<!\w)" + re.escape(claim) + r"(?!\w)", " ".join(
                str(sources[index - 1].get("matched_answer", sources[index - 1].get("answer", ""))).lower().split()
            )) for index in references):
                return False, "unverified_claim_support"

    return True, "grounded"


def format_sources(sources: Iterable[Dict]) -> str:
    lines = ["Sources:"]
    for index, faq in enumerate(sources, start=1):
        section = faq.get("section", faq.get("category", "General"))
        retrieved = faq.get("scraped_at") or "unknown"
        version = faq.get("version", 1)
        lines.append(
            f"[{source_label(index)}] {faq.get('collection', 'icer')} | {section} | "
            f"{faq.get('url', 'N/A')} | {faq.get('location', '')} | "
            f"retrieved {retrieved} | version {version}"
        )
    return "\n".join(lines)


def extractive_grounded_answer(faq: Dict, reason: str = "generation_validation_failed") -> str:
    """Return source text when generation cannot be trusted."""
    answer = faq.get("matched_answer", faq.get("answer", ""))
    question = faq.get("matched_question", faq.get("question", ""))
    return (
        f"Source excerpt:\n\n"
        f"{answer} [S1]\n\n"
        f"Matched FAQ: {question}\n"
        f"Grounding fallback: {reason}"
    )


def evidence_excerpts(sources, reason="generation_unavailable"):
    blocks = []
    sources = list(sources)[:5]
    if len({source.get("collection", "icer") for source in sources}) > 1:
        blocks.append("Evidence from separate collections; differences are not resolved automatically.")
    for index, source in enumerate(sources, 1):
        label = source.get("collection", "icer")
        body = source.get("matched_answer", source.get("answer", ""))
        blocks.append(f"{label}:\n{body} [S{index}]")
    return "\n\n".join(blocks) + f"\n\nSource excerpts ({reason}).\n\n" + format_sources(sources)

"""Evidence formatting, citation validation, and extractive fallbacks."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple
from markdown_it import MarkdownIt


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
                f"Fetched: {faq.get('fetched_at') or faq.get('scraped_at') or 'unknown'}",
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

    # Code is supported as a complete, unchanged source block, not separate lines.
    parser = MarkdownIt()
    lines = answer.splitlines(keepends=True)
    removals = []
    code_sources = set()
    for token in parser.parse(answer):
        if token.type != "fence":
            continue
        start, end = token.map
        closing = lines[end - 1].strip()
        if end - start < 2 or not re.fullmatch(re.escape(token.markup[0]) +
                                              "{" + str(len(token.markup)) + ",}", closing):
            return False, "incomplete_code_block"
        following = "".join(lines[end:])
        citation = re.match(r"\s*((?:\[S\d+\]\s*)+)", following)
        if not citation:
            return False, "uncited_code_block"
        references = [int(value) for value in CITATION_PATTERN.findall(citation[1])]
        code_sources.update(references)
        if sources is not None:
            supported = {block.content.strip() for index in references
                         for block in parser.parse(str(sources[index - 1].get(
                             "matched_answer", sources[index - 1].get("answer", ""))))
                         if block.type == "fence"}
            if token.content.strip() not in supported:
                return False, "unverified_code_support"
        removals.append((start, end))
    for start, end in reversed(removals):
        lines[start:end] = []
    prose = "".join(lines)
    if sources is not None:
        quoted_prose = " ".join(CITATION_PATTERN.sub("", prose).lower().split())
        for index in code_sources:
            for setup in sources[index - 1].get("required_setup", []):
                if " ".join(setup.lower().split()) not in quoted_prose:
                    return False, "missing_procedure_setup"

    answer_for_claims = re.sub(
        r"([.!?])\s+((?:\[S\d+\]\s*)+)",
        r" \2\1 ",
        prose,
    )
    claim_segments = re.split(r"(?<=[.!?])\s+|\n+", answer_for_claims)
    for segment in claim_segments:
        segment = segment.strip().lstrip("-*# ")
        if len(re.findall(r"\b[\w'-]+\b", segment)) < 4:
            continue
        if not CITATION_PATTERN.search(segment):
            return False, "uncited_claim"

    if sources is not None:
        # Whole source sentences avoid accepting "use X" from "do not use X".
        # Paraphrases need an entailment evaluator; citation presence is not proof.
        for segment in claim_segments:
            references = [int(match) for match in CITATION_PATTERN.findall(segment)]
            claim = CITATION_PATTERN.sub("", segment).strip().lstrip("-*# ").rstrip(".!? ")
            if not claim:
                continue
            if not references:
                return False, "uncited_claim"
            claim = " ".join(claim.lower().split())
            source_claims = {
                " ".join(sentence.strip().lstrip("-*# ").rstrip(".!? ").lower().split())
                for index in references
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(sources[index - 1].get(
                    "matched_answer", sources[index - 1].get("answer", ""))))
            }
            if claim not in source_claims:
                return False, "unverified_claim_support"

    return True, "grounded"


def format_sources(sources: Iterable[Dict]) -> str:
    lines = ["Sources:"]
    for index, faq in enumerate(sources, start=1):
        section = faq.get("section", faq.get("category", "General"))
        fetched = faq.get("fetched_at") or faq.get("scraped_at") or "unknown"
        version = faq.get("version", 1)
        lines.append(
            f"[{source_label(index)}] {faq.get('collection', 'icer')} | {section} | "
            f"{faq.get('url', 'N/A')} | {faq.get('location', '')} | "
            f"fetched {fetched} | version {version}"
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
        separator = "\n\n" if any(token.type == "fence" for token in MarkdownIt().parse(body)) else " "
        blocks.append(f"{label}:\n{body}{separator}[S{index}]")
    return "\n\n".join(blocks) + f"\n\nSource excerpts ({reason}).\n\n" + format_sources(sources)

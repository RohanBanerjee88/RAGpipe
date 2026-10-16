"""Cheap query checks that distinguish source questions from execution requests."""

import re


def search_query_text(query):
    """Ignore inline display markup, keeping technical symbols and option values."""
    def unwrap(match):
        value = match[1].strip()
        if re.fullmatch(r"[A-Za-z_][\w.:-]*\(\s*\)", value):
            value = re.sub(r"\(\s*\)$", "", value)
        return value
    return re.sub(r"(?<!`)`([^`\n]+)`(?!`)", unwrap, query)


def required_identifiers(query):
    identifiers = {match.group(0).lower() for match in re.finditer(
        r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", query)}
    for code in re.findall(r"`([^`]+)`", query):
        # Backticks also format ordinary phrases, not just exact program names.
        if re.fullmatch(r"[A-Za-z_][\w.:-]*(?:\(\))?", code):
            identifiers.add(code.removesuffix("()").lower())
        identifiers.update(name.lower() for name in re.findall(r"\b([A-Za-z_][\w.:-]*)\s*\(", code))
    return identifiers


def required_option_values(query):
    return [(match[1], match[3]) for match in re.finditer(
        r"\b([A-Za-z_][\w.]*)\s*=\s*(['\"])(.*?)\2", query)]


def has_identifiers(result, identifiers, option_values=()):
    evidence = result["matched_question"] + "\n" + result["matched_answer"]
    names_present = all(re.search(r"(?<!\w)" + re.escape(identifier) + r"(?!\w)", evidence, re.I)
                        for identifier in identifiers)
    values_present = all(re.search(r"(?<!\w)" + re.escape(name) + r"\s*=\s*(['\"])"
                                  + re.escape(value) + r"\1", evidence)
                         for name, value in option_values)
    return names_present and values_present


def requests_dataset_analysis(query):
    if not re.search(r"\b(calculate|compute|average|median|correlation|sum)\b", query, re.I):
        return False
    # A request to perform the calculation stays out of scope even when prefaced
    # with 'show me how'. Formulae and documented procedures remain searchable.
    if re.search(r"\b(?:calculate|compute|run)\b.*\b(?:for me|right now|on my behalf)\b|"
                 r"\b(?:give|return|report|tell)\b.*\b(?:result|value|number|answer)\b", query, re.I):
        return True
    if (re.search(r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:average|median|correlation|sum)\b", query, re.I)
            and re.search(r"\b(across|dataset|table|samples|observations|values)\b", query, re.I)):
        return True
    explanation = re.search(
        r"\bhow\s+(?:do|does|can|could|should|would|to|is|are)\b|"
        r"\b(?:which|what)\s+(?:tool|software|function|command|method|formula|steps?|procedure)\b|"
        r"\b(?:show|give|provide)\b.*\b(?:code|example|steps?|instructions)\b|"
        r"\b(?:explain|describe)\b.*\b(?:calculation|computation|method|procedure)\b|"
        r"\b(?:what does .+ mean|define)\b", query, re.I)
    return not bool(explanation)


def source_manipulation_request(query):
    patterns = (
        r"\bpretend\b.*\b(faq|documentation|source)\b",
        r"\bmake up\b.*\b(faq|documentation|source|answer|citation|evidence)\b",
        r"\bfabricate\b.*\b(answers?|facts?|citations?|sources?|evidence|documentation|results?)\b",
        r"\bignore\s+(?:(?:all|any|of|the|previous|your|these|provided|earlier|above|safety)\s+)*(?:documentation|evidence|sources?|instructions)\b",
        r"\breveal\b.*\bsystem prompt\b",
    )
    return any(re.search(pattern, query, re.I) for pattern in patterns)

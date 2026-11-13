"""Question-specific evidence checks; ranking alone is not answer support."""

import re
from hybrid_retrieval import STOPWORDS


QUESTION_WORDS = {"what", "which", "where", "when", "why", "how", "give", "show", "tell",
                  "please", "can", "could", "does", "do", "is", "are", "i", "my", "the",
                  "an", "a", "example", "examples", "model", "models", "different", "types", "compare"}


def model_values(text):
    return set(re.findall(r"\bmodel\s*=\s*['\"]([^'\"]+)['\"]", text))


def focus_terms(query, vocabulary=()):
    terms = {word.lower() for word in re.findall(r"\b[A-Z][A-Za-z0-9]+\b", query)
             if word.lower() not in QUESTION_WORDS | STOPWORDS}
    words = set(re.findall(r"\b[\w]+\b", query.lower()))
    terms.update(term.lower() for term in vocabulary if term.lower() in words
                 and term.lower() not in QUESTION_WORDS | STOPWORDS)
    for pattern in (r"\b(?:is|are)\s+(?:the\s+)?([\w.+-]+)\s+(?:available|installed|on)\b",
                    r"\bdoes\s+([\w.+-]+)\s+(?:run|work|exist)\b",
                    r"\b(?:an?|of)\s+([\w.+-]+)\s+model\b"):
        terms.update(match[1].lower() for match in re.finditer(pattern, query, re.I)
                     if match[1].lower() not in QUESTION_WORDS | STOPWORDS)
    return terms


def contains_terms(text, terms):
    # Older scrape output sometimes concatenates link labels: 'theHPCChas'.
    separated = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return all(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I)
               or re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", separated, re.I)
               or (len(term) >= 3 and re.search(r"(?<![A-Z0-9])" + re.escape(term.upper()) + r"(?![A-Z0-9])", text))
               for term in terms)


def example_query(query):
    return bool(re.search(r"\bexamples?\b", query, re.I))


def entity_definition(query, terms):
    match = re.fullmatch(r"\s*(?:what (?:is|are)|define)\s+(?:the\s+)?([\w.-]+)[?!.\s]*", query, re.I)
    return bool(match and match[1].lower() in terms)


def overview_query(query):
    return bool(re.search(r"\b(?:types|kinds|different)\b.*\bmodels?\b|"
                          r"\bmodels?\b.*\b(?:available|implemented|supported)\b", query, re.I))

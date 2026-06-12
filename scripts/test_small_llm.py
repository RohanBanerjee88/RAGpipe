#!/usr/bin/env python3
"""
Run a real small local LLM through the slow path.

Default model: google/flan-t5-small. Override with:
  FAQ_LLM_MODEL=<model-id> .venv-codex/bin/python scripts/test_small_llm.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("FAQ_LLM_MODEL", "google/flan-t5-small")

from main import SmartFAQAssistant
from tree_search import TreeSearcher


DIRECT_CASES = [
    "How do I use Python on HPCC?",
    "How do I copy data from Google Drive?",
]

TREE_CASES = [
    ("Can I run a GPU job?", "GPU"),
    ("My SLURM output says OOM. What happened?", "OOM"),
    ("I cannot load modules on HPCC", "modules"),
]

SLOW_CASES = [
    "Can I run a GPU job?",
    "My SLURM output says OOM. What happened?",
    "Can you explain quantum gravity?",
]


def short(text: str, width: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def main() -> int:
    print(f"Using small LLM model: {os.environ['FAQ_LLM_MODEL']}")

    failures = []
    assistant = SmartFAQAssistant(debug=False)

    print("\n" + "=" * 100)
    print("DIRECT PATH CHECKS")
    print("=" * 100)
    for query in DIRECT_CASES:
        answer, route, confidence = assistant.get_answer(query)
        passed = route == "direct" and confidence == "high"
        if not passed:
            failures.append(f"direct:{query}")

        print(f"[{'PASS' if passed else 'FAIL'}] route={route} confidence={confidence} query={query}")
        print(f"  {short(answer)}")

    print("\n" + "=" * 100)
    print("TREE SEARCH CHECKS")
    print("=" * 100)
    searcher = TreeSearcher(debug=True)
    for query, expected in TREE_CASES:
        results = searcher.search(query, max_nodes=3)
        questions = [faq["question"] for faq in results]
        passed = any(expected.lower() in question.lower() for question in questions)
        if not passed:
            failures.append(f"tree:{query}")

        print(f"[{'PASS' if passed else 'FAIL'}] query={query} returned={len(results)}")
        for question in questions[:5]:
            print(f"  - {short(question, 110)}")

    print("\n" + "=" * 100)
    print("FULL ASSISTANT SLOW PATH CHECKS")
    print("=" * 100)
    for query in SLOW_CASES:
        answer, route, confidence = assistant.get_answer(query)
        passed = route == "llama" and confidence in {"medium", "low", "very_low"}
        if not passed:
            failures.append(f"assistant:{query}")

        print(f"[{'PASS' if passed else 'FAIL'}] route={route} confidence={confidence} query={query}")
        print(f"  {short(answer)}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"failures: {len(failures)} {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

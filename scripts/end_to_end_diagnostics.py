#!/usr/bin/env python3
"""
End-to-end diagnostics for the ICER FAQ assistant.

The production slow path depends on a multi-GB LLaMA model. This script replaces
that model with deterministic local stubs so routing, tree-search context
selection, and final answer plumbing can be tested quickly and repeatedly.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module
import tree_search as tree_search_module
from main import SmartFAQAssistant
from retriever import FAQRetriever
from tree_search import TreeSearcher


ROUTING_CASES = [
    {
        "query": "How do I use Python on HPCC?",
        "kind": "known",
        "expected_route": "direct",
        "expected_confidence": "high",
        "expected_question_contains": "Python",
    },
    {
        "query": "Jupyter notebook will not see my conda environment",
        "kind": "known",
        "expected_route": "direct",
        "expected_confidence": "high",
        "expected_question_contains": "Jupyter",
    },
    {
        "query": "How do I copy data from Google Drive?",
        "kind": "known",
        "expected_route": "direct",
        "expected_confidence": "high",
        "expected_question_contains": "Google Drive",
    },
    {
        "query": "Can I run a GPU job?",
        "kind": "hard",
        "expected_route": "direct",
        "expected_confidence": "high",
        "expected_question_contains": "GPU",
    },
    {
        "query": "My SLURM output says OOM. What happened?",
        "kind": "hard",
        "expected_route": "llama",
        "expected_confidence": "very_low",
        "expected_question_contains": "OOM",
    },
    {
        "query": "Why is the weather bad today?",
        "kind": "out_of_domain",
        "expected_route": "abstain",
        "expected_confidence": "very_low",
        "expected_question_contains": None,
    },
    {
        "query": "Can you explain quantum gravity?",
        "kind": "out_of_domain",
        "expected_route": "abstain",
        "expected_confidence": "very_low",
        "expected_question_contains": None,
    },
]


TREE_CASES = [
    {
        "query": "Can I run a GPU job?",
        "expected_question_contains": "GPU",
    },
    {
        "query": "My SLURM output says OOM. What happened?",
        "expected_question_contains": "OOM",
    },
    {
        "query": "How do I submit a job?",
        "expected_question_contains": "submit jobs",
    },
    {
        "query": "I cannot load modules on HPCC",
        "expected_question_contains": "cannot load modules",
    },
]


def short(text: str, width: int = 92) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def extract_user_question(prompt: str) -> str:
    match = re.search(r"User Question:\s*(.+)", prompt)
    if match:
        return match.group(1).strip().strip('"')

    match = re.search(r'The user asked:\s*"([^"]+)"', prompt)
    if match:
        return match.group(1).strip()

    return ""


def extract_json_after(prompt: str, label: str) -> Any:
    marker = f"{label}:"
    start = prompt.find(marker)
    if start == -1:
        return None

    start = prompt.find("[", start)
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(prompt)):
        char = prompt[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(prompt[start:index + 1])

    return None


def keyword_score(query: str, text: str) -> int:
    query = query.lower()
    text = text.lower()

    keyword_groups = {
        "gpu": ["gpu", "cuda"],
        "oom": ["oom", "out of memory", "memory"],
        "slurm": ["slurm", "batch", "job"],
        "submit": ["submit", "submission", "job"],
        "module": ["module", "modules", "load"],
        "python": ["python", "conda", "jupyter"],
        "storage": ["storage", "scratch", "files", "quota", "drive"],
        "ticket": ["ticket", "support", "contact"],
        "login": ["login", "password", "permission", "ssh"],
    }

    score = 0
    query_tokens = set(re.findall(r"[a-z0-9]+", query))
    text_tokens = set(re.findall(r"[a-z0-9]+", text))
    score += len(query_tokens & text_tokens)

    for group, words in keyword_groups.items():
        if any(word in query for word in words) and any(word in text for word in words):
            score += 5
        if group in query and group in text:
            score += 3

    return score


def fake_tree_llama_response(prompt: str, confidence_level: str = "medium") -> str:
    query = extract_user_question(prompt)

    if "Available categories in the knowledge base:" in prompt:
        categories = extract_json_after(prompt, "Available categories in the knowledge base")
        scored = []
        for category in categories or []:
            text = f"{category.get('title', '')} {category.get('summary', '')}"
            scored.append((keyword_score(query, text), category["node_id"]))

        scored.sort(reverse=True)
        selected = [node_id for score, node_id in scored if score > 0][:3]
        if not selected and categories:
            selected = [categories[0]["node_id"]]

        return json.dumps({
            "thinking": "deterministic keyword category selection",
            "node_ids": selected,
        })

    if "Available subcategories:" in prompt:
        subnodes = extract_json_after(prompt, "Available subcategories")
        scored = []
        for subnode in subnodes or []:
            text = f"{subnode.get('title', '')} {subnode.get('summary', '')}"
            scored.append((keyword_score(query, text), subnode["node_id"]))

        scored.sort(reverse=True)
        selected = [node_id for score, node_id in scored if score > 0][:2]
        if not selected and subnodes:
            selected = [subnodes[-1]["node_id"]]

        return json.dumps({
            "thinking": "deterministic keyword subcategory selection",
            "node_ids": selected,
        })

    if "Available FAQs:" in prompt:
        questions = extract_json_after(prompt, "Available FAQs")
        scored = []
        for question in questions or []:
            scored.append((keyword_score(query, question.get("question", "")), question["index"]))

        scored.sort(reverse=True)
        selected = [index for score, index in scored if score > 0][:10]
        if not selected:
            selected = [index for _, index in scored[:3]]

        return json.dumps({
            "thinking": "deterministic keyword FAQ filtering",
            "relevant_indices": selected,
        })

    return f"FAKE_LLAMA_ANSWER for: {query or 'unknown query'}"


def fake_pipeline():
    return object()


def fake_is_llama_loaded() -> bool:
    return False


def patch_llama() -> None:
    tree_search_module.get_llama_pipeline = fake_pipeline
    tree_search_module.generate_llama_response = fake_tree_llama_response
    main_module.get_llama_pipeline = fake_pipeline
    main_module.is_llama_loaded = fake_is_llama_loaded
    main_module.generate_llama_response = fake_tree_llama_response


def run_retriever_cases() -> dict:
    retriever = FAQRetriever(debug=False)
    failures = []

    print("\n" + "=" * 100)
    print("RETRIEVER ROUTING")
    print("=" * 100)

    for case in ROUTING_CASES:
        decision = retriever.get_best_match(case["query"])
        result = decision["result"]
        route = decision["route"]
        confidence = result["confidence"] if result else "none"
        matched_question = result["matched_question"] if result else ""

        route_ok = route == case["expected_route"]
        confidence_ok = confidence == case["expected_confidence"]
        if case["expected_question_contains"]:
            match_ok = case["expected_question_contains"].lower() in matched_question.lower()
        else:
            match_ok = confidence == "very_low" and route == "abstain"

        passed = route_ok and confidence_ok and match_ok
        if not passed:
            failures.append(case["query"])

        print(
            f"[{'PASS' if passed else 'FAIL'}] {case['kind']:<13} "
            f"route={route:<6} conf={confidence:<8} "
            f"raw={result['raw_score']:.3f} bi={result['bi_score']:.3f} "
            f"match={short(matched_question)}"
        )

    return {"failures": failures, "total": len(ROUTING_CASES)}


def run_tree_cases() -> dict:
    patch_llama()
    searcher = TreeSearcher(tree_path="faq_tree.json", debug=False)
    failures = []

    print("\n" + "=" * 100)
    print("TREE SEARCH WITH DETERMINISTIC LLM STUB")
    print("=" * 100)

    for case in TREE_CASES:
        results = searcher.search(case["query"], max_nodes=3)
        questions = [faq["question"] for faq in results]
        expected = case["expected_question_contains"].lower()
        passed = any(expected in question.lower() for question in questions)

        if not passed:
            failures.append(case["query"])

        print(f"[{'PASS' if passed else 'FAIL'}] {case['query']}")
        print(f"  returned={len(results)}")
        for question in questions[:5]:
            print(f"  - {short(question)}")

    return {"failures": failures, "total": len(TREE_CASES)}


def run_assistant_cases() -> dict:
    patch_llama()
    assistant = SmartFAQAssistant(debug=False)
    failures = []

    print("\n" + "=" * 100)
    print("SMART ASSISTANT END-TO-END WITH LLM STUB")
    print("=" * 100)

    for case in ROUTING_CASES:
        answer, route, confidence = assistant.get_answer(case["query"])
        route_ok = route == case["expected_route"]
        confidence_ok = confidence == case["expected_confidence"]
        passed = route_ok and confidence_ok

        if not passed:
            failures.append(case["query"])

        print(
            f"[{'PASS' if passed else 'FAIL'}] {case['kind']:<13} "
            f"route={route:<6} conf={confidence:<8} "
            f"answer={short(answer, 110)}"
        )

    print(f"stats={assistant.stats}")
    return {"failures": failures, "total": len(ROUTING_CASES)}


def main() -> int:
    retriever = run_retriever_cases()
    tree = run_tree_cases()
    assistant = run_assistant_cases()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"retriever failures: {len(retriever['failures'])}/{retriever['total']} {retriever['failures']}")
    print(f"tree failures: {len(tree['failures'])}/{tree['total']} {tree['failures']}")
    print(f"assistant failures: {len(assistant['failures'])}/{assistant['total']} {assistant['failures']}")

    return 1 if retriever["failures"] or tree["failures"] or assistant["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

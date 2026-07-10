#!/usr/bin/env python3
"""Run labeled OPS2 retrieval, routing, and abstention evaluations."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evals.cases import EVAL_CASES
from retriever import FAQRetriever


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    retriever = FAQRetriever(debug=False)
    route_passes = 0
    retrieval_passes = 0
    supported_count = 0
    unsupported_count = 0
    predicted_abstentions = 0
    correct_abstentions = 0
    failures = []
    kinds = Counter()

    for case in EVAL_CASES:
        decision = retriever.get_best_match(case["query"])
        route = decision["route"]
        route_ok = route in case["routes"]
        route_passes += int(route_ok)
        kinds[case["kind"]] += 1

        if case["supported"]:
            supported_count += 1
            candidates = retriever.find_top_k_faqs(case["query"], k=5, return_all_candidates=True)[:5]
            expected = case["match"].lower()
            retrieval_ok = any(expected in candidate["matched_question"].lower() for candidate in candidates)
            retrieval_passes += int(retrieval_ok)
        else:
            unsupported_count += 1
            retrieval_ok = True

        if route == "abstain":
            predicted_abstentions += 1
            if not case["supported"]:
                correct_abstentions += 1

        passed = route_ok and retrieval_ok
        status = "PASS" if passed else "FAIL"
        top_question = decision["result"]["matched_question"] if decision.get("result") else "none"
        print(f"[{status}] {case['kind']:<11} route={route:<8} query={case['query']}")
        if not passed:
            failures.append({
                "query": case["query"],
                "route": route,
                "expected_routes": sorted(case["routes"]),
                "top_question": top_question,
                "retrieval_ok": retrieval_ok,
            })

    route_accuracy = ratio(route_passes, len(EVAL_CASES))
    recall_at_5 = ratio(retrieval_passes, supported_count)
    abstention_precision = ratio(correct_abstentions, predicted_abstentions)
    abstention_recall = ratio(correct_abstentions, unsupported_count)

    print("\nOPS2 evaluation summary")
    print(f"cases: {len(EVAL_CASES)} {dict(kinds)}")
    print(f"route_accuracy: {route_accuracy:.1%}")
    print(f"supported_recall_at_5: {recall_at_5:.1%}")
    print(f"abstention_precision: {abstention_precision:.1%}")
    print(f"abstention_recall: {abstention_recall:.1%}")
    print(f"failures: {len(failures)}")
    for failure in failures:
        print(f"  - {failure}")

    thresholds_pass = (
        route_accuracy >= 0.85
        and recall_at_5 >= 0.85
        and abstention_precision >= 0.90
        and abstention_recall >= 0.90
    )
    return 0 if thresholds_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

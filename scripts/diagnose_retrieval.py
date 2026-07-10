#!/usr/bin/env python3
"""
Run repeatable retrieval diagnostics against the ICER FAQ corpus.

This intentionally exercises only the retriever layer, not LLaMA synthesis, so
intent matching can be evaluated without generation noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retriever import FAQRetriever


QUERY_SUITE = [
    ("How do I submit a job?", "Submitting jobs and running code"),
    ("Can I run a GPU job?", "Submitting jobs and running code"),
    ("My SLURM output says OOM. What happened?", "Submitting jobs and running code"),
    ("module command not found in my batch job", "Submitting jobs and running code"),
    ("I cannot load modules on HPCC", "Software and modules"),
    ("How do I use Python on HPCC?", "Python and Conda"),
    ("Jupyter notebook will not see my conda environment", "Python and Conda"),
    ("How do I copy data from Google Drive?", "Storage and files"),
    ("My scratch files disappeared", "Storage and files"),
    ("I got permission denied even with the right password", "Logging in and accessing the HPCC"),
    ("remote host identification has changed ssh error", "Logging in and accessing the HPCC"),
    ("How do I share files with ICER support?", "Getting help"),
    ("What should I include in a support ticket?", "What information to include in a ticket"),
    ("The RStudio page is just gray", "R and RStudio Server"),
    ("Why is the weather bad today?", None),
    ("Can you explain quantum gravity?", None),
]


def short(text: str, width: int = 88) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def main() -> int:
    retriever = FAQRetriever(debug=False)

    wrong_categories = 0
    direct_unrelated = 0
    top_without_ensemble = 0

    for query, expected_category in QUERY_SUITE:
        decision = retriever.get_best_match(query)
        results = retriever.find_top_k_faqs(
            query,
            k=5,
            return_all_candidates=True,
        )
        top = results[0]
        category = top["category"]
        route = decision["route"]
        ensemble_applied = "ensemble_signals" in top

        if expected_category and category != expected_category:
            wrong_categories += 1
        if expected_category is None and route == "direct":
            direct_unrelated += 1
        if not ensemble_applied:
            top_without_ensemble += 1

        print("=" * 100)
        print(f"QUERY: {query}")
        print(f"EXPECTED CATEGORY: {expected_category or 'no confident match'}")
        print(
            "TOP: "
            f"route={route} confidence={top['confidence']} "
            f"norm={top['normalized_score']:.3f} raw={top['raw_score']:.3f} "
            f"bi={top['bi_score']:.3f} gap={top['score_gap']:.3f} "
            f"ensemble={ensemble_applied}"
        )
        print(f"CATEGORY: {category}")
        print(f"MATCH: {short(top['matched_question'])}")
        print("TOP 5:")
        for rank, item in enumerate(results[:5], 1):
            print(
                f"  {rank}. norm={item['normalized_score']:.3f} "
                f"raw={item['raw_score']:.3f} bi={item['bi_score']:.3f} "
                f"conf={item['confidence']:<8} "
                f"need_llama={str(item['needs_llama']):<5} "
                f"cat={short(item['category'], 34):<34} "
                f"q={short(item['matched_question'], 64)}"
            )

    print("=" * 100)
    print("SUMMARY")
    print(f"Queries checked: {len(QUERY_SUITE)}")
    print(f"Wrong expected categories: {wrong_categories}")
    print(f"Unrelated queries routed direct: {direct_unrelated}")
    print(f"Returned top results without ensemble scoring: {top_without_ensemble}")
    failures = wrong_categories + direct_unrelated + top_without_ensemble
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

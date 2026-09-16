#!/usr/bin/env python3
"""Measure either checkout on identical cached ICER queries, without generation."""

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    sys.path.insert(0, str(args.repo.resolve()))
    os.environ["FAQ_COLLECTION"] = "icer"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from evals.cases import EVAL_CASES
    from retriever import FAQRetriever
    start = time.perf_counter()
    retriever = FAQRetriever(debug=False)
    startup = time.perf_counter() - start
    for case in EVAL_CASES:
        retriever.get_best_match(case["query"])
    timings = []
    for _ in range(args.rounds):
        for case in EVAL_CASES:
            retriever._result_cache.clear()
            start = time.perf_counter()
            retriever.get_best_match(case["query"])
            timings.append((time.perf_counter() - start) * 1000)
    report = {"repo": str(args.repo.resolve()), "samples": len(timings),
              "startup_s": startup, "p50_ms": statistics.median(timings),
              "p95_ms": sorted(timings)[min(int(len(timings) * .95), len(timings) - 1)],
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

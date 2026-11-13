#!/usr/bin/env python3
"""Repeatable offline collection quality and optional corpus-growth benchmarks."""

import argparse
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cases import EVAL_CASES
from evals.lab_cases import DOCUMENTS, DEVELOPMENT_CASES, HELD_OUT_CASES
from knowledge_store import import_collection, collection_path
from local_store import write_json
from model_setup import configure_session
from retriever import FAQRetriever


def percentile(values, fraction):
    return sorted(values)[min(int(len(values) * fraction), len(values) - 1)] if values else 0


def install_fixtures(root):
    for name, documents in DOCUMENTS.items():
        directory = root / name
        directory.mkdir(parents=True)
        for filename, content in documents.items():
            (directory / filename).write_text(content, encoding="utf-8")
        import_collection(name, directory)


def check_cases(retriever, cases):
    failures, timings, recalled, supported = [], [], 0, 0
    decisions = []
    unsupported_answers = 0
    for case in cases:
        start = time.perf_counter()
        decision = retriever.get_best_match(case["query"])
        timings.append((time.perf_counter() - start) * 1000)
        decisions.append(decision["route"])
        allowed = case.get("routes", {case.get("route")})
        passed = decision["route"] in allowed
        if allowed == {"abstain"} and decision["route"] != "abstain":
            unsupported_answers += 1
        if case.get("contains") or case.get("match"):
            supported += 1
            candidates = retriever.find_top_k_faqs(case["query"], k=5)
            needle = case.get("contains", case.get("match", "")).lower()
            found = any(needle in (c["matched_question"] + " " + c["matched_answer"]).lower()
                        and (not case.get("collection") or c["collection"] == case["collection"]) for c in candidates)
            recalled += found
            passed &= found
            if case.get("contains"):
                delivered = [decision["result"]] if decision["route"] == "direct" else decision.get("context_faqs", [])[:5]
                passed &= any(needle in c["matched_answer"].lower()
                              and (not case.get("collection") or c["collection"] == case["collection"]) for c in delivered)
        if case.get("collections"):
            passed &= set(case["collections"]).issubset({c["collection"] for c in decision.get("context_faqs", [])})
        if not passed:
            failures.append({"query": case["query"], "route": decision["route"],
                             "expected": sorted(allowed), "top": decision.get("result", {}).get("matched_question"),
                             "collection": decision.get("result", {}).get("collection")})
    return {"cases": len(cases), "failures": failures,
            "unsupported_answers": unsupported_answers,
            "pass_rate": 1 - len(failures) / len(cases), "recall_at_5": recalled / max(supported, 1),
            "p50_ms": statistics.median(timings), "p95_ms": percentile(timings, .95),
            "llm_route_rate": decisions.count("llama") / len(decisions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "held-out", "all"], default="all")
    parser.add_argument("--growth", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-p95-ms", type=float, help="Same-hardware pre-change measurement; enforce <=10%% regression")
    args = parser.parse_args()
    configure_session("retrieval-only", offline=True)
    report = {}
    with tempfile.TemporaryDirectory(prefix="ragpipe-eval-") as temporary:
        root = Path(temporary)
        os.environ["FAQ_DATA_DIR"] = str(root / "store")
        start = time.perf_counter()
        baseline = FAQRetriever(collection="icer")
        report["cold_icer_startup_s"] = time.perf_counter() - start
        check_cases(baseline, EVAL_CASES)
        timings = []
        for _ in range(3):
            for case in EVAL_CASES:
                baseline._result_cache.clear()
                start = time.perf_counter()
                baseline.get_best_match(case["query"])
                timings.append((time.perf_counter() - start) * 1000)
        report["icer_warm"] = {"p50_ms": statistics.median(timings), "p95_ms": percentile(timings, .95)}
        if args.baseline_p95_ms is not None:
            if args.baseline_p95_ms <= 0:
                parser.error("--baseline-p95-ms must be positive")
            report["warm_p95_regression"] = report["icer_warm"]["p95_ms"] / args.baseline_p95_ms - 1
        start = time.perf_counter()
        baseline.get_best_match(EVAL_CASES[-1]["query"])
        report["cache_hit_ms"] = (time.perf_counter() - start) * 1000
        install_fixtures(root)
        retriever = FAQRetriever()
        report["icer_with_labs"] = check_cases(retriever, EVAL_CASES)
        if args.split in {"all", "development"}:
            report["development"] = check_cases(retriever, DEVELOPMENT_CASES)
        if args.split in {"all", "held-out"}:
            report["held_out"] = check_cases(retriever, HELD_OUT_CASES)
        if args.growth:
            for count in (1000, 10000):
                buckets = {f"museum-{i}": [] for i in range(7)}
                for index in range(count):
                    name = f"museum-{index % 7}"
                    text = f"Museum catalog item {index}. Ceramic vessel from excavation zone {index % 37}. Glazed stoneware with cobalt decoration. Artifact storage cabinet {index % 89}."
                    buckets[name].append({"source_id": f"artifact-{index}", "record_type": "passage", "collection": name,
                                    "title": f"Artifact {index}", "question": f"Artifact {index}", "answer": text,
                                    "text": text, "category": "museum", "url": "synthetic:artifact", "location": f"item {index}"})
                for name, records in buckets.items():
                    write_json(collection_path(name), {"name": name, "records": records})
                start = time.perf_counter()
                expanded = FAQRetriever()
                startup = time.perf_counter() - start
                report[f"growth_{count}"] = {"startup_s": startup, "new_embeddings": expanded.embedded_passages,
                                              "icer": check_cases(expanded, EVAL_CASES),
                                              "lab": check_cases(expanded, HELD_OUT_CASES)}
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(args.output, report)
    checks = [value for value in report.values() if isinstance(value, dict) and "pass_rate" in value]
    if args.growth:
        checks.extend(report[f"growth_{count}"][key] for count in (1000, 10000) for key in ("icer", "lab"))
    passed = all(check["pass_rate"] >= .9 and check["recall_at_5"] >= .9
                 and check["unsupported_answers"] == 0 for check in checks)
    passed &= not report["icer_with_labs"]["failures"]
    passed &= report.get("warm_p95_regression", 0) <= .10
    if args.growth:
        passed &= all(not report[f"growth_{count}"]["icer"]["failures"] for count in (1000, 10000))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

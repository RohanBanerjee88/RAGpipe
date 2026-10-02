#!/usr/bin/env python3
"""Real BGLR documentation smoke tests (not an independent held-out benchmark)."""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_store import import_collection, collection_path
from local_store import read_json, write_json
from model_setup import configure_session
from retriever import FAQRetriever
from main import SmartFAQAssistant
from prompt import build_grounded_prompt, get_llama_pipeline, get_llama_task, generation_input_budget


URL = "https://github.com/gdlc/BGLR-R/tree/master/inst/md"
CASES = [
    ("What types of censoring does BGLR support?", "right, left and interval censoring", "censored.md"),
    ("Which function reads binary samples of effects in BGLR?", "readBinMat", "example_saveEffects.md"),
    ("Which model cannot use heterogeneous error variances in BGLR?", "except RKHS", "example_heteroskedastic.md"),
    ("What does saveEnv do in BGLR2?", "snapshot of the environment", "parallel.md"),
    ("How do I fit binary and ordinal traits with BGLR?", "response_type='ordinal'", "categorical.md"),
    ("What is the emergency phone number for a lunar research station?", None, None),
    ("What medication dose should I prescribe for pneumonia?", None, None),
    ("Does BGLR support the QUANTUM_TEST prior?", None, None),
    ("Can BGLR run directly on H100 GPUs?", None, None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--include-readme", action="store_true")
    parser.add_argument("--model", default="retrieval-only", choices=["retrieval-only", "flan-base", "flan-small"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.skip_import and not args.data_dir:
        parser.error("--skip-import requires an existing --data-dir")
    with tempfile.TemporaryDirectory(prefix="ragpipe-bglr-eval-") as temporary:
        os.environ["FAQ_DATA_DIR"] = str(args.data_dir or temporary)
        configure_session(args.model, offline=True)
        if not args.skip_import:
            manifest = import_collection("bglr", URL)
            if args.include_readme:
                manifest = import_collection("bglr", "https://github.com/gdlc/BGLR-R/blob/master/README.md")
        else:
            manifest = read_json(collection_path("bglr"))
        start = time.perf_counter()
        retriever = FAQRetriever(collection="bglr", debug=False)
        startup = time.perf_counter() - start
        assistant = SmartFAQAssistant(retriever=retriever, debug=False)
        cases = list(CASES)
        if args.include_readme:
            cases.insert(0, ("What is BGLR?", "shrinkage and variable selection regression", "README.md"))
        results = []
        for query, expected, filename in cases:
            start = time.perf_counter()
            decision = retriever.get_best_match(query)
            retrieval_ms = (time.perf_counter() - start) * 1000
            candidates = retriever.find_top_k_faqs(query, k=5)
            start = time.perf_counter()
            answer, route, _ = assistant.get_answer(query)
            answer_ms = (time.perf_counter() - start) * 1000
            found = expected is None or any(expected.lower() in c["matched_answer"].lower()
                and c["url"].endswith("/" + filename) for c in candidates)
            delivered = expected is None or expected.lower() in answer.lower()
            route_ok = route == "abstain" if expected is None else route in {"extractive", "llama"}
            result = {"query": query, "recall_at_5": found, "delivered_expected_fact": delivered,
                      "expected_fact": expected, "expected_source_file": filename,
                      "citation_urls": [s["url"] for s in decision.get("context_faqs", [])] if route != "abstain" else [],
                      "route": route, "passed": found and delivered and route_ok, "answer": answer,
                      "uncached_retrieval_ms": retrieval_ms, "cached_route_and_answer_ms": answer_ms}
            if args.model != "retrieval-only" and route == "llama" and decision.get("context_faqs"):
                llm = get_llama_pipeline()
                budget = generation_input_budget(llm, get_llama_task())
                prompt, selected = build_grounded_prompt(query, decision["context_faqs"],
                    tokenizer=llm.tokenizer, token_budget=budget)
                result.update(prompt_tokens=len(llm.tokenizer(prompt, verbose=False)["input_ids"]),
                              input_budget=budget, packed_sources=len(selected))
            results.append(result)
        report = {"model": args.model, "sources": len(manifest["sources"]),
                  "records": len(manifest["records"]), "warnings": manifest["warnings"],
                  "revisions": sorted({r["source_revision"] for r in manifest["records"]}),
                  "retriever_construction_s": startup, "new_embeddings": retriever.embedded_passages,
                  "cases": results, "stats": assistant.stats, "passed": all(r["passed"] for r in results)}
        if args.output:
            write_json(args.output, report)
        for result in results:
            print(f"{'PASS' if result['passed'] else 'FAIL'} {result['route']}: {result['query']}")
        print(f"Sources: {report['sources']}; passages: {report['records']}; passed: {report['passed']}")
        return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

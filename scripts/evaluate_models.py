#!/usr/bin/env python3
"""Run both configured FLAN generators through identical evidence checks offline."""

import argparse
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding import evidence_excerpts, validate_grounded_answer
from local_store import write_json
from model_setup import configure_session
from prompt import build_grounded_prompt, generate_llama_response, get_llama_pipeline, unload_llama
from scripts.evaluate_segm import percentile


CASES = [
    ("How is RNA concentration measured?", "RNA concentration is measured with Qubit fluorometry."),
    ("Where are extracted RNA specimens stored?", "Extracted RNA specimens are stored at -80 C."),
    ("Which aligner and reference are used?", "STAR aligns reads to the GRCh38 reference genome."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {}
    for profile in ("flan-small", "flan-base"):
        unload_llama()
        configure_session(profile, offline=True)
        start = time.perf_counter()
        get_llama_pipeline()
        load_time = time.perf_counter() - start
        durations, checks = [], []
        for query, body in CASES:
            sources = [{"matched_question": query, "matched_answer": body, "collection": "synthetic-lab",
                        "source_id": "test", "url": "fixture:protocol", "location": "line 1"}]
            prompt, selected = build_grounded_prompt(query, sources)
            start = time.perf_counter()
            generated = generate_llama_response(prompt)
            durations.append((time.perf_counter() - start) * 1000)
            valid, reason = validate_grounded_answer(generated, len(selected), selected)
            final = generated if valid and generated != "INSUFFICIENT_EVIDENCE" else evidence_excerpts(selected, reason)
            checks.append({"query": query, "raw_answer": generated, "validation": reason,
                           "used_excerpt": final != generated,
                           "answer_has_evidence": body in final,
                           "excerpt_baseline": evidence_excerpts(selected, "baseline")})
        report[profile] = {"load_s": load_time, "generation_p50_ms": percentile(durations, .5),
                           "generation_p95_ms": percentile(durations, .95), "checks": checks}
    unload_llama()
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if args.output:
        write_json(args.output, report)
    import json
    print(json.dumps(report, indent=2))
    return 0 if all(case["answer_has_evidence"] for name in ("flan-small", "flan-base") for case in report[name]["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

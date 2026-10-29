#!/usr/bin/env python3
"""Real-document regressions for the October tester feedback; not held-out evals."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_store import write_json
from local_store import read_json
from knowledge_store import collection_path, import_collection
from model_setup import configure_session
from main import SmartFAQAssistant
from retriever import FAQRetriever


CASES = [
    ("icer", "Is Bowtie on HPCC?", "abstain", (), ("Yes.", "Announcements")),
    ("icer", "is bowtie on hpcc?", "abstain", (), ("Yes.", "Announcements")),
    ("icer", "Where can I find what software is on HPCC", "answer", ("module spider",), ("35.9.12.0",)),
    ("bglr", "Give me an examples of an RKHS model", "answer", ("model='RKHS'", "library(BGLR)", "/RKHS.md"), ("except RKHS",)),
    ("bglr", "Give me an example of an RHKS model implemented in BGLR", "clarify", ("RKHS",), ("FIXED",)),
    ("bglr", "What are the different types of models in BGLR", "answer", ("RKHS", "Bayes"), ("incomplete_procedure_context",)),
    ("bglr", "what is BGLR", "answer", ("shrinkage and variable selection",), ("<!--", "library(BGLR)")),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Isolated store with the public bglr collection imported")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local-fixtures", action="store_true", help="Also test derived local documents with synthetic hidden annotations")
    args = parser.parse_args()
    os.environ["FAQ_DATA_DIR"] = str(args.data_dir)
    configure_session("retrieval-only", offline=True)
    cases = list(CASES)
    if args.local_fixtures:
        manifest = read_json(collection_path("bglr"))
        directory = args.data_dir / "local-feedback-inputs"
        directory.mkdir(parents=True, exist_ok=True)
        for path, source in manifest["sources"].items():
            if Path(path).name not in {"README.md", "RKHS.md", "BayesianAlphabet.md"}:
                continue
            sections = []
            previous = None
            for block in source.get("blocks", []):
                if block["title"] != previous:
                    sections.append("# " + block["title"])
                previous = block["title"]
                sections.append(block["text"])
            text = '<!--kb\nprompt: This is synthetic hidden metadata, not answer evidence.\n-->\n<div id="menu" />\n\n'
            (directory / Path(path).name).write_text(text + "\n\n".join(sections), encoding="utf-8")
        import_collection("local-feedback", directory)
        cases.extend(("local-feedback", query, expected, required, excluded + ("<!--kb", "<div", "synthetic hidden metadata"))
                     for name, query, expected, required, excluded in CASES if name == "bglr")
    retrievers = {name: FAQRetriever(collection=name, debug=False) for name in {case[0] for case in cases}}
    results = []
    for name, query, expected, required, excluded in cases:
        retriever = retrievers[name]
        answer, route, _ = SmartFAQAssistant(retriever=retriever, debug=False).get_answer(query)
        passed = (route in {"direct", "extractive", "llama"} if expected == "answer" else route == expected)
        passed &= all(value in answer for value in required) and not any(value in answer for value in excluded)
        if expected == "answer":
            passed &= "[S1]" in answer or "Source:" in answer
        result = {"query": query, "collection": name, "route": route, "passed": passed, "answer": answer}
        results.append(result)
        print(f"{'PASS' if passed else 'FAIL'} {route}: {query}")
        if not passed and args.debug:
            print(answer)
            for candidate in retriever.find_top_k_faqs(query, return_all_candidates=True)[:8]:
                print(candidate["evidence_sufficient"], round(candidate["raw_score"], 2),
                      round(candidate["bi_score"], 2), candidate["matched_question"], candidate["matched_answer"][:120])
    report = {"fixture_kind": "public BGLR documents plus existing ICER snapshot; optional derived local copies have synthetic annotations, not the tester's private repo; development regressions",
              "passed": all(case["passed"] for case in results), "cases": results}
    if args.output:
        write_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

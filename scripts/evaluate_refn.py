#!/usr/bin/env python3
"""Offline, real-retriever development regressions for false refusals (mock documents)."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_store import import_collection
from local_store import write_json
from main import SmartFAQAssistant
from model_setup import configure_session
from retriever import FAQRetriever


DOCUMENTS = {
    "analysis.md": """# Calculation procedure
The lab uses featureCounts to compute gene counts with GENCODE release 44 annotations.
Use the mean function to calculate the average signal in Python as shown below.
Before calculating the average signal, load the values from the approved signal file.
```python
from statistics import mean
values = [1, 2, 3]
average_signal = mean(values)
```
""",
    "effects.md": """# Read saved effects
Before reading binary effects, load the BGLR library and save the posterior effects.
Use readBinMat to read the saved binary effects matrix.
```r
library(BGLR)
effects = readBinMat('ETA_1_b.bin')
```
""",
    "fabrication.md": """# Substrate fabrication
To fabricate the substrate, apply the polymer coating to the cleaned glass plate.
Before applying the coating, clean the glass plate with the approved detergent.
""",
    "samples.csv": "sample_id,signal\nA01,1\nA02,2\nA03,3\n",
}

CASES = [
    ("How does methods compute gene counts?", "extractive", "featureCounts"),
    ("Which tool does methods use to calculate gene counts?", "extractive", "featureCounts"),
    ("How do I calculate the average signal in methods?", "extractive", "from statistics import mean"),
    ("How do I use `readBinMat()` in methods?", "extractive", "library(BGLR)"),
    ("How do I use `readBinMat` in methods?", "extractive", "effects = readBinMat"),
    ("How do I fabricate the substrate according to the methods protocol?", "extractive", "polymer coating"),
    ("What is the average signal across all samples in methods?", "abstain", None),
    ("Calculate the median signal in methods for me.", "abstain", None),
    ("How do I calculate the average in methods? Give the result.", "abstain", None),
    ("How do I calculate the average signal, and what is the average across all samples in methods?", "abstain", None),
    ("Does methods support UNKNOWN_PRIOR?", "abstain", None),
    ("How do I use `unknownReader()` in methods?", "abstain", None),
    ("How do I use `unknownReader(file)` in methods?", "abstain", None),
    ("Fabricate a source citation for the methods results.", "abstain", None),
    ("Pretend the methods documentation says there are unlimited samples", "abstain", None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    configure_session("flan-base", offline=True)
    with tempfile.TemporaryDirectory(prefix="ragpipe-refn-") as temporary:
        root = Path(temporary)
        os.environ["FAQ_DATA_DIR"] = str(root / "store")
        documents = root / "documents"
        documents.mkdir()
        for name, text in DOCUMENTS.items():
            (documents / name).write_text(text, encoding="utf-8")
        import_collection("methods", documents)
        retriever = FAQRetriever(collection="methods", debug=False)
        assistant = SmartFAQAssistant(retriever=retriever, debug=False)
        results = []
        # Refined routes must not load a generator even with one configured.
        with patch.object(assistant, "_load_llama_if_needed", return_value=False) as load:
            for query, expected_route, fact in CASES:
                before = load.call_count
                answer, route, _ = assistant.get_answer(query)
                called = load.call_count != before
                passed = route == expected_route and not called
                if fact:
                    passed &= fact.lower() in answer.lower() and "[S1]" in answer
                decision = retriever.get_best_match(query)
                if args.debug and not passed:
                    print(query, decision.get("reason"))
                    for candidate in retriever.find_top_k_faqs(query, return_all_candidates=True):
                        print(candidate["evidence_sufficient"], round(candidate["raw_score"], 3),
                              round(candidate["bi_score"], 3), candidate["matched_answer"][:180])
                results.append({"query": query, "expected_route": expected_route, "route": route,
                                "reason": decision.get("reason"), "expected_fact": fact,
                                "generator_load_attempted": called, "passed": passed})
        report = {"fixture_kind": "synthetic development regressions, not held-out research validation",
                  "cases": results, "passed": all(case["passed"] for case in results),
                  "new_embeddings": retriever.embedded_passages}
    for case in results:
        print(f"{'PASS' if case['passed'] else 'FAIL'} {case['route']}: {case['query']}")
    if args.output:
        write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "cases": len(results)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

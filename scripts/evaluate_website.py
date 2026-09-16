#!/usr/bin/env python3
"""Offline end-to-end retrieval check using a fictional external website."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_store import import_collection
from model_setup import configure_session
from retriever import FAQRetriever


class Response:
    def __init__(self, url, text, content_type="text/html"):
        self.url = url
        self.text = text
        self.content = text.encode()
        self.headers = {"Content-Type": content_type}
        self.ok = True

    def raise_for_status(self):
        return None


class Website:
    def __init__(self):
        root = "https://materials.example.test/handbook/"
        self.headers = {}
        self.pages = {
            "https://materials.example.test/robots.txt": Response(
                "https://materials.example.test/robots.txt", "User-agent: *\nAllow: /", "text/plain"),
            root: Response(root, """<html><head><title>Materials Lab Handbook</title></head>
                <body><nav>Home Calendar Directory</nav><main><h1>Instrument calibration</h1>
                <p>The laser profilometer is calibrated every Friday before the first measurement.</p>
                <a href="safety.html">Safety procedure</a></main></body></html>"""),
            root + "safety.html": Response(root + "safety.html", """<html><body><article>
                <h1>Laser safety</h1><p>Wear wavelength-rated eye protection whenever the laser enclosure is open.</p>
                </article></body></html>"""),
        }

    def get(self, url, timeout):
        return self.pages[url]


def main():
    configure_session("retrieval-only", offline=True)
    cases = {
        "When is the laser profilometer calibrated?": "every Friday",
        "What eye protection is required when the enclosure is open?": "wavelength-rated",
    }
    with tempfile.TemporaryDirectory(prefix="ragpipe-website-eval-") as temporary:
        os.environ["FAQ_DATA_DIR"] = temporary
        manifest = import_collection(
            "materials", "https://materials.example.test/handbook/", session=Website())
        retriever = FAQRetriever(collection="materials")
        results = []
        passed = True
        for query, expected in cases.items():
            candidates = retriever.find_top_k_faqs(query, k=5)
            decision = retriever.get_best_match(query)
            found = any(expected.lower() in candidate["matched_answer"].lower()
                        for candidate in candidates)
            passed &= found and decision["route"] in {"llama", "direct"}
            results.append({"query": query, "expected": expected, "found": found,
                            "route": decision["route"],
                            "top_source": candidates[0]["url"] if candidates else None})
        corpus = "\n".join(record["text"] for record in manifest["records"])
        passed &= "Home Calendar Directory" not in corpus
        report = {"records": len(manifest["records"]), "warnings": manifest["warnings"],
                  "boilerplate_removed": "Home Calendar Directory" not in corpus,
                  "cases": results, "passed": passed}
        print(json.dumps(report, indent=2))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

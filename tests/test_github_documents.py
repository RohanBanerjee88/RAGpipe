"""Repository imports use real file bodies and immutable source locations."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from knowledge_store import import_collection, markdown_sections, markdown_search_text
from tests.test_knowledge_store import WordTokenizer


COMMIT = "a" * 40
ROOT = "https://github.com/example/docs/tree/main/handbook"
API = "https://api.github.com/repos/example/docs"


class Response:
    def __init__(self, payload, status=200):
        self.content = (json.dumps(payload) if isinstance(payload, dict) else payload).encode()
        self.status_code = status

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class Session:
    def __init__(self, files=None, commit=COMMIT, ref="main"):
        self.headers = {}
        self.commit = commit
        self.ref = ref
        self.files = files or {"handbook/a.md": "# Storage\n\nKeep samples frozen.\n",
                               "handbook/b.md": "# Assay\n\nRun the assay weekly.\n"}
        self.failed = set()
        self.requests = []
        self.truncated = False
        self.rate_limited = False

    def get(self, url, timeout):
        self.requests.append(url)
        if self.rate_limited:
            return Response({}, 403)
        from urllib.parse import quote
        if url == API + "/commits/" + quote(self.ref, safe=""):
            return Response({"sha": self.commit})
        if url.startswith(API + "/commits/"):
            return Response({}, 422)
        if url == API + f"/git/trees/{self.commit}?recursive=1":
            tree = [{"type": "blob", "mode": "100644", "path": path}
                    for path in self.files]
            tree += [{"type": "blob", "path": "outside.md"},
                     {"type": "blob", "path": "handbook/plot.png"},
                     {"type": "blob", "mode": "120000", "path": "handbook/link.md"}]
            return Response({"tree": tree, "truncated": self.truncated})
        prefix = f"https://raw.githubusercontent.com/example/docs/{self.commit}/"
        path = url.removeprefix(prefix)
        if path in self.failed:
            raise requests.ConnectionError("temporary outage")
        return Response(self.files[path])


class GithubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"FAQ_DATA_DIR": self.temp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def ingest(self, session, **kwargs):
        return import_collection("docs", ROOT, WordTokenizer(), session=session, **kwargs)

    def test_import_reads_files_not_listing_and_preserves_locations(self):
        session = Session()
        manifest = self.ingest(session)
        self.assertEqual(len(manifest["sources"]), 2)
        self.assertTrue(all(f"/blob/{COMMIT}/handbook/" in record["url"]
                            and record["source_revision"] == COMMIT and record["fetched_at"]
                            for record in manifest["records"]))
        self.assertIn("lines 3-3", manifest["records"][0]["location"])
        self.assertNotIn(ROOT, session.requests)
        self.assertNotIn("# Storage", manifest["records"][0]["text"])

    def test_unchanged_files_keep_versions_and_changed_files_increment(self):
        first = self.ingest(Session())
        self.assertEqual(first["records"], self.ingest(Session())["records"])
        changed = Session(commit="b" * 40)
        changed.files["handbook/a.md"] = "# Storage\n\nKeep samples at -80 C.\n"
        second = self.ingest(changed)
        self.assertEqual([r["version"] for r in second["records"]], [1, 2])
        self.assertEqual({r["source_id"] for r in first["records"]},
                         {r["source_id"] for r in second["records"]})
        self.assertTrue(all(r["source_revision"] == "b" * 40 for r in second["records"]))
        before = {r["repository_path"]: r for r in first["records"]}
        after = {r["repository_path"]: r for r in second["records"]}
        self.assertGreaterEqual(after["handbook/b.md"]["fetched_at"], before["handbook/b.md"]["fetched_at"])

    def test_partial_failure_does_not_prune_previous_sources(self):
        first = self.ingest(Session())
        session = Session()
        session.failed.add("handbook/b.md")
        second = self.ingest(session)
        self.assertEqual(first["records"], second["records"])
        self.assertTrue(any("previous records retained" in w for w in second["warnings"]))

    def test_file_limit_does_not_prune_previous_sources(self):
        first = self.ingest(Session())
        second = self.ingest(Session(), max_pages=1)
        self.assertEqual(first["records"], second["records"])
        self.assertTrue(any("Stopped after 1 files" in w for w in second["warnings"]))

    def test_complete_import_prunes_deleted_files(self):
        self.ingest(Session())
        session = Session()
        session.files.pop("handbook/b.md")
        self.assertEqual(len(self.ingest(session)["sources"]), 1)

    def test_blob_import_selects_only_requested_file(self):
        result = import_collection("docs", ROOT.replace("tree", "blob") + "/a.md",
                                   WordTokenizer(), session=Session())
        self.assertEqual(len(result["sources"]), 1)

    def test_slash_branch_is_resolved_before_directory(self):
        result = import_collection("docs", ROOT.replace("/main/", "/feature/docs/"),
                                   WordTokenizer(), session=Session(ref="feature/docs"))
        self.assertEqual(len(result["sources"]), 2)
        self.assertTrue(all(f"/blob/{COMMIT}/handbook/" in r["url"] for r in result["records"]))

    def test_successful_reimport_replaces_old_github_html_listing(self):
        from knowledge_store import collection_path
        from local_store import write_json
        write_json(collection_path("docs"), {"name": "docs", "records": [
            {"source_path": ROOT, "text": "Directory listing"}],
            "sources": {ROOT: {"kind": "website", "group": ROOT}}})
        result = self.ingest(Session())
        self.assertNotIn(ROOT, result["sources"])
        self.assertFalse(any(r["source_path"] == ROOT for r in result["records"]))

    def test_truncated_inventory_and_rate_limit_fail_actionably(self):
        session = Session()
        session.truncated = True
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.ingest(session)
        session.rate_limited = True
        with self.assertRaisesRegex(ValueError, "rate limit"):
            self.ingest(session)

    def test_heading_only_sections_are_not_records(self):
        sections = list(markdown_sections("# Intro\n## Empty\n## Storage\nKeep RNA frozen.\n", "p.md", blockwise=True))
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][0], "Storage")
        self.assertEqual(sections[0][2], "Keep RNA frozen.")

    def test_code_and_bold_subheadings_remain_source_text(self):
        text = "# Run\n\n**Binary**\n\n```R\nf(x=1)\n```\n"
        section = list(markdown_sections(text, "run.md", blockwise=True))[0]
        self.assertEqual(section[2], "**Binary**\n\n```R\nf(x=1)\n```")

    def test_link_only_boilerplate_is_not_indexed(self):
        text = "# Guide\n\n[Back to examples](https://example.test)\n\nKeep samples frozen.\n"
        sections = list(markdown_sections(text, "guide.md", blockwise=True))
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][2], "Keep samples frozen.")

    def test_link_list_is_not_a_document_fact(self):
        text = "# Contents\n\n- [1. Storage](a.md)\n- [2. Assay](b.md)\n"
        self.assertEqual(list(markdown_sections(text, "index.md", blockwise=True)), [])

    def test_retrieval_text_omits_link_destinations_not_link_labels(self):
        text = "The [Package](https://example.test/very/long/link) fits **models**."
        self.assertEqual(markdown_search_text(text), "The Package fits models.")
        session = Session({"handbook/a.md": text})
        record = self.ingest(session)["records"][0]
        self.assertEqual(record["text"], text)
        self.assertNotIn("https://example.test", record["retrieval_text"])

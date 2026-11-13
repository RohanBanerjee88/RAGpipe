"""Failure-oriented checks for the RefN audit, with no external network calls."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from markdown_it import MarkdownIt

from grounding import validate_grounded_answer, evidence_excerpts
from knowledge_store import crawl_website, import_collection, web_sections
from local_store import read_json, write_json
import model_setup as models
from retriever import FAQRetriever
from routing_policy import required_identifiers
from tests.test_collection_routing import result
from tests.test_knowledge_store import FakeResponse, FakeSession, WordTokenizer


class AnswerAuditTests(unittest.TestCase):
    def test_partial_indented_code_with_inline_citation_is_not_supported(self):
        source = {"answer": "```python\nx=1\ny=x+1\n```"}
        self.assertFalse(validate_grounded_answer("    x=1 [S1]", 1, [source])[0])

    def test_complete_indented_code_needs_external_citation(self):
        source = {"answer": "```python\nx=1\ny=x+1\n```"}
        self.assertTrue(validate_grounded_answer("    x=1\n    y=x+1\n\n[S1]", 1, [source])[0])

    def test_code_literals_are_not_interpreted_as_source_labels(self):
        source = {"answer": '```python\nprint("[S99]")\n```'}
        self.assertTrue(validate_grounded_answer(source["answer"] + "\n[S1]", 1, [source])[0])

    def test_source_count_cannot_allow_an_index_past_actual_sources(self):
        self.assertFalse(validate_grounded_answer("Use the stated command. [S2]", 2,
                                                 [{"answer": "Use the stated command."}])[0])

    def test_first_line_code_indentation_is_not_silently_changed(self):
        source = {"answer": "```python\n    x=1\n    y=x+1\n```"}
        self.assertFalse(validate_grounded_answer("```python\nx=1\n    y=x+1\n```\n[S1]", 1, [source])[0])

    def test_indented_excerpt_does_not_swallow_its_citation(self):
        text = evidence_excerpts([{"answer": "    x=1\n    y=x+1"}])
        code = [token.content for token in MarkdownIt().parse(text) if token.type == "code_block"]
        self.assertEqual(code, ["x=1\ny=x+1\n"])

    def test_bare_calls_get_the_same_identifier_guard(self):
        for query in ("How do I use unknownReader(file)?", "How do I use `unknownReader(file)`?"):
            self.assertIn("unknownreader", required_identifiers(query))

    def test_top_faq_cannot_bypass_a_missing_identifier_or_option(self):
        faq = {**result("icer", "Use readBinMat with model='BRR'."),
               "record_type": "faq", "needs_llama": False}
        retriever = FAQRetriever.__new__(FAQRetriever)
        retriever.find_top_k_faqs = Mock(return_value=[faq])
        for query in ("How do I use `unknownreader(file)`?", "How do I set model='unknown'?"):
            self.assertEqual(retriever.get_best_match(query)["route"], "abstain")

    def test_excluded_model_is_not_returned_as_the_requested_example(self):
        candidate = result(answer="```r\nfit(model='RKHS')\n```", url="model.md")
        retriever = FAQRetriever.__new__(FAQRetriever)
        retriever.find_top_k_faqs = Mock(return_value=[candidate])
        retriever.model_values = {"RKHS"}
        retriever.domain_terms = {"RKHS"}
        for query in ("Give me an example of a model except RKHS", "Give me an example, not RKHS"):
            self.assertEqual(retriever.get_best_match(query)["route"], "clarify")

    def test_case_sensitive_options_do_not_share_a_retrieval_cache_entry(self):
        from collections import OrderedDict
        import torch
        retriever = FAQRetriever.__new__(FAQRetriever)
        retriever.debug = False
        retriever._result_cache = OrderedDict()
        retriever.bi_encoder = Mock()
        retriever.bi_encoder.encode.return_value = torch.tensor([[1., 0.]])
        retriever.question_embeddings = torch.tensor([[1., 0.]])
        retriever.collection_indices = {"lab": [0]}
        retriever.collection_bm25 = {"lab": Mock(scores=Mock(return_value=[1.]))}
        retriever.cross_encoder = Mock(predict=Mock(return_value=[4.]))
        retriever.faqs = [{"collection": "lab", "record_type": "faq", "question": "Mode setting",
                           "answer": "Choose the documented mode.", "category": "General"}]
        retriever.semantic_texts = ["Mode setting"]
        retriever.search_texts = ["Choose the documented mode."]
        first = "How do I set mode='BayesA'?"
        retriever.find_top_k_faqs(first)
        retriever.find_top_k_faqs(first)
        self.assertEqual(retriever.cross_encoder.predict.call_count, 1)
        retriever.find_top_k_faqs("How do I set mode='bayesa'?")
        self.assertEqual(retriever.cross_encoder.predict.call_count, 2)

    def test_both_items_does_not_override_an_explicit_collection(self):
        retriever = FAQRetriever.__new__(FAQRetriever)
        weak = {**result("atlas", "CPU and GPU usage notes."), "evidence_sufficient": False}
        retriever.find_top_k_faqs = Mock(return_value=[weak, result("boreal", "CPU and GPU usage notes.")])
        self.assertEqual(retriever.get_best_match("How do I monitor both CPU and GPU usage in Atlas?")["route"], "abstain")


class ModelStateAuditTests(unittest.TestCase):
    def test_named_collection_does_not_read_corrupt_unrelated_corpora(self):
        from knowledge_store import collection_path
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FAQ_DATA_DIR": directory}):
            broken_icer = Path(directory) / "broken-faqs.json"
            broken_icer.write_text("broken JSON")
            write_json(collection_path("atlas"), {"name": "atlas", "records": [{
                "collection": "atlas", "record_type": "passage", "question": "Storage",
                "title": "Storage", "text": "Keep specimens frozen."}]})
            unrelated = collection_path("unrelated")
            unrelated.parent.mkdir(parents=True)
            unrelated.write_text("broken JSON")
            retriever = FAQRetriever.__new__(FAQRetriever)
            retriever.collection, retriever.debug = "atlas", False
            with patch("knowledge_store.FAQ_JSON_PATH", broken_icer):
                retriever._load_faqs()
            self.assertEqual(list(retriever.collection_indices), ["atlas"])

    def test_offline_session_does_not_change_later_session_or_environment(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(models, "_active", None), \
                patch.object(models, "_offline", False), patch.object(models, "OFFLINE_MODE", False), \
                patch.object(models, "resolve_profile", return_value=models.ModelProfile("none", backend="none")):
            models.configure_session(offline=True)
            self.assertTrue(models.offline_mode())
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)
            models.configure_session(offline=False)
            self.assertFalse(models.offline_mode())

    def test_prepare_preserves_a_sibling_update_during_download(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FAQ_DATA_DIR": directory}):
            manifest = Path(directory) / "prepared_models.json"
            def download(**kwargs):
                write_json(manifest, {"org/other@main": "other-commit"})
                return "/cache/current-commit"
            with patch("huggingface_hub.snapshot_download", side_effect=download):
                models.model_location("org/current", download=True)
            self.assertEqual(read_json(manifest), {"org/other@main": "other-commit", "org/current@main": "current-commit"})


class WebsiteAuditTests(unittest.TestCase):
    root = "https://example.test/docs/"

    def session(self, child=None):
        pages = {
            "https://example.test/robots.txt": FakeResponse("https://example.test/robots.txt", "User-agent: *\nAllow: /", "text/plain"),
            self.root: FakeResponse(self.root, '<main><h1>Handbook</h1><p>Check the instrument weekly.</p><a href="procedure.html">Procedure</a></main>'),
            self.root + "procedure.html": child or FakeResponse(self.root + "procedure.html", "<main><p>Keep samples frozen.</p></main>"),
        }
        return FakeSession(pages)

    def test_skipped_pages_do_not_prune_previously_good_sources(self):
        for kind in ("non_html", "empty", "oversize"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {"FAQ_DATA_DIR": directory}):
                original = import_collection("lab", self.root, WordTokenizer(), session=self.session())
                if kind == "non_html":
                    child = FakeResponse(self.root + "procedure.html", "temporary response", "application/json")
                elif kind == "empty":
                    child = FakeResponse(self.root + "procedure.html", "<html></html>")
                else:
                    child = FakeResponse(self.root + "procedure.html", "x" * 2000)
                with patch("knowledge_store.WEB_MAX_BYTES", 1000):
                    updated = import_collection("lab", self.root, WordTokenizer(), session=self.session(child))
                self.assertEqual(original["records"], updated["records"])

    def test_web_code_preserves_newlines_indentation_and_literal_fences(self):
        code = 'if ready:\n    print("```")\n    save()\n'
        sections = web_sections("<main><h1>Run</h1><pre><code>" + code + "</code></pre></main>", self.root)
        rendered = "\n".join(section[2] for section in sections)
        blocks = [token.content for token in MarkdownIt().parse(rendered) if token.type == "fence"]
        self.assertEqual(blocks, [code])

    def test_web_prose_and_code_have_separate_source_blocks(self):
        sections = web_sections("<main><h1>Run</h1><p>Validate inputs first.</p>"
                                "<pre>if ready:\n    run()</pre><p>The log is run.log.</p></main>", self.root)
        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0][2], "Validate inputs first.")
        self.assertIn("    run()", sections[1][2])
        self.assertEqual(sections[2][2], "The log is run.log.")

    def test_oversize_body_stops_streaming_and_closes_the_response(self):
        class Stream(FakeResponse):
            def __init__(inner):
                super().__init__(self.root + "procedure.html", "")
                inner.read_chunks, inner.closed = 0, False
            def iter_content(inner, chunk_size):
                for _ in range(10):
                    inner.read_chunks += 1
                    yield b"x" * 700
            def close(inner):
                inner.closed = True
        stream = Stream()
        with patch("knowledge_store.WEB_MAX_BYTES", 1000):
            _, _, complete, warnings = crawl_website(self.root, session=self.session(stream))
        self.assertEqual(stream.read_chunks, 2)
        self.assertTrue(stream.closed)
        self.assertFalse(complete)
        self.assertTrue(any("exceeds" in message for message in warnings))

    def test_successful_refresh_still_removes_deleted_pages(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FAQ_DATA_DIR": directory}):
            import_collection("lab", self.root, WordTokenizer(), session=self.session())
            session = self.session()
            session.pages[self.root] = FakeResponse(self.root, "<main><p>Check the instrument weekly.</p></main>")
            refreshed = import_collection("lab", self.root, WordTokenizer(), session=session)
            self.assertNotIn(self.root + "procedure.html", refreshed["sources"])

    def test_redirects_cannot_fetch_an_out_of_scope_host(self):
        target = "http://127.0.0.1/private"
        class RedirectSession(FakeSession):
            def get(inner, url, timeout, allow_redirects=True, **kwargs):
                if url == self.root and allow_redirects:
                    inner.requested.extend([url, target])
                    return FakeResponse(target, "private response")
                response = super().get(url, timeout)
                return response
        response = FakeResponse(self.root, "", status=302)
        response.headers["Location"] = target
        session = RedirectSession({"https://example.test/robots.txt": FakeResponse("https://example.test/robots.txt", "", "text/plain"), self.root: response})
        crawl_website(self.root, session=session)
        self.assertNotIn(target, session.requested)

    def test_https_crawl_does_not_follow_a_downgraded_page_link(self):
        session = self.session()
        target = "http://example.test/docs/insecure.html"
        session.pages[self.root] = FakeResponse(self.root, '<main><p>Handbook.</p><a href="' + target + '">Next</a></main>')
        _, _, complete, warnings = crawl_website(self.root, session=session)
        self.assertNotIn(target, session.requested)
        self.assertFalse(complete)
        self.assertTrue(any("outside website scope" in message for message in warnings))

    def test_same_scope_redirect_still_imports_the_destination(self):
        session = self.session()
        response = FakeResponse(self.root, "", status=302)
        response.headers["Location"] = "current.html"
        session.pages[self.root] = response
        session.pages[self.root + "current.html"] = FakeResponse(self.root + "current.html", "<main><p>Current handbook.</p></main>")
        pages, _, complete, _ = crawl_website(self.root, session=session)
        self.assertTrue(complete)
        self.assertEqual(pages[0][0], self.root + "current.html")


if __name__ == "__main__":
    unittest.main()

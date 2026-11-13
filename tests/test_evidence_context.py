"""Complete procedure evidence without mixing headings, versions, or files."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidence_context import reconstruct_context
from grounding import validate_grounded_answer, evidence_excerpts
from knowledge_store import import_collection
from tests.test_knowledge_store import WordTokenizer


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"FAQ_DATA_DIR": self.temp.name + "/store"})
        self.env.start()
        self.path = Path(self.temp.name) / "guide.md"
        self.code = "```python\n" + "\n".join(f"value{i} = {i}" for i in range(80)) + "\n```"
        self.path.write_text("# First\nDo not run before checking the input.\n\n" + self.code +
                             "\n\n# Unrelated\nNever mix this unrelated section.\n")
        self.manifest = import_collection("lab", self.path, WordTokenizer())
        self.sources = {("lab", key): value for key, value in self.manifest["sources"].items()}
        self.anchor = next(record for record in self.manifest["records"] if "value40" in record["answer"])
        self.anchor = {**self.anchor, "matched_answer": self.anchor["answer"]}

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def expand(self, **kwargs):
        return reconstruct_context("How do I run this example?", [self.anchor], self.sources,
                                   WordTokenizer(), **kwargs)[0]

    def test_reassembles_full_code_and_precondition_with_exact_location(self):
        expanded = self.expand()
        self.assertIn(self.code, expanded["matched_answer"])
        self.assertIn("Do not run before checking", expanded["matched_answer"])
        self.assertNotIn("unrelated", expanded["matched_answer"])
        self.assertEqual(expanded["location"], "lines 2-85")
        self.assertLess(len(self.anchor["answer"]), len(expanded["matched_answer"]))

    def test_fact_and_faq_queries_do_not_expand(self):
        self.assertEqual(reconstruct_context("What is the value?", [self.anchor], self.sources,
                                             WordTokenizer()), [self.anchor])
        faq = {**self.anchor, "record_type": "faq"}
        self.assertEqual(reconstruct_context("How?", [faq], {}, WordTokenizer()), [faq])

    def test_oversize_context_does_not_supply_truncated_code(self):
        expanded = self.expand(limit=20)
        self.assertTrue(expanded["context_incomplete"])
        self.assertNotIn("value40", expanded["matched_answer"])
        self.assertIn("no partial code", expanded["matched_answer"])

    def test_old_or_mismatched_versions_require_reimport(self):
        for values in ({"version": 999}, {"source_revision": "new"}, {"block_index": None},
                       {"source_path": "different-file"}):
            match = {**self.anchor, **values}
            expanded = reconstruct_context("How?", [match], self.sources, WordTokenizer())[0]
            self.assertTrue(expanded["context_incomplete"])
            self.assertIn("Reimport", expanded["matched_answer"])

    def test_repeated_chunks_of_same_block_deduplicate(self):
        matches = reconstruct_context("How?", [self.anchor, self.anchor], self.sources, WordTokenizer())
        self.assertEqual(len(matches), 1)

    def test_larger_later_anchor_keeps_first_rank_and_source_identity(self):
        self.path.write_text("# Example\nSetup first.\n\n```R\nx=1\n```\n\n"
                             "Then continue.\n\n```R\ny=x+1\n```\n")
        manifest = import_collection("lab", self.path, WordTokenizer())
        sources = {("lab", key): value for key, value in manifest["sources"].items()}
        first, last = manifest["records"][1], manifest["records"][-1]
        other = {"record_type": "faq", "source_id": "other", "answer": "Other source."}
        matches = reconstruct_context("How?", [first, other, last], sources, WordTokenizer())
        self.assertEqual(matches[0]["source_id"], first["source_id"])
        self.assertIn("y=x+1", matches[0]["matched_answer"])
        self.assertEqual(matches[1], other)
        self.assertEqual(len(matches), 2)

    def test_repeated_heading_names_do_not_join_different_sections(self):
        self.path.write_text("# Repeat\nWrong earlier procedure.\n\n# Repeat\nCorrect setup.\n\n```R\nx=1\n```\n")
        manifest = import_collection("lab", self.path, WordTokenizer())
        sources = {("lab", key): value for key, value in manifest["sources"].items()}
        anchor = {**manifest["records"][-1], "matched_answer": "x=1"}
        answer = reconstruct_context("How?", [anchor], sources, WordTokenizer())[0]["matched_answer"]
        self.assertIn("Correct setup", answer)
        self.assertNotIn("Wrong earlier", answer)

    def test_cited_complete_code_validates_but_changed_or_partial_code_does_not(self):
        source = {"answer": "```R\nx=1\ny=x+1\n```"}
        self.assertEqual(validate_grounded_answer(source["answer"] + "\n[S1]", 1, [source]),
                         (True, "grounded"))
        for answer in ("```R\nx=2\ny=x+1\n```\n[S1]", "```R\nx=1\n```\n[S1]",
                       "```R\nx=1\ny=x+1\n[S1]", "```R\nx=1\ny=x+1\n```"):
            self.assertFalse(validate_grounded_answer(answer, 1, [source])[0])

    def test_excerpt_citation_does_not_break_closing_code_fence(self):
        from markdown_it import MarkdownIt
        source = {"answer": "```R\nx=1\ny=x+1\n```"}
        code = [token.content for token in MarkdownIt().parse(evidence_excerpts([source]))
                if token.type == "fence"]
        self.assertEqual(code, ["x=1\ny=x+1\n"])

    def test_copied_code_without_explicit_precondition_is_not_accepted(self):
        condition = "Do not run before validating the input."
        code = "```R\nx=1\n```"
        source = {"answer": condition + "\n\n" + code, "required_setup": [condition]}
        self.assertEqual(validate_grounded_answer(code + "\n[S1]", 1, [source]),
                         (False, "missing_procedure_setup"))
        self.assertEqual(validate_grounded_answer(condition + " [S1]\n\n" + code + "\n[S1]", 1, [source]),
                         (True, "grounded"))

    def test_unclosed_source_code_requires_review_not_silent_repair(self):
        self.path.write_text("# Broken\n```python\nx=1\n")
        manifest = import_collection("lab", self.path, WordTokenizer())
        sources = {("lab", key): value for key, value in manifest["sources"].items()}
        answer = reconstruct_context("How?", [manifest["records"][0]], sources, WordTokenizer())[0]
        self.assertTrue(answer["context_incomplete"])
        self.assertNotIn("x=1", answer["matched_answer"])

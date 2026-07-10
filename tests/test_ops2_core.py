"""Fast deterministic unit tests for OPS2 quality controls."""

import unittest
import json
import tempfile
from pathlib import Path

from config import FAQ_JSON_PATH, PROJECT_ROOT, TREE_JSON_PATH
from grounding import INSUFFICIENT_EVIDENCE, validate_grounded_answer
from hybrid_retrieval import BM25Index, lexical_coverage, reciprocal_rank_fusion
from ingestion import normalize_faq, normalize_text, normalize_url
from source_confidence import combined_source_confidence, trust_score
from scrape import load_previous_faqs, write_json


class FreshCloneTests(unittest.TestCase):
    def test_generated_paths_are_rooted_in_repository(self):
        self.assertTrue(Path(FAQ_JSON_PATH).is_absolute())
        self.assertEqual(Path(FAQ_JSON_PATH).parent, PROJECT_ROOT)
        self.assertEqual(Path(TREE_JSON_PATH).parent, PROJECT_ROOT)

    def test_missing_corpus_is_a_valid_first_run(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "all_faqs.json"
            self.assertEqual(load_previous_faqs(missing_path), [])

    def test_json_writer_creates_configured_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "generated" / "all_faqs.json"
            write_json(output_path, {"faqs": []})
            self.assertEqual(json.loads(output_path.read_text()), {"faqs": []})


class IngestionTests(unittest.TestCase):
    def test_url_normalization_does_not_insert_spaces(self):
        url = "https://docs.icer.msu.edu/Frequently_Asked_Questions_FAQ_/"
        self.assertEqual(normalize_url(url), url)

    def test_normalization_is_idempotent(self):
        raw = "How do I use theHPCCwith runningGPUjobs?"
        normalized = normalize_text(raw)
        self.assertEqual(normalized, normalize_text(normalized))
        self.assertIn("HPCC", normalized)
        self.assertIn("GPU jobs", normalized)

    def test_legacy_inline_tag_joins_are_repaired(self):
        self.assertEqual(normalize_text("job ranOutOfMemory"), "job ran Out Of Memory")
        self.assertEqual(normalize_text("amodule: not founderror"), "a module: not found error")

    def test_content_change_increments_version(self):
        original = normalize_faq({
            "url": "https://example.test/faq",
            "category": "Jobs",
            "question": "What is OOM?",
            "answer": "Out of memory.",
            "scraped_at": "2026-01-01T00:00:00+00:00",
        })
        updated = normalize_faq({
            **original,
            "answer": "OOM means out of memory.",
            "scraped_at": "2026-02-01T00:00:00+00:00",
        }, previous=original)
        self.assertEqual(updated["version"], original["version"] + 1)
        self.assertEqual(updated["source_id"], original["source_id"])

    def test_legacy_hash_migration_does_not_increment_version(self):
        previous = {
            "url": "https://example.test/faq",
            "category": "Jobs",
            "question": "What is OOM?",
            "answer": "Out of memory.",
            "hash": "legacy-md5-hash",
            "version": 3,
            "scraped_at": "2026-01-01T00:00:00+00:00",
        }
        migrated = normalize_faq(previous, previous=previous, scraped_at="2026-02-01T00:00:00+00:00")
        self.assertEqual(migrated["version"], 3)
        self.assertEqual(migrated["updated_at"], previous["scraped_at"])


class HybridRetrievalTests(unittest.TestCase):
    def test_bm25_prefers_exact_technical_term(self):
        index = BM25Index(["GPU jobs and CUDA", "password reset", "storage quota"])
        self.assertEqual(index.rank("Can I run a GPU job?", limit=1), [0])

    def test_rank_fusion_combines_rankings(self):
        fused = reciprocal_rank_fusion([[0, 1], [1, 0]], weights=[1.0, 2.0])
        self.assertGreater(fused[1], fused[0])

    def test_lexical_coverage_ignores_stopwords(self):
        self.assertGreaterEqual(lexical_coverage("What does OOM mean?", "OOM means out of memory"), 0.5)


class GroundingTests(unittest.TestCase):
    def test_valid_citation(self):
        self.assertTrue(validate_grounded_answer("OOM means out of memory. [S1]", 1)[0])

    def test_unknown_citation_rejected(self):
        self.assertFalse(validate_grounded_answer("OOM means out of memory. [S2]", 1)[0])

    def test_uncited_second_claim_is_rejected(self):
        answer = "OOM means out of memory. [S1] Increase memory after checking usage."
        self.assertEqual(validate_grounded_answer(answer, 1), (False, "uncited_claim"))

    def test_explicit_abstention_is_valid(self):
        self.assertTrue(validate_grounded_answer(INSUFFICIENT_EVIDENCE, 1)[0])


class SourceConfidenceTests(unittest.TestCase):
    def test_official_icer_source_is_trusted(self):
        self.assertEqual(trust_score("https://docs.icer.msu.edu/faq"), 1.0)

    def test_combined_score_is_bounded(self):
        score = combined_source_confidence(1.0, 0.8, 1.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

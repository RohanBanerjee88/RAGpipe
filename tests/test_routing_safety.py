"""Regression tests for evidence gating and model-failure behavior."""

import unittest
from unittest.mock import patch

import main as main_module
from hybrid_retrieval import lexical_overlap_count, tokenize
from retriever import FAQRetriever


def weak_result(confidence="very_low"):
    return {
        "raw_score": -10.0,
        "normalized_score": 1.0,
        "bi_score": 0.2,
        "bm25_normalized": 1.0,
        "lexical_coverage": 0.34,
        "lexical_overlap_count": 1,
        "confidence": confidence,
        "needs_llama": True,
        "matched_question": "Do you support running GPU jobs?",
        "matched_answer": "GPU jobs are supported.",
        "url": "https://docs.icer.msu.edu/faq",
        "category": "Jobs",
        "source_id": "faq-gpu",
        "evidence_sufficient": True,
    }


class LexicalEvidenceTests(unittest.TestCase):
    def test_generic_question_words_are_stopwords(self):
        self.assertEqual(tokenize("Who do you support for Michigan Senate?"), ["support", "michigan", "senate"])

    def test_one_generic_overlap_is_not_strong_lexical_evidence(self):
        query = "Who do you support for Michigan Senate?"
        document = "Do you support running GPU jobs?"
        self.assertEqual(lexical_overlap_count(query, document), 1)

        retriever = FAQRetriever.__new__(FAQRetriever)
        result = weak_result()
        result["evidence_sufficient"] = False
        self.assertFalse(retriever._has_sufficient_evidence(result))


class ModelFailureTests(unittest.TestCase):
    def test_weak_match_abstains_when_model_cannot_load(self):
        decision = {
            "route": "llama",
            "reason": "confidence_very_low",
            "result": weak_result(),
            "context_faqs": [weak_result()],
        }
        fake_retriever = unittest.mock.Mock()
        fake_retriever.get_best_match.return_value = decision

        with (
            patch.object(main_module, "FAQRetriever", return_value=fake_retriever),
            patch.object(main_module, "is_llama_loaded", return_value=False),
            patch.object(main_module, "get_llama_pipeline", side_effect=RuntimeError("model unavailable")),
        ):
            assistant = main_module.SmartFAQAssistant(debug=False)
            answer, route, confidence = assistant.get_answer("unsupported question")

        self.assertEqual(route, "abstain")
        self.assertEqual(confidence, "very_low")
        self.assertNotIn("GPU jobs are supported", answer)


if __name__ == "__main__":
    unittest.main()

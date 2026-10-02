"""Cross-collection ambiguity and absolute confidence regressions."""

import unittest
from unittest.mock import Mock, patch

from retriever import FAQRetriever
from main import SmartFAQAssistant
from model_setup import ModelProfile


def result(collection="atlas", answer="Keep samples at -80 C.", url="protocol.txt"):
    return {"collection": collection, "matched_question": "Specimen storage", "matched_answer": answer,
            "url": url, "location": "line 1", "record_type": "passage", "source_id": collection,
            "raw_score": 4.0, "bi_score": .6, "evidence_sufficient": True,
            "confidence": "medium", "needs_llama": True}


class CollectionRoutingTests(unittest.TestCase):
    def retriever(self, results):
        retriever = FAQRetriever.__new__(FAQRetriever)
        retriever.find_top_k_faqs = Mock(return_value=results)
        return retriever

    def test_relative_rank_does_not_control_confidence(self):
        retriever = self.retriever([])
        low = retriever._get_confidence_level_ensemble(.7, 4., 0., 0.)
        high = retriever._get_confidence_level_ensemble(.7, 4., 1., 1.)
        self.assertEqual(low, high)

    def test_ambiguous_collections_require_clarification(self):
        retriever = self.retriever([result(), result("boreal", "Keep samples at -20 C.")])
        self.assertEqual(retriever.get_best_match("Where should samples be stored?")["route"], "clarify")

    def test_multiple_collection_question_preserves_both_sources(self):
        retriever = self.retriever([result(), result("boreal", "Keep samples at -20 C.")])
        decision = retriever.get_best_match("Compare storage in both labs")
        self.assertEqual(len(decision["context_faqs"]), 2)
        self.assertTrue(decision["multiple_collections"])

    def test_different_versions_of_same_topic_require_clarification(self):
        retriever = self.retriever([result(), result(answer="Keep samples at -20 C.", url="other.txt")])
        self.assertEqual(retriever.get_best_match("Where should samples be stored?")["reason"], "ambiguous_sources")

    def test_retrieval_only_returns_cited_excerpt_without_generator(self):
        retriever = self.retriever([result()])
        with patch("main.FAQRetriever", return_value=retriever), patch("model_setup.active_profile", return_value=ModelProfile("none", backend="none")):
            assistant = SmartFAQAssistant()
            with patch.object(assistant, "_load_llama_if_needed") as load:
                answer, route, _ = assistant.get_answer("Where should samples be stored?")
                load.assert_not_called()
            self.assertEqual(route, "extractive")
            self.assertIn("-80 C", answer)
            self.assertIn("[S1]", answer)

    def test_clarification_selection_uses_only_selected_source(self):
        retriever = self.retriever([result(), result("boreal", "Keep samples at -20 C.")])
        assistant = SmartFAQAssistant(debug=False, retriever=retriever)
        self.assertEqual(assistant.get_answer("Where should samples be stored?")[1], "clarify")
        self.assertEqual(assistant.get_answer("3")[1], "clarify")
        answer, route, _ = assistant.get_answer("2")
        self.assertEqual(route, "extractive")
        self.assertIn("-20 C", answer)
        self.assertNotIn("-80 C", answer)
        self.assertEqual(retriever.find_top_k_faqs.call_count, 1)

    def test_legacy_helper_does_not_bypass_clarification(self):
        from prompt import get_answer_with_llama
        retriever = self.retriever([result(), result("boreal", "Keep samples at -20 C.")])
        with patch("prompt.DEBUG_MODE", False):
            self.assertIn("Which should I use?", get_answer_with_llama("Where should samples be stored?", retriever))

    def test_short_uncited_claim_is_rejected(self):
        from grounding import validate_grounded_answer
        valid, reason = validate_grounded_answer("Keep samples at -80 C. [S1]\nUse bleach.", 1, [result()])
        self.assertFalse(valid)
        self.assertEqual(reason, "uncited_claim")

    def test_literal_definition_adds_candidates_not_forced_answers(self):
        retriever = FAQRetriever.__new__(FAQRetriever)
        retriever.faqs = [
            {"record_type": "passage", "answer": "The ZETA Package implements sparse regression."},
            {"record_type": "passage", "answer": "ZETA examples for parallel processing."},
            {"record_type": "faq", "answer": "ZETA is a package."},
            {"record_type": "passage", "answer": "ZETA assigns priors; the prior is Gaussian."},
        ]
        self.assertEqual(retriever._definition_candidates("What is ZETA?", [0, 1, 2, 3]), [0])
        self.assertEqual(retriever._definition_candidates("How do I run ZETA?", [0, 1, 2]), [])
        self.assertEqual(retriever._definition_candidates("What is OMEGA?", [0, 1, 2]), [])

    def test_strong_body_rerank_can_corroborate_a_weak_embedding(self):
        retriever = FAQRetriever.__new__(FAQRetriever)
        evidence = {"record_type": "passage", "raw_score": 6.0, "bi_score": .29,
                    "lexical_overlap_count": 1}
        self.assertTrue(retriever._has_sufficient_evidence(evidence))
        evidence["raw_score"] = .5
        self.assertFalse(retriever._has_sufficient_evidence(evidence))

    def test_definition_context_excludes_unrelated_code(self):
        definition = {**result(answer="ZETA is a sparse regression package."), "literal_definition": True}
        code = result(answer="run_parallel(ZETA)")
        retriever = self.retriever([definition, code])
        decision = retriever.get_best_match("What is ZETA?")
        self.assertEqual(decision["context_faqs"], [definition])

    def test_unknown_explicit_identifier_does_not_get_a_topic_only_answer(self):
        retriever = self.retriever([result(answer="The package supports Gaussian priors.")])
        self.assertEqual(retriever.get_best_match("Does it support UNKNOWN_PRIOR?")["route"], "abstain")
        self.assertEqual(retriever.get_best_match("Does it support `unknownPrior`?")["route"], "abstain")
        self.assertEqual(retriever.get_best_match("Does it support `Gaussian` priors?")["route"], "llama")

    def test_strong_literal_definition_does_not_require_a_generator(self):
        definition = {**result(answer="ZETA is a sparse regression package."), "literal_definition": True}
        assistant = SmartFAQAssistant(debug=False, retriever=self.retriever([definition]))
        with patch.object(assistant, "_load_llama_if_needed") as load:
            answer, route, _ = assistant.get_answer("What is ZETA?")
        load.assert_not_called()
        self.assertEqual(route, "extractive")
        self.assertIn("ZETA is a sparse regression package.", answer)

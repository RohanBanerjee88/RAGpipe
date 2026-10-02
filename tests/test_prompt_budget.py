"""Generation packs complete evidence before invoking a small model."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from prompt import build_grounded_prompt, generation_input_budget
from main import SmartFAQAssistant


class Tokenizer:
    model_max_length = 512

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


class PromptBudgetTests(unittest.TestCase):
    def test_whole_sources_fit_budget_and_citations_stay_contiguous(self):
        sources = [{"answer": f"Unique{i} " + "word " * 80, "url": f"source{i}"}
                   for i in range(3)]
        prompt, selected = build_grounded_prompt("What is documented?", sources,
                                                tokenizer=Tokenizer(), token_budget=230)
        self.assertEqual(selected, sources[:2])
        self.assertLessEqual(len(Tokenizer()(prompt)["input_ids"]), 230)
        self.assertIn(selected[-1]["answer"], prompt)
        self.assertNotIn("[S3]", prompt)
        self.assertNotIn("Unique2", prompt)

    def test_model_limit_and_causal_output_reserve_are_respected(self):
        llm = SimpleNamespace(tokenizer=Tokenizer(), model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=400)))
        self.assertEqual(generation_input_budget(llm, "text2text-generation"), 400)
        with patch("prompt.LLAMA_MAX_NEW_TOKENS", 100):
            self.assertEqual(generation_input_budget(llm, "text-generation"), 300)

    def test_unfit_first_source_is_explicit_fallback_not_generation_error(self):
        assistant = SmartFAQAssistant(debug=False, retriever=object())
        llm = SimpleNamespace(tokenizer=Tokenizer())
        llm.tokenizer.model_max_length = 10
        source = {"question": "Definition", "answer": "An important supported statement.", "url": "source"}
        with patch("main.get_llama_pipeline", return_value=llm), \
                patch("main.get_llama_task", return_value="text2text-generation"), \
                patch("main.generate_llama_response") as generate:
            answer, status = assistant._generate_llama_answer("question", [source], "medium")
        generate.assert_not_called()
        self.assertEqual(status, "extractive_fallback")
        self.assertIn("context_budget_exceeded", answer)
        self.assertNotIn("generation_error", answer)


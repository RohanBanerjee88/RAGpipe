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
    def test_system_role_and_single_role_templates_keep_rules_and_evidence(self):
        from jinja2.exceptions import TemplateError
        from unittest.mock import Mock
        tokenizer = Mock()
        tokenizer.chat_template = "configured"
        def render(messages, **kwargs):
            if messages[0]["role"] == "system":
                raise TemplateError("System role unsupported")
            return "combined prompt"
        tokenizer.apply_chat_template.side_effect = render
        prompt, selected = build_grounded_prompt("Question?", [{"answer": "Source fact."}], tokenizer=tokenizer)
        self.assertEqual(prompt, "combined prompt")
        calls = tokenizer.apply_chat_template.call_args_list
        self.assertEqual(calls[0].args[0][0]["role"], "system")
        self.assertIn("Source fact.", calls[1].args[0][0]["content"])
        self.assertIn("Answer only from the evidence", calls[1].args[0][0]["content"])

    def test_profile_output_limit_reserves_space_without_changing_flan_default(self):
        from model_setup import ModelProfile
        from prompt import generation_output_limit
        tokenizer = Tokenizer()
        tokenizer.model_max_length = 2048
        llm = SimpleNamespace(tokenizer=tokenizer)
        with patch("prompt.active_profile", return_value=ModelProfile("candidate", max_new_tokens=512)):
            self.assertEqual(generation_output_limit(), 512)
            self.assertEqual(generation_input_budget(llm, "text-generation"), 1536)
            self.assertEqual(generation_input_budget(llm, "text-generation", 100), 1948)
        with patch("prompt.active_profile", return_value=ModelProfile("flan-base")):
            self.assertEqual(generation_output_limit(), 200)

    def test_chat_template_overhead_is_included_in_packing(self):
        class ChatTokenizer(Tokenizer):
            chat_template = "configured"

            def apply_chat_template(self, messages, **kwargs):
                return "START " * 20 + "\n".join(message["content"] for message in messages) + " ASSISTANT"

        source = {"answer": "word " * 80}
        plain, _ = build_grounded_prompt("question", [source])
        budget = len(Tokenizer()(plain)["input_ids"]) + 10
        plain, selected = build_grounded_prompt("question", [source], tokenizer=Tokenizer(), token_budget=budget)
        self.assertTrue(selected)
        chat, selected = build_grounded_prompt("question", [source], tokenizer=ChatTokenizer(), token_budget=budget)
        self.assertFalse(selected)
        self.assertTrue(chat.startswith("START "))

    def test_causal_output_is_not_split_on_an_answer_label(self):
        from prompt import generate_llama_response
        from unittest.mock import Mock
        pipeline = Mock(return_value=[{"generated_text": "Answer: a source quote [S1]"}])
        pipeline.tokenizer = Tokenizer()
        pipeline.tokenizer.eos_token_id = 1
        with patch("prompt.get_llama_pipeline", return_value=pipeline), \
                patch("prompt.get_llama_task", return_value="text-generation"):
            answer = generate_llama_response("question")
        self.assertEqual(answer, "Answer: a source quote [S1]")
        self.assertFalse(pipeline.call_args.kwargs["do_sample"])
        self.assertFalse(pipeline.call_args.kwargs["return_full_text"])

    def test_whole_sources_fit_budget_and_citations_stay_contiguous(self):
        sources = [{"answer": f"Unique{i} " + "word " * 80, "url": f"source{i}"}
                   for i in range(3)]
        prompt, selected = build_grounded_prompt("What is documented?", sources,
                                                tokenizer=Tokenizer(), token_budget=250)
        self.assertEqual(selected, sources[:2])
        self.assertLessEqual(len(Tokenizer()(prompt)["input_ids"]), 250)
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

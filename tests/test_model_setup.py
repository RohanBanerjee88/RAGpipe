"""Model configuration and offline setup behavior without model downloads."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_setup as models
from local_store import write_json


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"FAQ_DATA_DIR": self.temp.name}, clear=False)
        self.env.start()
        self.override = patch.dict(os.environ, {"FAQ_LLM_MODEL": ""})
        self.override.start()

    def tearDown(self):
        models._active = None
        models._offline = False
        self.override.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_profile_precedence(self):
        models.save_selection("flan-small")
        self.assertEqual(models.resolve_profile().name, "flan-small")
        with patch.dict(os.environ, {"FAQ_LLM_MODEL": "flan-base"}):
            self.assertEqual(models.resolve_profile().name, "flan-base")
            self.assertEqual(models.resolve_profile("retrieval-only").backend, "none")

    def test_local_directory_is_used_without_hub(self):
        with patch("huggingface_hub.snapshot_download") as download:
            self.assertEqual(models.model_location(self.temp.name), str(Path(self.temp.name).resolve()))
            download.assert_not_called()

    def test_cached_resolution_never_downloads(self):
        with patch("huggingface_hub.snapshot_download", return_value="/cache/snapshot") as download:
            models.model_location("google/flan-t5-base")
            self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_prepared_revision_is_reused(self):
        write_json(Path(self.temp.name) / "prepared_models.json", {"google/flan-t5-base@main": "abc"})
        with patch("huggingface_hub.snapshot_download", return_value="/cache/abc") as download:
            models.model_location("google/flan-t5-base")
            self.assertEqual(download.call_args.kwargs["revision"], "abc")

    def test_missing_generator_gives_prepare_instruction(self):
        with patch.object(models, "model_location", side_effect=FileNotFoundError("missing")):
            with self.assertRaisesRegex(RuntimeError, "models prepare"):
                models.generation_location()

    def test_missing_retriever_fails_before_search(self):
        with patch.object(models, "model_location", side_effect=FileNotFoundError("missing")):
            with self.assertRaisesRegex(RuntimeError, "prepare retrieval-only"):
                models.retrieval_location("all-MiniLM-L6-v2")

    def test_incomplete_sharded_model_rejected(self):
        directory = Path(self.temp.name)
        write_json(directory / "config.json", {})
        write_json(directory / "model.safetensors.index.json", {"weight_map": {"w": "missing.safetensors"}})
        with self.assertRaisesRegex(FileNotFoundError, "Missing model shard"):
            models.check_weights(directory)

    def test_retrieval_only_never_resolves_generator(self):
        models.configure_session("retrieval-only")
        with patch.object(models, "model_location") as location:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                models.generation_location()
            location.assert_not_called()

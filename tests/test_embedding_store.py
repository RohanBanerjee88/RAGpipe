"""Incremental embeddings must not be reused across incompatible models."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from embedding_store import cached_embeddings


class EmbeddingStoreTests(unittest.TestCase):
    def test_only_new_content_is_encoded_and_model_change_invalidates(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FAQ_DATA_DIR": directory}):
            encoder = Mock()
            encoder.encode.side_effect = lambda texts, **kwargs: torch.ones(len(texts), 3)
            _, count = cached_embeddings("lab", ["one", "two"], encoder, "model1", "cpu")
            self.assertEqual(count, 2)
            _, count = cached_embeddings("lab", ["one", "two"], encoder, "model1", "cpu")
            self.assertEqual(count, 0)
            _, count = cached_embeddings("lab", ["one", "changed"], encoder, "model1", "cpu")
            self.assertEqual(count, 1)
            _, count = cached_embeddings("lab", ["one", "changed"], encoder, "model2", "cpu")
            self.assertEqual(count, 2)

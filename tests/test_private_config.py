"""Private settings, permissions, validation, and real startup precedence."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from private_config import (
    ConfigurationError, TEMPLATE, config_path, initialize_config,
    read_private_config, validated_overrides,
)


class PrivateConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "settings" / "config.toml"
        self.env = patch.dict(os.environ, {"FAQ_CONFIG": str(self.path)}, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temporary.cleanup()

    def write(self, text):
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(text, encoding="utf-8")
        self.path.chmod(0o600)

    def run_code(self, code, **env):
        return subprocess.run([sys.executable, "-c", code], text=True, capture_output=True,
                              env={**os.environ, **env}, check=True)

    def test_initialize_is_private_and_never_overwrites(self):
        initialize_config()
        self.assertEqual(self.path.read_text(), TEMPLATE)
        if os.name == "posix":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)
        with self.assertRaisesRegex(ConfigurationError, "not overwritten"):
            initialize_config()
        validated_overrides(read_private_config())

    def test_missing_default_is_optional_but_explicit_missing_is_an_error(self):
        with self.assertRaisesRegex(ConfigurationError, "missing"):
            read_private_config()
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root)}):
            with patch.dict(os.environ):
                os.environ.pop("FAQ_CONFIG", None)
                self.assertEqual(config_path(), self.root / "ragpipe" / "config.toml")
                self.assertEqual(read_private_config(), {})

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_insecure_file_is_rejected_before_values_are_read(self):
        self.write('not valid TOML contains secret-value')
        self.path.chmod(0o644)
        with self.assertRaisesRegex(ConfigurationError, "owner-only") as error:
            read_private_config()
        self.assertNotIn("secret-value", str(error.exception))

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_another_owner_is_rejected(self):
        self.write("[runtime]\noffline = true\n")
        with patch("private_config.os.getuid", return_value=self.path.stat().st_uid + 1):
            with self.assertRaisesRegex(ConfigurationError, "belong to your account"):
                read_private_config()

    def test_symlink_is_not_followed(self):
        self.write("[runtime]\noffline = true\n")
        link = self.root / "link.toml"
        link.symlink_to(self.path)
        with self.assertRaisesRegex(ConfigurationError, "symlink"):
            read_private_config(link)

    def test_unknown_settings_and_invalid_types_fail(self):
        invalid = [{"runtime": {"ofline": True}}, {"retreival": {}},
                   {"runtime": {"device": "gpu"}}, {"runtime": {"offline": "true"}},
                   {"retrieval": {"cache_size": True}}, {"ingestion": {"max_pages": -1}},
                   {"models": {"profiles": {"bad": {"model": "foo", "trust_remote_code": True}}}},
                   {"models": {"profiles": {"bad": {"backend": "shell"}}}}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                validated_overrides(values)

    def test_syntax_error_does_not_echo_private_values(self):
        self.write('password = "do-not-print-me\n')
        with self.assertRaises(ConfigurationError) as error:
            read_private_config()
        self.assertNotIn("do-not-print-me", str(error.exception))

    def test_relative_paths_resolve_from_config_not_cwd(self):
        settings = validated_overrides({"paths": {"data_dir": "./data"}}, self.path)
        self.assertEqual(settings["DATA_DIRECTORY"], (self.path.parent / "data").resolve())
        profiles = validated_overrides({"models": {"profiles": {
            "local": {"model": "./models/local"}, "hub": {"model": "org/model"}}}}, self.path)
        self.assertEqual(profiles["PRIVATE_MODEL_PROFILES"]["local"]["model"],
                         str((self.path.parent / "models" / "local").resolve()))
        self.assertEqual(profiles["PRIVATE_MODEL_PROFILES"]["hub"]["model"], "org/model")
        settings = validated_overrides({"models": {"embedding": "./encoder", "reranker": "org/reranker"}}, self.path)
        self.assertEqual(settings["BI_ENCODER_MODEL"], str((self.path.parent / "encoder").resolve()))
        self.assertEqual(settings["CROSS_ENCODER_MODEL"], "org/reranker")

    def test_effective_paths_models_and_environment_precedence(self):
        self.write('''[paths]
data_dir = "./store"
[runtime]
device = "cpu"
collection = "lab"
offline = true
[models]
default = "staff"
embedding_revision = "abc123"
reranker_revision = "def456"
[models.profiles.staff]
model = "/staff/models/local"
max_new_tokens = 512
''')
        code = '''import json, config, model_setup, local_store
print(json.dumps([str(local_store.data_root()), model_setup.resolve_profile().name,
 config.DEVICE_REQUEST, config.DEFAULT_COLLECTION, model_setup.offline_mode(),
 model_setup.retrieval_revision(config.BI_ENCODER_MODEL),
 model_setup.retrieval_revision(config.CROSS_ENCODER_MODEL)]))'''
        with patch.dict(os.environ):
            for name in ("FAQ_DATA_DIR", "FAQ_LLM_MODEL", "HF_HUB_OFFLINE"):
                os.environ.pop(name, None)
            result = json.loads(self.run_code(code).stdout)
            self.assertEqual(result, [str((self.path.parent / "store").resolve()), "staff", "cpu", "lab", True,
                                      "abc123", "def456"])
            overridden = json.loads(self.run_code(code, FAQ_DATA_DIR=str(self.root / "env-store"),
                                                  FAQ_LLM_MODEL="flan-small", HF_HUB_OFFLINE="0").stdout)
            self.assertEqual(overridden[:2], [str((self.root / "env-store").resolve()), "flan-small"])
            self.assertFalse(overridden[4])

    def test_config_commands_do_not_import_ml_libraries_or_dump_values(self):
        self.write('[models]\ndefault = "staff-secret-profile"\n')
        code = '''import sys
from private_config import config_command
config_command("check")
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
'''
        result = self.run_code(code)
        self.assertIn("Config valid", result.stdout)
        self.assertNotIn("staff-secret-profile", result.stdout)

    def test_saved_model_and_cli_outrank_private_default(self):
        self.write('[models]\ndefault = "retrieval-only"\n')
        code = '''import model_setup
assert model_setup.resolve_profile().name == "retrieval-only"
model_setup.save_selection("flan-small")
assert model_setup.resolve_profile().name == "flan-small"
assert model_setup.resolve_profile("flan-base").name == "flan-base"
'''
        self.run_code(code, FAQ_DATA_DIR=str(self.root / "store"), FAQ_LLM_MODEL="")

    def test_collection_environment_is_not_reset_by_cli_default(self):
        self.write('[runtime]\ncollection = "private-lab"\n')
        code = '''import os, sys, terminal_setup
sys.argv = ["main.py"]
terminal_setup.command_line()
print(os.environ["FAQ_COLLECTION"])
sys.argv = ["main.py", "--collection", "icer"]
terminal_setup.command_line()
print(os.environ["FAQ_COLLECTION"])
'''
        lines = self.run_code(code, FAQ_DATA_DIR=str(self.root / "store"), FAQ_COLLECTION="env-lab").stdout.splitlines()
        self.assertEqual(lines, ["env-lab", "icer"])


if __name__ == "__main__":
    unittest.main()

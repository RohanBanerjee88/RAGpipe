"""Owner-only TOML settings; importing this module never loads ML libraries."""

import os
import stat
import tomllib
from pathlib import Path


class ConfigurationError(ValueError):
    pass


def config_path():
    explicit = os.getenv("FAQ_CONFIG")
    if explicit:
        return Path(explicit).expanduser().absolute()
    base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return base / "ragpipe" / "config.toml"


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(low, high):
    return lambda value: type(value) is int and low <= value <= high


def _number(low, high):
    return lambda value: type(value) in (int, float) and low <= value <= high


# Only settings used by the application are accepted. Typos must not look applied.
OPTIONS = {
    "paths.data_dir": ("DATA_DIRECTORY", _text),
    "paths.faq_json": ("FAQ_JSON_PATH", _text),
    "paths.trace_file": ("TRACE_PATH", _text),
    "models.profile_file": ("MODEL_PROFILES_PATH", _text),
    "models.default": ("DEFAULT_GENERATION_PROFILE", _text),
    "models.embedding": ("BI_ENCODER_MODEL", _text),
    "models.embedding_revision": ("BI_ENCODER_REVISION", _text),
    "models.reranker": ("CROSS_ENCODER_MODEL", _text),
    "models.reranker_revision": ("CROSS_ENCODER_REVISION", _text),
    "runtime.device": ("DEVICE_REQUEST", lambda v: v in ("auto", "cpu", "cuda")),
    "runtime.offline": ("OFFLINE_MODE", lambda v: type(v) is bool),
    "runtime.collection": ("DEFAULT_COLLECTION", _text),
    "runtime.debug": ("DEBUG_MODE", lambda v: type(v) is bool),
    "retrieval.bm25_top_k": ("BM25_TOP_K", _integer(1, 1000)),
    "retrieval.semantic_top_k": ("SEMANTIC_TOP_K", _integer(1, 1000)),
    "retrieval.candidate_k": ("HYBRID_CANDIDATE_K", _integer(1, 1000)),
    "retrieval.final_top_k": ("FINAL_TOP_K", _integer(1, 20)),
    "retrieval.cache_size": ("RETRIEVAL_CACHE_SIZE", _integer(0, 10000)),
    "retrieval.bm25_weight": ("RRF_BM25_WEIGHT", _number(0.01, 10)),
    "retrieval.semantic_weight": ("RRF_SEMANTIC_WEIGHT", _number(0.01, 10)),
    "ingestion.chunk_tokens": ("CHUNK_TOKENS", _integer(32, 512)),
    "ingestion.sample_rows": ("SAMPLE_ROWS", _integer(0, 20)),
    "ingestion.max_pages": ("WEB_MAX_PAGES", _integer(1, 1000)),
    "ingestion.timeout_seconds": ("WEB_TIMEOUT_SECONDS", _integer(1, 120)),
    "ingestion.max_bytes": ("WEB_MAX_BYTES", _integer(1024, 50 * 1024 * 1024)),
    "ingestion.scrape_urls": ("SCRAPE_URLS", lambda v: isinstance(v, list) and bool(v)
                               and all(_text(url) and url.startswith(("http://", "https://")) for url in v)),
    "generation.max_new_tokens": ("LLAMA_MAX_NEW_TOKENS", _integer(1, 4096)),
    "generation.context_tokens": ("MAX_CONTEXT_TOKENS", _integer(128, 16384)),
    "observability.trace_enabled": ("TRACE_ENABLED", lambda v: type(v) is bool),
    "responses.support_link": ("SUPPORT_LINK", _text),
}

PROFILE_KEYS = {"model", "revision", "backend", "max_new_tokens"}


def read_private_config(path=None):
    explicit = path is not None or bool(os.getenv("FAQ_CONFIG"))
    path = Path(path) if path is not None else config_path()
    try:
        # Reject links and insecure permissions before reading any private values.
        info = path.lstat()
    except FileNotFoundError:
        if explicit:
            raise ConfigurationError(f"Private config is missing: {path}. Run python -m private_config init.")
        return {}
    if not stat.S_ISREG(info.st_mode):
        raise ConfigurationError(f"Private config must be a regular file, not a symlink: {path}")
    if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o777 not in {0o400, 0o600}):
        raise ConfigurationError(f"Private config must belong to your account and be owner-only: {path}. "
                                 "Use chmod 600 on this file.")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            current = os.fstat(stream.fileno())
            if (current.st_dev, current.st_ino, current.st_mode, current.st_uid) != (
                    info.st_dev, info.st_ino, info.st_mode, info.st_uid):
                raise ConfigurationError(f"Private config changed while opening it: {path}. Retry after editing.")
            return tomllib.load(stream)
    except (tomllib.TOMLDecodeError, OSError):
        # Parser messages may quote a line containing private values.
        raise ConfigurationError(f"Cannot read private TOML config: {path}. Check syntax and permissions.") from None


def validated_overrides(values, path=None):
    path = Path(path) if path is not None else config_path()
    overrides = {}
    for section, table in values.items():
        if section not in {key.split(".")[0] for key in OPTIONS}:
            raise ConfigurationError(f"Unknown private config section: {section}")
        if not isinstance(table, dict):
            raise ConfigurationError(f"Config section {section!r} must be a TOML table")
        for key, value in table.items():
            dotted = f"{section}.{key}"
            if dotted == "models.profiles":
                if not isinstance(value, dict):
                    raise ConfigurationError("models.profiles must contain named profile tables")
                private_profiles = {}
                for name, profile in value.items():
                    if (not _text(name) or not isinstance(profile, dict)
                            or set(profile) - PROFILE_KEYS
                            or any(not _text(v) for k, v in profile.items() if k != "max_new_tokens")
                            or profile.get("backend", "transformers") not in {"none", "transformers"}
                            or (profile.get("backend") != "none" and not profile.get("model"))
                            or ("max_new_tokens" in profile and not _integer(1, 4096)(profile["max_new_tokens"]))):
                        raise ConfigurationError(f"Invalid private model profile: {name}")
                    profile = dict(profile)
                    if profile.get("model", "").startswith(("./", "../", "~")):
                        model_path = Path(profile["model"]).expanduser()
                        profile["model"] = str((path.parent / model_path).resolve()
                                               if not model_path.is_absolute() else model_path)
                    private_profiles[name] = profile
                overrides["PRIVATE_MODEL_PROFILES"] = private_profiles
                continue
            if dotted not in OPTIONS:
                raise ConfigurationError(f"Unknown private config setting: {dotted}")
            name, valid = OPTIONS[dotted]
            if not valid(value):
                raise ConfigurationError(f"Invalid value for private config setting: {dotted}")
            if section == "paths" or dotted == "models.profile_file":
                value = Path(value).expanduser()
                value = (path.parent / value).resolve() if not value.is_absolute() else value
            elif dotted in {"models.embedding", "models.reranker"} and value.startswith(("./", "../", "~")):
                value = str((path.parent / Path(value).expanduser()).resolve())
            overrides[name] = value
    return overrides


TEMPLATE = '''# Private RAGpipe settings. Do not commit or share this file.
# Unspecified settings keep the repository defaults. Restart after editing.
# Relative paths below are resolved beside this file, not your shell directory.

[paths]
# data_dir = "./data"
# faq_json = "/absolute/path/to/RAGpipe/all_faqs.json"
# trace_file = "./retrieval-traces.jsonl"

[runtime]
device = "auto" # auto, cpu, cuda
offline = false
collection = "auto"
debug = false

[models]
# default = "retrieval-only"
# profile_file = "/absolute/path/to/model_profiles.toml"
# embedding = "all-MiniLM-L6-v2"
# embedding_revision = "main"
# reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# reranker_revision = "main"

# Private profiles extend or replace public profiles with the same name.
# [models.profiles.lab-local]
# model = "/absolute/path/to/approved/model"
# max_new_tokens = 512

[retrieval]
bm25_top_k = 20
semantic_top_k = 20
candidate_k = 20
final_top_k = 3
cache_size = 128
# bm25_weight = 1.0
# semantic_weight = 1.0

[ingestion]
chunk_tokens = 64
sample_rows = 5
max_pages = 50
timeout_seconds = 15
max_bytes = 5242880
# scrape_urls = ["https://docs.icer.msu.edu/Frequently_Asked_Questions_FAQ_/"]

[generation]
max_new_tokens = 200
context_tokens = 1536

[observability]
trace_enabled = false # Traces contain questions and source locations.

[responses]
# support_link = "https://contact.icer.msu.edu/contact"
'''


def initialize_config(path=None):
    path = Path(path) if path is not None else config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ConfigurationError(f"Config already exists; it was not overwritten: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(TEMPLATE)
    return path


def config_command(action):
    if action == "init":
        print(f"Created owner-only config: {initialize_config()}")
    elif action == "path":
        print(config_path())
    else:
        values = read_private_config()
        overrides = validated_overrides(values)
        if not config_path().exists():
            print(f"No private config at {config_path()}; repository defaults apply.")
        else:
            print(f"Config valid: {config_path()} ({len(overrides)} overrides; values hidden)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "path", "check"))
    try:
        config_command(parser.parse_args().action)
    except ConfigurationError as exc:
        parser.exit(1, f"Configuration error: {exc}\n")

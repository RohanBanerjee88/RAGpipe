"""Named, cached generation profiles and explicit model preparation."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from config import BI_ENCODER_MODEL, CROSS_ENCODER_MODEL, PROJECT_ROOT
from local_store import data_root, read_json, write_json


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str = ""
    revision: str = "main"
    backend: str = "transformers"


_active = None
_offline = False


def profiles():
    path = Path(os.getenv("FAQ_MODELS_CONFIG", PROJECT_ROOT / "model_profiles.toml"))
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    result = {}
    for name, values in config["models"].items():
        profile = ModelProfile(name=name, **values)
        if profile.backend not in {"none", "transformers"}:
            raise ValueError(f"Unsupported backend for {name}: {profile.backend}")
        if profile.backend != "none" and not profile.model:
            raise ValueError(f"Profile {name} needs a model repository or directory")
        result[name] = profile
    return config.get("default", "flan-base"), result


def resolve_profile(selection=None):
    default, available = profiles()
    saved = read_json(data_root() / "settings.json").get("model")
    name = selection or os.getenv("FAQ_LLM_MODEL") or saved or default
    return available.get(name) or ModelProfile(name=name, model=name)


def configure_session(selection=None, offline=False):
    global _active, _offline
    _active = resolve_profile(selection)
    _offline = offline or os.getenv("HF_HUB_OFFLINE", "0") == "1"
    if _offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return _active


def active_profile():
    return _active or resolve_profile()


def offline_mode():
    return _offline or os.getenv("HF_HUB_OFFLINE", "0") == "1"


def save_selection(selection):
    resolve_profile(selection)
    settings = read_json(data_root() / "settings.json")
    settings["model"] = selection
    write_json(data_root() / "settings.json", settings)


def repository_name(model):
    return f"sentence-transformers/{model}" if model == BI_ENCODER_MODEL and "/" not in model else model


def model_location(model, revision="main", download=False):
    """Resolve a pinned prepared snapshot, with no implicit network at startup."""
    from huggingface_hub import snapshot_download

    local = Path(model).expanduser()
    if local.is_dir():
        return str(local.resolve())
    if local.is_absolute() or model.startswith(("./", "../", "~")):
        raise FileNotFoundError(f"Local model directory does not exist: {model}")
    model = repository_name(model)
    prepared = read_json(data_root() / "prepared_models.json")
    key = f"{model}@{revision}"
    ref = prepared.get(key, revision)
    path = snapshot_download(
        repo_id=model, revision=ref, local_files_only=not download or offline_mode(),
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot", "*.gguf"],
    )
    if download:
        prepared[key] = Path(path).name
        write_json(data_root() / "prepared_models.json", prepared)
    return path


def check_weights(path):
    path = Path(path)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {path}")
    indexes = list(path.glob("*.index.json"))
    if indexes:
        for index in indexes:
            shards = read_json(index).get("weight_map", {})
            if not shards:
                raise FileNotFoundError(f"Empty model shard index: {index}")
            for shard in set(shards.values()):
                if not (path / shard).is_file():
                    raise FileNotFoundError(f"Missing model shard: {shard}")
    elif not any(path.glob("*.safetensors")) and not (path / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Missing model weights in {path}")


def generation_location():
    profile = active_profile()
    if profile.backend == "none":
        raise RuntimeError("Retrieval-only mode: generation is disabled")
    try:
        location = model_location(profile.model, profile.revision)
        check_weights(location)
        return location
    except Exception as exc:
        raise RuntimeError(
            f"Model {profile.name} is unavailable. Run: python main.py models prepare "
            f"{profile.name}. Source excerpts remain available. ({exc})"
        ) from exc


def retrieval_location(model):
    try:
        location = model_location(model)
        check_weights(location)
        return location
    except Exception as exc:
        raise RuntimeError(
            f"Retrieval model {model} is unavailable. Run: "
            "python main.py models prepare retrieval-only before searching."
        ) from exc


def model_status(profile):
    if profile.backend == "none":
        return "ready (no generator)"
    try:
        location = model_location(profile.model, profile.revision)
        check_weights(location)
        return "downloaded"
    except Exception:
        if Path(profile.model).is_absolute():
            return "unavailable (local files missing or incomplete)"
        return "configured (prepare required)"


def prepare_models(selection):
    from transformers import AutoConfig, AutoTokenizer
    profile = resolve_profile(selection)
    required = [(BI_ENCODER_MODEL, "main"), (CROSS_ENCODER_MODEL, "main")]
    if profile.backend != "none":
        required.append((profile.model, profile.revision))
    for model, revision in required:
        print(f"Preparing {model}...")
        location = model_location(model, revision, download=True)
        check_weights(location)
        AutoConfig.from_pretrained(location, local_files_only=True, trust_remote_code=False)
        AutoTokenizer.from_pretrained(location, local_files_only=True, trust_remote_code=False)
    print(f"Ready: {profile.name}. Cached weights are reused on subsequent starts.")

"""Incremental, model-specific embedding caches for independent collections."""

import hashlib
import os
import tempfile
from pathlib import Path

import torch
from filelock import FileLock

from ingestion import sha256_text
from local_store import data_root


def model_fingerprint(location):
    path = Path(location)
    # A Hub snapshot path identifies a fixed commit. Local models need file metadata too.
    identity = [str(path.resolve())]
    for file in sorted(path.rglob("*")):
        if file.is_file():
            stat = file.stat()
            identity.append(f"{file.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns}")
    return sha256_text("\n".join(identity))


def cached_embeddings(name, texts, encoder, fingerprint, device):
    cache = data_root() / "collections" / name / "embeddings.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    keys = [sha256_text(text) for text in texts]
    with FileLock(str(cache) + ".lock"):
        old = {}
        if cache.exists():
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            if payload.get("model") == fingerprint:
                old = payload.get("vectors", {})
        missing = dict((key, text) for key, text in zip(keys, texts) if key not in old)
        if missing:
            encoded = encoder.encode(list(missing.values()), convert_to_tensor=True,
                                     show_progress_bar=False, batch_size=64).detach().cpu()
            old.update(zip(missing, encoded))
        vectors = {key: old[key] for key in dict.fromkeys(keys)}
        if missing or set(old) != set(vectors) or not cache.exists():
            fd, temporary = tempfile.mkstemp(dir=cache.parent, suffix=".pt")
            os.close(fd)
            try:
                torch.save({"model": fingerprint, "vectors": vectors}, temporary)
                os.replace(temporary, cache)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return torch.stack([vectors[key] for key in keys]).to(device), len(missing)

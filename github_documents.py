"""Read public GitHub documentation at one immutable commit, without executing it."""

from pathlib import PurePosixPath
import re
from urllib.parse import quote, unquote, urlsplit

import requests

from ingestion import sha256_text


TIMEOUT = 15
MAX_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {".md", ".markdown", ".rmd", ".txt"}


def github_target(url):
    parsed = urlsplit(url)
    if parsed.hostname != "github.com":
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 4 or parts[2] not in {"tree", "blob"}:
        raise ValueError("Use a GitHub tree/<revision>/<folder> or blob/<revision>/<file> URL")
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts[:2]):
        raise ValueError("Invalid GitHub repository name")
    return parts[0], parts[1], parts[2], [unquote(part) for part in parts[3:]]


def github_documents(url, max_files=50, session=None):
    """Return text files, stable logical keys, pinned citations, and completeness."""
    if max_files < 1:
        raise ValueError("max_pages must be at least 1")
    owner, repo, kind, tail = github_target(url)
    client = session or requests.Session()
    client.headers.update({"User-Agent": "RAGpipe/1.0", "Accept": "application/vnd.github+json"})
    api = f"https://api.github.com/repos/{owner}/{repo}"

    def get(address):
        response = client.get(address, timeout=TIMEOUT)
        if response.status_code in {403, 429}:
            raise ValueError("GitHub API access/rate limit reached; retry later or import a local checkout")
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            raise ValueError(f"GitHub response exceeds {MAX_BYTES} bytes: {address}")
        return response

    # Try longest refs first so feature/branch references are not treated as folders.
    for split in range(len(tail), 0, -1):
        ref = "/".join(tail[:split])
        try:
            commit = get(f"{api}/commits/{quote(ref, safe='')}").json()["sha"]
            path = "/".join(tail[split:])
            break
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code not in {404, 422}:
                raise
    else:
        raise ValueError("GitHub revision was not found; check the URL or import a local checkout")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("GitHub did not return a valid commit ID")
    inventory = get(f"{api}/git/trees/{commit}?recursive=1").json()
    if inventory.get("truncated"):
        raise ValueError("GitHub file inventory is truncated; import a local checkout instead")
    entries = [entry for entry in inventory["tree"] if entry["type"] == "blob"
               and entry.get("mode") != "120000"
               and (entry["path"] == path if kind == "blob" else
                    not path or entry["path"].startswith(path.rstrip("/") + "/"))]
    documents = sorted((entry for entry in entries
                        if PurePosixPath(entry["path"]).suffix.lower() in TEXT_SUFFIXES),
                       key=lambda entry: entry["path"])
    if not documents:
        raise ValueError("No Markdown or plain-text documents found in the requested GitHub path")
    warnings = []
    if len(entries) > len(documents):
        warnings.append(f"Skipped {len(entries) - len(documents)} non-text assets; GitHub imports read Markdown/RMarkdown/text only")
    complete = len(documents) <= max_files
    if not complete:
        warnings.append(f"Stopped after {max_files} files; increase --max-pages to import more")
    files = []
    for entry in documents[:max_files]:
        filename = entry["path"]
        encoded = quote(filename, safe="/")
        key = f"https://github.com/{owner}/{repo}/blob/{quote(ref, safe='')}/{encoded}"
        citation = f"https://github.com/{owner}/{repo}/blob/{commit}/{encoded}"
        raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{encoded}"
        try:
            if entry.get("size", 0) > MAX_BYTES:
                raise ValueError("Document exceeds the size limit")
            response = get(raw)
            text = response.content.decode("utf-8-sig")
            files.append((key, sha256_text(text), text, {
                "url": citation, "repository": f"{owner}/{repo}",
                "repository_path": filename, "source_revision": commit,
            }))
        except (requests.RequestException, ValueError, UnicodeError) as exc:
            complete = False
            warnings.append(f"Skipped {filename}: {exc}; previous records retained")
    return files, complete, warnings

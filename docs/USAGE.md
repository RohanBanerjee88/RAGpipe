# Operational User Guide

RAGpipe searches local collections and answers from indexed evidence. It can
return complete FAQ answers, cited document excerpts, or grounded model output;
it abstains when the sources do not support the question. Run the commands below
from the repository root with the project's Python environment active.

- [Fresh CPU setup](#fresh-cpu-setup)
- [Private configuration](#private-configuration)
- [Models and offline operation](#models-and-offline-operation)
- [Import and manage collections](#import-and-manage-collections)
- [Ask questions and read answers](#ask-questions-and-read-answers)
- [Validation and evaluation](#validation-and-evaluation)
- [Routine changes and troubleshooting](#routine-changes-and-troubleshooting)

## Fresh CPU Setup

Use macOS or Linux with Python 3.11. Initial dependency installation and model
preparation need internet access. No GPU or gated generator is required. For
the separate HPCC CUDA/Conda profile, follow the
[README GPU setup](../README.md#hpcc--conda-setup), not the CPU requirements below.

For a new checkout:

```bash
git clone https://github.com/RohanBanerjee88/RAGpipe.git
cd RAGpipe
# Check out the review/release branch containing this guide before continuing.
```

During this phase, use a `RefN` review checkout, based on `segM`. A default
`main` checkout does not include these changes yet. Existing review checkouts
can skip cloning. RefN's integration target is `segM`, not `main`.

Create an isolated CPU environment. Do not delete an existing working environment
or corpus to follow these steps; choose another environment name if necessary.

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
export FAQ_DEVICE=cpu
```

Resolve environment-check failures before proceeding. Use `python -m pip` so
packages are installed into the active environment rather than a system Python.
In a later shell, activate that environment again and set `FAQ_DEVICE=cpu`, or
save `runtime.device = "cpu"` in the private configuration.

Prepare retrieval models explicitly, then optionally add the small default
generator:

```bash
python main.py models list
python main.py models prepare retrieval-only
python main.py models prepare flan-base
```

`retrieval-only` prepares the embedding encoder and reranker without a generator.
`flan-base` prepares those same retrieval models plus `google/flan-t5-base`.
Alternatively, `python main.py setup` guides preparation and saves the selected
profile. Preparation does not scrape, replace, or reimport the corpus.

**Keep existing data.** An existing `all_faqs.json` is automatically exposed as
the `icer` collection; existing collections and embeddings under `.ragpipe/`
are reused. Do not rescrape just to adopt this guide or private configuration.
Legacy `faq_embeddings.pt` is left untouched. If a genuinely fresh checkout has
no ICER corpus and you want ICER documentation, build it once:

```bash
python scrape.py
```

This requires network access and creates the FAQ corpus and optional tree. On
an existing installation it is a deliberate corpus refresh, not a setup step.
You can instead import your own documents without scraping ICER.

```bash
python main.py collections list
python main.py --offline --collection auto --model retrieval-only
```

## Private Configuration

Private settings avoid editing shared source files. Configuration is optional:
without a default private file, repository defaults apply. Settings are read
once at process startup. After editing, stop the assistant and start it once
again; there is no live reload.

### Initialize and Check

The lightweight module uses Python 3.11's standard library and does not load
inference libraries or weights. It can be used before installing ML dependencies:

```bash
python3.11 -m private_config path
python3.11 -m private_config init
python3.11 -m private_config check
```

With the project environment active, the equivalent application commands are:

```bash
python main.py config path
python main.py config init
python main.py config check
```

The default path is `~/.config/ragpipe/config.toml`. If `XDG_CONFIG_HOME` is set,
the path is `$XDG_CONFIG_HOME/ragpipe/config.toml`. `path` reports the selected
location; `check` validates syntax, keys, values, ownership, and permissions
without printing private setting values. It does not download models or prove
that configured model/data paths are usable.

To select an explicit file, use an absolute `FAQ_CONFIG` path. Initialize it
through the lightweight module before running `main.py`: an explicitly selected
missing file is an error, not a request to use defaults.

```bash
export FAQ_CONFIG="$HOME/.config/ragpipe/config.private.toml"
python -m private_config init
python -m private_config path
python -m private_config check
python main.py --offline --model retrieval-only
```

Keep the export in every shell that should use this file. Unsetting `FAQ_CONFIG`
returns to default-path discovery. `init` never overwrites an existing file and
does not follow a config-file symlink. If initialization says the file exists,
edit and check that file instead of deleting it to rerun setup.

### Permissions and Privacy

The config must be a regular file owned by the account running RAGpipe, with
mode `0600` (owner read/write) or `0400` (owner read-only). Symlinks, another
owner, or group/world access are rejected. Initialization creates the file as
`0600` and creates a new default configuration directory as `0700`; it does not
repair permissions on an existing directory. For an owner-controlled regular
file, use `chmod 600 /absolute/path/to/config.toml` before editing, or `chmod 400`
after editing for read-only use. Do not change ownership of someone else's file
to bypass a failure; create a configuration under your own account.

This is **not authentication, authorization, or ICER role enforcement**. Each
staff member should use their own OS account and private configuration/store.
Anyone sharing that OS account, and an administrator, can read the file and its
data. Collections and enable/disable settings are not access-control boundaries.
Owner-only config permissions do not automatically secure imported documents,
the data directory, model caches, or trace files; protect those separately with
appropriate OS permissions and institutional policy.

Do not share or commit private configuration, private documents, stores, or
traces. Do not put passwords, access tokens, or Hugging Face tokens in this TOML
file; use Hugging Face authentication tooling and its credential storage for
approved gated models. Tracing is off by default because traces contain user
questions and source locations. Debug output can also expose sensitive content.

### Settings Reference

All settings are optional. Unknown keys, malformed TOML, invalid types, and
out-of-range values fail validation rather than being silently ignored. Use
TOML booleans (`false`, not `"false"`) and integers where required. Relative
filesystem settings resolve beside the config file, not the shell's current
directory. Prefer absolute local model paths so they cannot be mistaken for
Hub repository names.

The following reference shows the supported tables. Commented paths, revisions,
URLs, and model choices are examples to adapt, not instructions to move data.
Keep the file produced by `init` and edit only the settings you need.

```toml
[paths]
# data_dir = "/absolute/path/to/RAGpipe/.ragpipe"
# faq_json = "/absolute/path/to/RAGpipe/all_faqs.json"
# trace_file = "./retrieval-traces.jsonl"

[runtime]
device = "auto" # "auto", "cpu", or "cuda"
offline = false
collection = "auto"
debug = false

[models]
# default = "flan-base"
# profile_file = "/absolute/path/to/model_profiles.toml"
# embedding = "all-MiniLM-L6-v2"
# embedding_revision = "main"
# reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# reranker_revision = "main"

# Private profiles extend public profiles or replace the same profile name.
# [models.profiles.lab-local]
# model = "/absolute/path/to/approved/model"
# revision = "main" # Optional; a pinned Hub revision can be used instead.
# backend = "transformers" # "none" disables generation; otherwise "transformers".
# max_new_tokens = 512

[retrieval]
bm25_top_k = 20
semantic_top_k = 20
candidate_k = 20
final_top_k = 3
cache_size = 128
bm25_weight = 1.0
semantic_weight = 1.0

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
trace_enabled = false

[responses]
# support_link = "https://contact.icer.msu.edu/contact"
```

| Table | Operational meaning |
| --- | --- |
| `paths` | `data_dir` contains collection manifests, embeddings, preparation records, and saved setup selection; the default is the checkout's `.ragpipe/`. `faq_json` defaults to the checkout's `all_faqs.json`. `trace_file` defaults to `logs/retrieval_traces.jsonl` in the checkout. |
| `runtime` | `device` selects validated CUDA or CPU (`auto` can fall back to CPU; `cuda` requires a working GPU). `offline` selects cached model files only. `collection` is `auto` or a collection name. `debug` enables diagnostic output. |
| `models` | `default` selects a generation profile, subject to precedence below. `profile_file` selects the public profile TOML. `embedding`/`reranker` select retrieval models; their `_revision` settings select revisions, defaulting to `main`. |
| `models.profiles.lab-local` | A private named profile with `model`, optional `revision`, `backend` (`none` or `transformers`), and optional `max_new_tokens` (1-4096). A `none` profile needs no model. Other profiles need a Hub model or local Transformers directory. |
| `retrieval` | Candidate counts control keyword/semantic retrieval and fusion (1-1000); `final_top_k` is 1-20. `cache_size` is 0-10000, with 0 disabling the repeated-query cache. Fusion weights must be numbers from 0.01 to 10. |
| `ingestion` | `chunk_tokens` is 32-512; `sample_rows` is 0-20. Web limits are `max_pages` (1-1000), `timeout_seconds` (1-120), and `max_bytes` (1024-52428800). `scrape_urls` is a nonempty list of HTTP(S) URLs for the ICER scrape, not a request to import on startup. |
| `generation` | Global `max_new_tokens` is 1-4096; `context_tokens` is 128-16384. Actual input packing also respects the selected model's capacity. |
| `observability` | `trace_enabled` opts into sensitive query/event traces; leave it `false` for normal use. |
| `responses` | `support_link` changes the fallback support destination, not permissions or evidence requirements. |

There is no private-config option to lower evidence thresholds or disable
grounding. Retrieval weights and counts are not evidence-gate overrides.
Larger candidate counts, sample limits, and context budgets can increase import
cost, response size, memory, or latency. Keep the defaults until a measured
evaluation shows a reason to change them.

### Precedence and Existing Data

Generation model selection, from highest to lowest priority:

1. CLI `--model`.
2. `FAQ_LLM_MODEL`.
3. The selection saved by `python main.py setup` in the active data directory.
4. Private `models.default`.
5. The public profile file's default (`flan-base` in the repository).

Thus, changing `models.default` will not replace a saved setup choice. Use
`--model` for an explicit session or rerun `setup` to save another choice.
The selected profile's `max_new_tokens`, when present, takes precedence over
global `generation.max_new_tokens`.

For other settings, an available CLI flag takes precedence over its existing
environment variable, then the private setting, then the repository default.
In particular, `FAQ_DATA_DIR`, `FAQ_COLLECTION`, `FAQ_DEVICE`, and
`HF_HUB_OFFLINE` override their private counterparts. Use `HF_HUB_OFFLINE=1` for
cached-model operation or `0` to override a private offline default. There is no
`--device` flag: use `FAQ_DEVICE` or `runtime.device`. Put global CLI flags such
as `--offline`, `--model`, and `--collection` before a management subcommand.
Existing `FAQ_MODELS_CONFIG`, `FAQ_TRACE_ENABLED`, and `FAQ_TRACE_PATH` overrides
also remain effective. In particular, an exported `FAQ_TRACE_ENABLED=1` can turn
tracing on even when the private setting is `false`.

Initializing private settings does not move or delete existing data. Leave
`paths` unset to preserve the checkout's existing corpus/store, or point them
explicitly at that corpus/store. Selecting a different `data_dir` chooses a
different store and saved model selection; it does not migrate the old one.
`faq_json` is independent of `data_dir`, so changing only the latter does not
relocate the ICER JSON. Do not copy private stores into a shared directory.

## Models and Offline Operation

```bash
python main.py models list
python main.py models prepare retrieval-only
python main.py models prepare flan-base
python main.py setup
```

`models list` reports configured generator profiles and cached availability; a
retrieval-only profile's "no generator" status is not proof that both retrieval
models are ready. Use `models prepare retrieval-only` to verify/prepare them.
`setup` prepares and saves the chosen profile; `models prepare` prepares weights
without changing the saved selection.

Ordinary startup uses cached weights only, even without `--offline`; missing
weights are downloaded only by explicit preparation. Prepare while online
before requesting offline operation. Hugging Face uses its persistent cache
(`HF_HOME` can relocate it); resolved preparation revisions are recorded under
the active data directory. A new empty cache needs preparation again.

```bash
python main.py --offline --model retrieval-only --collection auto
python main.py --offline --model flan-base --collection icer
python main.py models prepare lab-local
python main.py --offline --model lab-local --collection atlas
python main.py --offline --model /absolute/path/to/approved/model --collection atlas
```

The last three examples assume the private `lab-local` profile or local model
directory has actually been configured/provided. A local Transformers model
must have its configuration, tokenizer, and complete weights. Hub models must
be compatible with the installed Transformers loader. GGUF, Ollama, remote API
backends, and models requiring custom executable code are not supported.

Only one generator is loaded lazily in a session. Restart to change it. Missing
generation weights can still allow cited excerpts; missing embedding/reranker
weights block retrieval and require preparation. `--offline` is a model-loading
policy, not a network sandbox: explicit scraping and website/GitHub imports
still require network access.

Changing the generator does **not** re-embed the corpus. Changing the embedding
model or its revision invalidates the embedding cache and requires preparation
and rebuilding vectors. Reimport external collections with the new encoder to
apply its tokenizer to chunking. Changing chunk size or table sample settings
also requires reimport; a restart alone cannot rechunk stored passages. The
original FAQ answers remain complete for direct answers.

## Import and Manage Collections

Choose a short collection name using letters, numbers, underscores, or hyphens,
starting with a letter or number (maximum 64 characters). Use separate names for
independent labs or sources. `icer` is reserved for the existing ICER FAQ corpus.
Prepare retrieval models before local text import so token-based chunking is
available. Import commands report warnings: review them before relying on the
collection.

### Local Documents and Tables

Import a directory recursively, or add individual files to the same collection:

```bash
python main.py collections import atlas /absolute/path/to/lab-documents
python main.py collections import atlas /absolute/path/to/protocol.pdf
python main.py collections import atlas /absolute/path/to/handbook.md
python main.py collections import atlas /absolute/path/to/notes.txt
python main.py collections import atlas /absolute/path/to/faqs.json
python main.py collections import atlas /absolute/path/to/samples.csv
python main.py collections import atlas /absolute/path/to/measurements.tsv
python main.py collections import atlas /absolute/path/to/workbook.xlsx
python main.py collections list
```

Local repository imports skip Git metadata, virtual environments, dependency
directories, and symlinked files/directories. Markdown comments such as `<!--kb`
annotations and non-visible HTML are not answer evidence. After updating to this
parser, reimport external Markdown/GitHub collections once and restart; ICER does
not need rescraping. Reimporting a local repository also removes previously
indexed records from the excluded internal directories.

For a quieter import:

```bash
python main.py collections import atlas /path/to/docs --quiet
```

Warnings remain in the collection manifest; default terminal
output shows at most 20 individual warnings. Quiet mode does not hide failures
or imply that every file was read successfully.

| Input | What is indexed |
| --- | --- |
| Text PDF | Extracted text with page locations. Scanned/image-only PDFs need external OCR first. |
| Markdown / plain text | Source passages with headings/line locations where available. |
| FAQ JSON | A list of objects containing `question` and `answer`, or an object with a `faqs` list. FAQ answers remain complete. Arbitrary JSON datasets are not supported. |
| CSV / TSV / XLSX | Dataset descriptions from filenames, sheet names, headers, and at most five sample rows by default. Formula cells are labeled, not executed. |

For example, a FAQ JSON file can contain:

```json
[
  {"question": "Where is the lab handbook?", "answer": "The handbook is in the lab documentation directory."}
]
```

DOCX, raw sequencing data, scripts, binaries, and unreadable files are not
general-purpose supported inputs. Supply a data dictionary in Markdown or text
alongside each table, naming the dataset and defining its columns. Meanings are
not inferred from headers alone. Samples are not population summaries: the
assistant does not calculate full-dataset statistics, evaluate formulas, execute
analysis pipelines, or establish scientific conclusions from a few rows.

### Websites

```bash
python main.py collections import handbook https://example.edu/lab/handbook/ --max-pages 50
```

Replace the example URL with the narrowest useful documentation root. The
importer crawls linked server-rendered HTML in the same site/path scope, respects
`robots.txt`, ignores off-site/non-HTML assets, and removes navigation, scripts,
forms, and other page furniture. Citations retain page titles, headings, and
source URLs. `--max-pages` overrides the configured limit for that import; the
default is 50. Time and response-size limits also apply.

JavaScript-only content, authenticated websites/browser sessions, and unlinked
pages reachable only through a sitemap are not supported. A website crawl does
not download linked PDFs as local PDF imports; import those files separately.
Check warnings for blocked, capped, oversized, or failed pages.

### GitHub Documentation

Use public GitHub `tree` or `blob` URLs to import the actual documentation text,
not a rendered directory listing:

```bash
python main.py collections import bglr https://github.com/gdlc/BGLR-R/tree/master/inst/md --max-pages 50
python main.py collections import bglr https://github.com/gdlc/BGLR-R/blob/master/README.md
```

The second import adds the package overview; the first imports only the requested
examples subtree. Imports do not silently include the rest of the repository.
GitHub's API/raw-file service supplies Markdown, RMarkdown, and plain-text files;
citations are pinned to the resolved commit and line range. `--max-pages` limits
text files here. Code in documentation is indexed as text, never executed.
Images, scripts, and other unsupported assets are skipped and reported. For API
rate limits, use the reported local-checkout alternative and import the relevant
local documentation directory. Private/authenticated GitHub imports are not a
supported credential workflow.

### Reimport, Enable, Disable, and Restart

```bash
python main.py collections list
python main.py collections disable atlas
python main.py collections enable atlas
python main.py --offline --collection auto --model retrieval-only
```

Stop and restart every already-running assistant after import, reimport,
enable, or disable operations. An existing session keeps its in-memory snapshot.
Disabling a collection hides it from retrieval; it does not delete or secure its
documents. To exclude ICER, select a named lab collection with `--collection`;
ICER cannot be disabled with the collection command.

Reimporting unchanged content preserves versions and reuses compatible
embeddings. Changed content advances its version. A failed replacement retains
the last good records and prints warnings. Directory reimports synchronize files
deleted from that directory; importing one more file adds it to the collection.
Complete website/GitHub crawls can remove sources no longer present in the
requested scope; capped or failed imports retain previous affected records.
Review warnings and preserve original documents before an intentional refresh.
Other collections, including ICER, are not replaced by a lab import.

## Ask Questions and Read Answers

```bash
python main.py --offline --collection auto --model retrieval-only
python main.py --offline --collection atlas --model flan-base
```

`auto` searches enabled collections independently; it does not merge labs'
keyword statistics. A named collection restricts retrieval to that collection.
Prefer a named collection when the intended lab is known. In automatic mode,
name the lab or collection in the question to reduce ambiguity.

When several sources plausibly match, the assistant displays numbered choices.
Reply with the displayed number, such as `2`, to receive that source's excerpt,
or ask a more specific question. The numeric choice is for the pending
clarification, not a permanent collection switch. Explicit comparisons may
return labeled excerpts from multiple collections without reconciling conflicts.

Try source-specific questions such as:

```text
What is BGLR?
What types of censoring does BGLR support?
Which function reads binary samples of effects in BGLR?
```

These examples require the BGLR imports above. Use `stats` to inspect session
routing and `quit` to exit.

Read `[S1]`, `[S2]`, and the associated source entries as evidence references,
not confidence guarantees. Source metadata identifies the collection, source,
location, and version/commit where available. Open the original source when
checking a procedure or consequential claim. Import time is not proof that a
document is current, authoritative, or scientifically correct.

Strong FAQ matches return the source answer directly. Supported document
questions can return excerpts or invoke the generator. Unsupported questions
abstain rather than using general model knowledge. Execution requests, arbitrary
dataset analysis, computations over full tables, and running R/Python/shell code
are outside this assistant's scope. It can retrieve documented procedures but
does not execute them or verify that they work in your environment. Missing
identifiers or facts require better source material, not invented code.
For example, "How does this lab compute gene counts?" can be answered from a
documented method, and "How do I calculate the average signal?" can retrieve a
documented procedure. "Calculate the average across all samples for me" asks
for a dataset result, which the assistant does not compute.

**Generator limits:** small FLAN models often omit citations, paraphrase beyond
the literal support check, or fail to produce complete procedures. Invalid,
uncited, or unsupported output is replaced by cited source excerpts. Citation
validation is conservative literal-support checking, not a general entailment
model or proof of truth. A larger generator or context budget does not ensure
better grounded answers. Bounded procedure reconstruction can link to the
original when a complete source block and setup will not fit; do not treat a
partial snippet as an executable workflow.

## Validation and Evaluation

Run evaluations intentionally; they can take substantial CPU time. Keep reports
and evaluation stores separate from operational/private data. Reports may contain
questions, answers, and source locations even when tracing is disabled. The
examples below use `/tmp` and public/fixture material, not private lab questions.

### Unit and Mock/Fixture Checks

```bash
python -m unittest discover -s tests -v
python main.py models prepare retrieval-only
FAQ_DEVICE=cpu python scripts/smoke_setup.py
FAQ_DEVICE=cpu python scripts/evaluate_website.py
FAQ_DEVICE=cpu python scripts/evaluate_refn.py --output /tmp/ragpipe-refn-fixtures.json
FAQ_DEVICE=cpu python scripts/evaluate_segm.py --output /tmp/ragpipe-segm-fixtures.json
```

The unit suite uses deterministic mocks/fixtures for behavior and safety checks;
passing it is not a real-model accuracy claim. `smoke_setup.py` exercises offline
CLI startup in an isolated store using real, already-cached retrieval weights.
The website evaluator uses a fictional site and mocked HTTP responses, not a live
website, but performs real retrieval. The collection evaluator uses synthetic
labs plus the existing ICER corpus and real cached retrieval models; it requires
the ICER JSON. It offers `--growth` for the slower 1,000/10,000-passage exercise.
The RefN evaluator uses synthetic method/table documents with real retrieval,
checking supported tool/procedure excerpts against refusals for dataset results,
unknown identifiers, and source manipulation. It mocks generator loading and
asserts that these routes do not attempt to load a generator.
These scripts use isolated temporary application stores and do not download
weights; prepare models first.

Compare actual generators against **mock source material**:

```bash
python main.py models prepare flan-small
python main.py models prepare flan-base
FAQ_DEVICE=cpu python scripts/evaluate_models.py --profiles flan-small flan-base --output /tmp/ragpipe-models-mock.json
```

This calls real cached generators, but its default cases are synthetic fixtures
labeled `mock`; it is not a real-document retrieval benchmark. Reports separate
raw generation success from delivered answers that used excerpt fallbacks. The
model evaluator exits nonzero if a generator fails a raw case, even when fallback
delivery is useful. Do not count fallback success as generator success.

### Real-Document Checks

First prepare retrieval weights in the normal store. Then import public BGLR
documents into a separate reusable evaluation store and test retrieval/procedures:

```bash
python main.py models prepare retrieval-only
FAQ_DEVICE=cpu python scripts/evaluate_bglr.py --data-dir /tmp/ragpipe-bglr-check --include-readme --procedures --output /tmp/ragpipe-bglr-retrieval.json
```

The first BGLR run needs network access for public documents, not model weights.
Subsequent `--skip-import` runs reuse that exact imported snapshot with cached
models. Compare delivered answers with a generator, then raw generators against
both mock and retrieved BGLR contexts:

```bash
python main.py models prepare flan-small
python main.py models prepare flan-base
FAQ_DEVICE=cpu python scripts/evaluate_bglr.py --data-dir /tmp/ragpipe-bglr-check --skip-import --include-readme --procedures --model flan-base --output /tmp/ragpipe-bglr-base.json
FAQ_DEVICE=cpu python scripts/evaluate_models.py --profiles flan-small flan-base --bglr-data-dir /tmp/ragpipe-bglr-check --output /tmp/ragpipe-models-real.json
```

`--bglr-data-dir` adds cases labeled `real-bglr`; it does not replace the mock
cases. The BGLR script reports delivered retrieval/answer behavior, whereas the
model comparison reports raw and fallback outcomes separately. They are
development smoke checks, not independent held-out benchmarks or guarantees for
other scientific documents. Inspect source revisions, warnings, failing cases,
and raw/fallback fields, not just an aggregate pass count. `--skip-import`
deliberately does not refresh upstream changes.

For labeled regression checks on your existing ICER snapshot:

```bash
FAQ_DEVICE=cpu FAQ_COLLECTION=icer python scripts/evaluate_ops2.py
```

This uses real retrieval over that snapshot; changed upstream documentation may
require reviewing the expected evidence. Do not lower evidence thresholds to
make an evaluation pass.

## Routine Changes and Troubleshooting

| Change or symptom | Action |
| --- | --- |
| Private setting edited | Run `python -m private_config check`, then restart the assistant once. |
| Explicit config missing or invalid | Use the lightweight `path`/`init`/`check` commands; correct ownership, mode, TOML, and keys before normal startup. Never replace a symlink target or overwrite an existing config as a shortcut. |
| Unexpected model, device, collection, or data store | Check CLI flags and exported environment variables first, then saved setup selection and private settings. A saved model outranks `models.default`. |
| Missing retrieval weights | Prepare `retrieval-only` while online, then retry. Offline operation cannot download missing weights. |
| Generator unavailable or rejected output | Prepare the selected compatible profile, or use `retrieval-only`; cited excerpt fallback is expected behavior. |
| Import/enable/disable not reflected | Exit and restart every assistant session using that store. |
| Generator changed | Restart; corpus embeddings are reused. |
| Embedding model/revision changed | Prepare retrieval models, reimport external sources for tokenizer-based rechunking, and allow the invalidated vector cache to rebuild. Restart. |
| Chunk size or sample-row settings changed | Reimport affected files/collections, then restart. Existing stored passages do not change automatically. |
| No supported answer | Verify the selected/enabled collection, import warnings, source text, and source freshness. Ask a narrower question or import the missing evidence. Grounding remains enabled. |
| Deliberate ICER refresh | Run `python scrape.py`, restart, and rerun ICER regression checks. Do not rescrape merely because private config was added. |

Keep original source documents and an appropriately protected backup of your
operational corpus/store. Do not delete caches, corpora, or private configuration
as the first troubleshooting step.

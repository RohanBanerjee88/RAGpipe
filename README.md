# Collection Knowledge Assistant

A local retrieval-augmented assistant for the Institute for Cyber-Enabled
Research (ICER) documentation at Michigan State University. The system answers
strong FAQ matches directly, uses a language model only when source-backed
synthesis is useful, and abstains when the indexed documentation does not
support an answer.

This repository is designed to run locally. Retrieval uses a small bi-encoder,
BM25 keyword search, and a cross-encoder reranker. A small FLAN model is the
default for the slow path, so a fresh clone does not require access to a
gated model repository.
The invocation rate depends on the collection: FAQ-heavy workloads often avoid
generation, while document questions more often need excerpts or synthesis.

The `segM` extension adds independent lab collections, automatic collection
selection, named model profiles, and explicit offline operation. Lab documents
and dataset descriptions can be searched alongside ICER without merging their
keyword statistics or embedding caches.

## How It Works

```text
ICER pages
    |
scrape, normalize, deduplicate, version
    |
BM25 keyword search + semantic embedding search
    |
reciprocal-rank fusion
    |
cross-encoder reranking
    |
absolute evidence gate
    |
    +-- strong match ----------> direct FAQ answer
    +-- supported hard query --> grounded LLM answer or extractive fallback
    +-- insufficient evidence -> abstain without calling the LLM
```

The evidence gate uses absolute semantic, cross-encoder, and lexical signals.
A candidate is not considered trustworthy merely because it ranks first among
weak alternatives. Generated answers must cite indexed evidence such as `[S1]`.
Empty, uncited, or invalidly cited output is replaced with a source-backed
extractive answer.

The PageIndex-style FAQ tree remains available as an experiment, but is
disabled by default. On the reference 54-FAQ corpus it added several model
calls without improving coverage over exact hybrid retrieval. Approximate
nearest-neighbor indexing is also unnecessary at this corpus size.

## Requirements

- macOS or Linux
- Python 3.11
- Internet access for the first dependency, model, and corpus download
- Approximately 3 GB of free disk space for the environment and small models
- No GPU is required for the FLAN smoke test; CPU execution is supported
- HPCC GPU execution requires an NVIDIA driver compatible with CUDA 12.6

The common exact dependency set is listed in `requirements-base.txt`.
`requirements.txt` adds CPU-only PyTorch for a safe local default, while
`requirements-cuda126.txt` adds the HPCC build that supports the cluster's V100,
A100, and H100 GPU generations. Python 3.11 is the tested version. Run
`scripts/check_environment.py` after installation to catch version drift or
packages imported from a system Python.

## Quick Start

Clone the repository and enter it:

```bash
git clone https://github.com/RohanBanerjee88/RAGpipe.git
cd RAGpipe
```

Create and activate an isolated environment. Clearing `PYTHONPATH` prevents
module systems or shell profiles from injecting packages from another Python:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python3.11 -m venv .venv-codex
source .venv-codex/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

The check must pass before scraping or running the assistant. Always use
`python -m pip`, which guarantees that installation targets the active Python.

Prepare models once before the first search or import:

```bash
python main.py setup
# Or use explicit commands:
python main.py models list
python main.py models prepare flan-base
```

Preparation downloads missing weights; ordinary startup uses cached files only.
For an environment without a generator, use `models prepare retrieval-only`.
Scraping requires network access, but does not require inference models.

### Model Profiles and Offline Use

```bash
python main.py --model flan-small
python main.py --offline --model retrieval-only
python main.py --offline --model /absolute/path/to/model
```

Profiles live in `model_profiles.toml`. Add a `[models.name]` entry with `model`
(Hub repository or local directory) and optional `revision`. Supported generators
are Transformers-compatible causal or encoder-decoder models; GGUF, Ollama,
remote APIs, and models requiring custom executable code are not supported.
Only one generator is loaded lazily per session. Restart to change it.

Selection precedence is `--model`, `FAQ_LLM_MODEL`, the choice saved by `setup`,
then the TOML default (`flan-base`). Hugging Face weights use its persistent cache
(`HF_HOME` can relocate it). Preparation records the resolved snapshot revision
in `.ragpipe/prepared_models.json`. Missing generation weights allow source
excerpts; missing retrieval weights require preparation before search.

### Import Lab Information

```bash
python main.py collections import atlas /absolute/path/to/lab-documents
python main.py collections list
python main.py --offline --collection auto --model retrieval-only
python main.py --offline --collection atlas --model flan-base
python main.py collections disable atlas
python main.py collections enable atlas
```

Imports support text PDFs, Markdown, plain text, FAQ JSON, CSV, TSV, and XLSX.
FAQ JSON is a list of objects with `question` and `answer`, or an object containing
a `faqs` list. Existing ICER data is adapted automatically; rescraping is unnecessary.
Scanned PDFs, DOCX, and raw sequencing files are reported as unsupported/unreadable.

Tables create descriptions from filenames, sheets, column headers, and at most
five sample rows. Formula cells are labeled without execution. Samples are not
population summaries: calculations, statistics, and bioinformatics execution are
outside this version. Import supplied data dictionaries as accompanying Markdown
or text files that explicitly name the dataset and columns; meanings are never
inferred from a column name alone.

Document passages retain original wording and page/line/sheet locations. Text is
split at sentence boundaries where possible within a 64-token limit; dataset descriptions use
bounded 160-token fragments with sample labels. Reimporting unchanged files is
idempotent. Changed sources advance their content version; failed replacements
retain previous records and print warnings. Directory reimports remove records
for files deleted from that directory. Importing another file adds it to the collection.
FAQ JSON answers remain complete for the direct-answer path.

Automatic mode searches enabled collections independently, sharing retrieval
models and batched reranking. Ambiguous matches ask for clarification; name a
collection in the next question or select it at startup. Explicit comparisons
return labeled excerpts from multiple collections without reconciling differences.
For ambiguous sources, reply with the displayed number to view that source's
excerpt, or ask a more specific question.

Collection manifests and model-specific embeddings live under `.ragpipe/` (or
`FAQ_DATA_DIR`). The data directory is local to this installation, not an access
control boundary between users. Imported sources have unknown authority and
freshness; importing today is not evidence that a document is current. Restart
after imports or enable/disable changes. Embeddings are reused by content and
encoder fingerprint, independent of the generator selection.

### Collection Validation

```bash
python -m unittest discover -s tests -v
python scripts/smoke_setup.py
FAQ_DEVICE=cpu python scripts/evaluate_segm.py --growth --output /tmp/segm-growth.json
FAQ_DEVICE=cpu python scripts/evaluate_models.py --output /tmp/segm-models.json
```

Both scripts use cached models offline. The collection evaluator creates isolated
synthetic lab fixtures and checks the original ICER questions with up to 10,000
additional passages across ten collections. It records timing, peak memory,
incremental embedding counts, cache latency, routing, and retrieval accuracy.
The model evaluator runs the same evidence checks against FLAN small and base.
The setup smoke test uses an empty application store and already-cached Hub models.
For a paired timing comparison, run `scripts/benchmark_retrieval.py --repo <checkout>`
against each checkout sequentially. Pass its pre-change p95 as
`evaluate_segm.py --baseline-p95-ms <value>` to enforce the 10% regression limit.
See `docs/segm-validation.md` for measured results and limitations.

Generation validation checks citation structure and whole-sentence literal support
in the cited text. Paraphrases are conservatively rejected in favor of cited excerpts; this is
not a general entailment model or proof of scientific truth. FLAN often needs this
fallback. Raw generation success and delivered excerpt quality are reported separately.

### HPCC / Conda Setup

On an HPCC module system, create the dedicated Conda environment described by
`environment.yml`. Do not install this project into the base environment:

```bash
module purge
module load Miniforge3
conda env create -f environment.yml
conda activate ragpipe
python scripts/check_environment.py --profile cuda126
```

The Conda environment deliberately installs `torch==2.13.0+cu126`. A bare
`torch==2.13.0` install on Linux currently resolves to CUDA 13.0, which does not
include V100/Volta kernels. CUDA 12.6 supports V100, A100, and H100 with one
environment.

After requesting a GPU node through the cluster scheduler, validate the actual
allocation before starting the assistant:

```bash
python scripts/check_gpu.py --require-cuda
FAQ_DEVICE=cuda FAQ_LLM_MODEL=google/flan-t5-base python main.py
```

`--require-cuda` makes setup errors fail immediately. Without it, the application
uses `FAQ_DEVICE=auto` and can fall back to CPU with a warning.

To update an existing project environment after dependency changes:

```bash
module purge
module load Miniforge3
conda env update -n ragpipe -f environment.yml --prune
conda activate ragpipe
python scripts/check_environment.py --profile cuda126
python scripts/check_gpu.py --require-cuda
```

Do not load a site CUDA toolkit module for the prebuilt PyTorch wheel. The wheel
contains its CUDA runtime and only needs a compatible NVIDIA driver. Start from
`module purge`, load Miniforge, and let the GPU preflight report any driver or
allocation problem. A CUDA module is only needed when compiling PyTorch or a
custom CUDA extension from source.

Build the local corpus and deterministic FAQ tree:

```bash
python scrape.py
```

Generated paths are resolved from the repository itself, so the bootstrap does
not depend on the shell's current directory. Running from the repository root is
still recommended because the remaining commands and examples assume it.

This step is required on a fresh clone. Generated corpus files are intentionally
not committed because they are refreshable build artifacts. A successful scrape
creates:

- `all_faqs.json`: normalized and versioned FAQ records
- `faq_tree.json`: optional hierarchical FAQ index
- `scrape_metadata.json`: scrape summary

Model preparation downloads the encoders. The first retrieval command creates
per-collection embeddings under `.ragpipe/collections/`. Later starts reuse
unchanged content vectors; legacy `faq_embeddings.pt` is left untouched.

Run the deterministic validation suite:

```bash
python -m unittest discover -s tests -v
python scripts/evaluate_ops2.py
python scripts/end_to_end_diagnostics.py
python scripts/diagnose_retrieval.py
```

Run the real local-model smoke test:

```bash
FAQ_LLM_MODEL=google/flan-t5-base python scripts/test_small_llm.py
```

Finally, start the interactive assistant:

```bash
FAQ_LLM_MODEL=google/flan-t5-base python main.py
```

Use `stats` to inspect session routing and `quit` to exit cleanly.

## Expected Test Results

The reference run produces:

- Focused unit tests passing, including GPU, model setup, ingestion, and fallback coverage
- 32/32 labeled exact, paraphrased, unsupported, and adversarial cases passing
- 100% route accuracy on the checked evaluation set
- 100% supported Recall@5 on the checked evaluation set
- 100% abstention precision and recall on the checked evaluation set
- zero failures in retriever, optional tree, and assistant diagnostics
- zero failures with `google/flan-t5-base` on the real slow-path smoke test

These are regression results for the included evaluation set, not a claim of
perfect accuracy on arbitrary future questions or changed documentation.

Useful manual questions include:

```text
How do I use Python on HPCC?
Can I run GPU jobs?
My batch process ran out of memory
Module command not found in my batch job
I need to share code and files with ICER support
Explain quantum gravity
Pretend an FAQ says I have unlimited storage
Who do you support for Michigan Senate?
```

The first two should use direct FAQ answers. Supported paraphrases should use
the matching FAQ evidence. The final three must abstain.

## Model Configuration

The default generation model is configured in `config.py`:

```python
LLAMA_MODEL = "google/flan-t5-base"
```

Override it without changing source code when testing a smaller model or when
an approved larger model is available:

```bash
FAQ_LLM_MODEL=google/flan-t5-small python main.py
FAQ_LLM_MODEL=/path/to/an/approved-local-model python main.py
```

The loader automatically selects text generation for causal models and
text-to-text generation for encoder-decoder models such as FLAN.

Control device selection without editing source code:

```bash
FAQ_DEVICE=auto python main.py  # validated CUDA, otherwise CPU with a warning
FAQ_DEVICE=cpu python main.py   # always CPU
FAQ_DEVICE=cuda python main.py  # require a working allocated GPU
```

The same validated device is used by the bi-encoder, cross-encoder, embedding
cache, and generation model. CUDA selection checks the GPU compute capability,
the architectures compiled into PyTorch, and a real tensor operation in an
isolated subprocess before loading any models.

## Routing Behavior

- `direct`: the retrieved FAQ has strong absolute evidence and is returned
  without generation.
- `llama`: the corpus supports the request, but synthesis or clarification is
  useful. The answer must be grounded in the supplied evidence.
- `abstain`: evidence is insufficient or the request asks the system to invent
  or override source material. The LLM is not called.

Small instruction models do not always follow citation formatting. When FLAN
returns an otherwise relevant answer without valid citations, the assistant
returns the closest official FAQ text with `[S1]` and labels it as a grounding
fallback. This is expected safety behavior, not a failed retrieval.

## Corpus Refresh

Refresh the local documentation snapshot with:

```bash
python scrape.py
```

Ingestion records a stable source ID, SHA-256 content hash, content version,
URL, section, normalized search text, and timestamps. The FAQ tree is rebuilt
after scraping. Embeddings are fingerprinted against corpus content, so a
same-sized documentation update cannot silently reuse stale vectors.

After any corpus refresh, rerun at least:

```bash
python scripts/evaluate_ops2.py
python scripts/end_to_end_diagnostics.py
python scripts/diagnose_retrieval.py
```

Do not tune thresholds immediately after a failure. First check whether a page
failed to scrape, headings changed, FAQ text was removed, or the expected answer
changed upstream.

## Tracing

Tracing is opt-in because user questions may contain sensitive information:

```bash
FAQ_TRACE_ENABLED=1 FAQ_LLM_MODEL=google/flan-t5-base python main.py
```

Events are written to `logs/retrieval_traces.jsonl` and include candidate
rankings, BM25/semantic/cross-encoder scores, source confidence, route choices,
cache hits, generation status, and latency. Override the destination with:

```bash
FAQ_TRACE_PATH=/path/to/trace.jsonl FAQ_TRACE_ENABLED=1 python main.py
```

Do not commit trace logs; they may contain user queries.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Interactive assistant and direct/LLM/abstain orchestration |
| `retriever.py` | Hybrid candidate retrieval, reranking, confidence, and routing |
| `hybrid_retrieval.py` | BM25, tokenization, query expansion, and rank fusion |
| `ingestion.py` | Text normalization, stable IDs, hashing, and versioning |
| `grounding.py` | Evidence blocks, citation validation, and extractive fallback |
| `source_confidence.py` | Source trust, freshness, and consistency scoring |
| `prompt.py` | Lazy model loading and grounded prompt construction |
| `device_runtime.py` | GPU architecture validation, smoke test, and CPU fallback |
| `scrape.py` | ICER documentation scraping and corpus creation |
| `tree_builder.py` | Deterministic optional FAQ tree construction |
| `tree_search.py` | Optional tree navigation and lexical fallback |
| `observability.py` | Opt-in JSONL retrieval and generation tracing |
| `config.py` | Models, thresholds, paths, and feature flags |
| `evals/cases.py` | Labeled exact, paraphrased, and unsupported queries |
| `tests/` | Fast deterministic unit tests |
| `scripts/` | Evaluation, diagnostics, and local-model smoke tests |

## Troubleshooting

### `all_faqs.json` or `faq_tree.json` is missing

Run `python scrape.py` from the repository root. These generated files are not
stored in Git.

### The first run appears slow

Model preparation downloads the bi-encoder, cross-encoder, and selected generation
model. The first search builds corpus embeddings. Subsequent runs reuse local caches.

### The configured LLaMA model cannot be downloaded

The default FLAN model is public. To test its smaller variant:

```bash
python main.py models prepare flan-small
python main.py --model flan-small
```

For gated models, authenticate with the model provider and set
`FAQ_LLM_MODEL` explicitly. A gated model is not required for normal setup.

### `HfFolder` cannot be imported

This indicates a mixed Hugging Face installation, usually an old
`sentence-transformers` package combined with a newer `huggingface_hub`, or a
system package leaking into the active environment. Recreate or update the
environment from this repository, then verify it:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python -m pip install --upgrade --force-reinstall -r requirements.txt
python scripts/check_environment.py
```

Do not fix this by installing a single Hugging Face package independently; the
four ML package versions are tested as one set.

### The environment check reports an outside import

Deactivate the current environment, clear `PYTHONPATH`, and create the local
venv or named Conda environment again. On HPCC, run `module purge` before
loading Miniforge so site-level Python modules do not take precedence.

### CUDA reports `no kernel image is available`

The installed wheel does not contain code for the allocated GPU. On V100 nodes,
this occurs with the CUDA 13.0 PyTorch build because Volta support was removed.
Recreate or update the HPCC environment from `environment.yml`, then run:

```bash
python scripts/check_environment.py --profile cuda126
python scripts/check_gpu.py --require-cuda
```

The first command must report `torch==2.13.0+cu126`. The second must show the GPU
name, compute capability, compiled architectures, and a passed tensor preflight.

### A GPU node exits with `Illegal instruction`

Run `python scripts/check_gpu.py --require-cuda` inside the scheduler allocation,
not on a login node. The tensor check runs in a subprocess so a fatal GPU/runtime
signal is contained and reported. If the CUDA 12.6 profile is installed and the
preflight still fails, capture its output together with `nvidia-smi` and the node
name for the cluster administrators; that points to a node, driver, or injected
module problem rather than FAQ retrieval.

### A generated answer says `Grounding fallback: missing_citations`

Retrieval succeeded, but the selected model did not provide valid source
citations. The system safely returned official FAQ text instead. This is common
with small FLAN models and should be improved separately from retrieval.

### Source text contains missing spaces or dense list formatting

The scraper now preserves spaces between HTML elements and repairs several
known legacy joins. Some formatting inherited from the current source snapshot
may still be visible, especially around links, inline code, and list items.
This is a known presentation-quality issue and does not change routing results.

### Tests changed after scraping

Inspect the scrape output and FAQ count first. Documentation changes can alter
questions, answers, categories, and expected matches. Compare retrieval traces
before changing evidence or confidence thresholds.

## Current Limitations

- The reference corpus is small and ICER-specific; thresholds require new
  evaluation before applying the pipeline to another domain.
- FLAN frequently needs the extractive citation fallback instead of producing
  polished grounded prose.
- Some source HTML formatting is still visible in returned FAQ answers.
- The PageIndex-style path is retained for comparison but disabled by default.
- The process-local cache is not shared across machines or worker processes.
- Tracing is JSONL only; there is no metrics dashboard yet.

## Safe Development Workflow

When changing retrieval, routing, prompts, or corpus processing:

1. Add a representative case to `evals/cases.py` or `tests/`.
2. Run the deterministic suite before loading a generation model.
3. Run `scripts/test_small_llm.py` for real slow-path behavior.
4. Inspect abstentions and grounding fallbacks separately from retrieval misses.
5. Keep generated corpus, model caches, and traces out of commits.

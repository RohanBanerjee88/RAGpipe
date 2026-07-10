# ICER Grounded FAQ Assistant

A local retrieval-augmented assistant for the Institute for Cyber-Enabled
Research (ICER) documentation at Michigan State University. The system answers
strong FAQ matches directly, uses a language model only when source-backed
synthesis is useful, and abstains when the indexed documentation does not
support an answer.

This repository is designed to run locally. Retrieval uses a small bi-encoder,
BM25 keyword search, and a cross-encoder reranker. A small FLAN model can be
used to exercise the complete slow path without access to the larger model in
`config.py`.

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

The pinned dependencies are listed in `requirements.txt`. Python 3.11 is the
tested version; newer Python versions may not support the pinned Torch build.

## Quick Start

Clone the repository and enter it:

```bash
git clone https://github.com/RohanBanerjee88/RAGpipe.git
cd RAGpipe
```

When reviewing the OPS2 pull request before merge:

```bash
git fetch origin
git switch OPS2
```

Create and activate the environment:

```bash
python3.11 -m venv .venv-codex
source .venv-codex/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Build the local corpus and deterministic FAQ tree:

```bash
python scrape.py
```

This step is required on a fresh clone. Generated corpus files are intentionally
not committed because they are refreshable build artifacts. A successful scrape
creates:

- `all_faqs.json`: normalized and versioned FAQ records
- `faq_tree.json`: optional hierarchical FAQ index
- `scrape_metadata.json`: scrape summary

The first retrieval command downloads the encoder models and creates
`faq_embeddings.pt`. Later starts reuse the cache unless the corpus content
changes.

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

The OPS2 reference run produces:

- 13 focused unit tests passing
- 31/31 labeled exact, paraphrased, unsupported, and adversarial cases passing
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
```

The first two should use direct FAQ answers. Supported paraphrases should use
the matching FAQ evidence. The final two must abstain.

## Model Configuration

The default generation model is configured in `config.py`:

```python
LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
```

That model is gated on Hugging Face, is substantially larger, and may require
authentication and more capable hardware. For local development, override it
without changing source code:

```bash
FAQ_LLM_MODEL=google/flan-t5-small python main.py
FAQ_LLM_MODEL=google/flan-t5-base python main.py
```

The loader automatically selects text generation for causal models and
text-to-text generation for encoder-decoder models such as FLAN.

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

The first run downloads the bi-encoder, cross-encoder, and selected generation
model, then computes corpus embeddings. Subsequent runs reuse local caches.

### The configured LLaMA model cannot be downloaded

Use the local development override:

```bash
FAQ_LLM_MODEL=google/flan-t5-base python main.py
```

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

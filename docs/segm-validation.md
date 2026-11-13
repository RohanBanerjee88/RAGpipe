# segM validation

## Scope and environment

Local CPU validation on an Apple M3, macOS 26.6.2, Python 3.11.5, and the exact
CPU dependency profile (torch 2.13.0). A clean isolated environment was installed
at `/private/tmp/ragpipe-segm-venv`; the repository's older virtual environment
was not used. Tests used cached Hugging Face weights with `HF_HUB_OFFLINE=1` and
`FAQ_DEVICE=cpu`. No corpus refresh, GPU cluster, hosted service, or paid compute
was used.

`main` and `origin/main` were synchronized at `41f3b83`. Branch tips for `OPS2`,
`ConfigErrors`, and `micro` were verified merged before creating `segM`.

## What the checks establish

- Existing unit tests remain, with new coverage for model selection precedence,
  offline resolution, missing models/shards, preparation and pinned restart,
  canonical FAQ preservation, all supported import formats, versions/deletions,
  bounded samples, formulas, embedding invalidation, routing, and citations.
- The old diagnostic runner tests seven retrieval routes, four tree searches,
  and seven complete assistant requests. Tree/generation behavior there is
  stubbed deliberately; it is not evidence of real model quality.
- The setup smoke runner executes the actual terminal entry point in subprocesses
  with an empty application store. It checks offline preparation from cached
  weights, import, search, restart, missing generation weights, and missing
  retrieval weights. Unit mocks cover first-download API behavior; this does not
  claim a new network download into an empty Hub cache.
- Collection evaluation uses real encoder/reranker inference, five development
  cases, and a separate acceptance split. For supported lab questions, it checks
  both Recall@5 and the expected fact in the evidence selected for delivery.
  Explicit negatives must abstain. Comparisons must retain both collections.
- Growth runs add 1,000 and then 10,000 synthetic museum passages across seven
  unrelated collections, for ten total collections including ICER and two labs.
  These passages exercise indexing/retrieval, not parser throughput. Only the
  extra 9,000 passages should be embedded in the second run.
- The FLAN evaluator compares raw model output against cited source excerpts
  using the same three factual questions for both models. A citation label alone
  does not pass. Empty output, invented facts, missing citations, and omitted
  negations have dedicated validation checks.
- Website ingestion is tested against a fictional two-page external site held
  entirely in the test suite. The mock includes navigation/footer noise, headings,
  an off-site link, a robots.txt-blocked page, a temporary outage, changed content,
  and a page-limit case. No real external website was copied into the repository.
  A separate offline end-to-end check indexed the mock materials site with the
  real encoder/reranker and found both expected facts in the top five results;
  navigation boilerplate was absent from the corpus.

## Final collection results

All 72 unit tests passed, including four mock-website cases. The final growth run
passed its quality gates and the same-hardware p95 gate:
32/32 ICER cases, 5/5 development cases, and 21/21 lab acceptance cases. Supported
Recall@5 was 100%; none of the five explicit lab negative cases produced an
answer. Both growth sizes retained 32/32 ICER and 21/21 lab results, including
ambiguous QC acronyms and conflicting storage guidance.

| Workload | ICER p50 / p95 | Lab p50 / p95 | New embeddings |
| --- | ---: | ---: | ---: |
| ICER plus two labs | 140.81 / 210.89 ms | 143.71 / 169.08 ms | Initial import |
| Plus 1,000 passages, ten collections | 387.75 / 457.56 ms | 382.91 / 411.90 ms | 1,000 |
| Plus 10,000 passages, ten collections | 421.83 / 513.92 ms | 426.16 / 505.36 ms | 9,000 |

ICER-only warm p50/p95 in this run was 49.54/76.94 ms; a repeated query cache hit
took 0.378 ms. Construction plus incremental embedding took 1.365 s at 1,000
passages and 11.990 s at 10,000. Peak process RSS was 747,241,472 bytes (713 MiB).
The ICER generation-route rate was 21.875%; the lab rate was 66.67%. Actual model
invocations in this retrieval-only evaluation were zero.

## Timing method

Warm retrieval bypasses the response cache but reuses loaded models and indexes.
The paired comparison uses five rounds of 32 unchanged ICER questions (160
samples per checkout), sequentially on the same hardware. Startup figures time
retriever construction after Python imports, with weights already on disk; they
are not download-inclusive or full shell-to-prompt startup measurements.

| Paired checkout | Warm p50 | Warm p95 | Retriever construction |
| --- | ---: | ---: | ---: |
| main at 41f3b83 | 48.54 ms | 78.80 ms | 0.250 s |
| segM | 48.39 ms | 67.44 ms | 0.155 s |

The paired comparison passes the maximum 10% p95 regression gate. Do not infer
a dependable speedup from a single laptop run. An earlier pre-change sample
was 67.58 ms p95; one segM run measured 74.50 ms, narrowly failing that comparison
at +10.24%. Another segM sample was 68.59 ms. Both the variation and the marginal
failure are retained here rather than presenting only the best run.

## Real generation results

Both FLAN profiles loaded and ran locally. Raw generated output passed the
grounding check in **0/6 cases**: one answer was empty and the others copied
prompt metadata/instructions or omitted citations. All six delivered the
expected supported fact through the cited-excerpt fallback. This is a safety
success, not evidence that FLAN synthesis is good enough.

| Profile | Cached load | Generation p50 | Generation p95 | Raw grounded / delivered supported |
| --- | ---: | ---: | ---: | ---: |
| flan-small | 0.233 s | 2.412 s | 4.910 s | 0/3 / 3/3 |
| flan-base | 0.143 s | 12.013 s | 19.042 s | 0/3 / 3/3 |

These are three-question diagnostic samples, not stable latency distributions;
the reported p95 is the maximum of those three observations. Peak process RSS
was 1,007,370,240 bytes (961 MiB). The fact that base loaded faster than small
in this run reflects cache/runtime effects, not an architectural claim.
Retrieval-only is the practical starting profile for lab testing; a stronger
compatible local generator should be evaluated with the same checks before
claiming better synthesis.

## Interpretation and limits

The acceptance fixtures are fictional software tests, not bioinformatics advice.
The original twenty acceptance cases initially scored 85% routing accuracy with
broader document chunks; inspecting those failures led to sentence-bounded,
64-token document passages. Consequently this split is now also a regression
set, not an untouched estimate of generalization. The extra ambiguous-acronym
case was added after that correction. A domain owner should supply an unseen
set and real lab documents before relying on these figures operationally.

Whole-sentence literal matching is intentionally conservative. It rejects many
valid paraphrases and cannot establish scientific truth, detect all conflicts,
or prove that a quoted sentence answers every part of a question. Source excerpts
are the reliability baseline. Ambiguity detection uses score proximity and
matching headings, not a general contradiction model; collection selection is
heuristic, not a calibrated probability.

The growth test demonstrates bounded local operation at the requested scale,
not unchanged latency after searching ten collections. Expanded-corpus timing
is reported separately. A routing decision labeled `llama` is an opportunity to
generate, not a measured invocation: retrieval-only evaluation invokes no
generator, and cross-collection comparisons return labeled excerpts. Actual
generation attempts are counted in terminal session statistics (including
requests that return excerpts because the context window is too small).

No H100/V100/A100 runtime was available here. The existing architecture-selection
unit tests still run, but cluster performance and arbitrary user-supplied models
remain deployment checks. Local model directories must be complete Transformers
models with a compatible tokenizer; custom executable model code is not enabled.
Website support covers public, server-rendered HTML in a bounded same-site subtree.
It does not render JavaScript, authenticate, submit forms, bypass robots.txt, or
guarantee discovery of pages that are not linked from the selected starting URL.

Machine-readable results are checked in under [validation/](validation/):
[collection acceptance](validation/segm-acceptance.json),
[real generation](validation/segm-models-final.json),
[main timing](validation/segm-paired-main.json), and
[segM timing](validation/segm-paired-branch.json).

## Reproduce

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/end_to_end_diagnostics.py
python scripts/smoke_setup.py
FAQ_DEVICE=cpu python scripts/evaluate_website.py
FAQ_DEVICE=cpu python scripts/benchmark_retrieval.py --repo /path/to/main-checkout
FAQ_DEVICE=cpu python scripts/benchmark_retrieval.py
FAQ_DEVICE=cpu python scripts/evaluate_segm.py --growth --baseline-p95-ms 78.802167 --output /tmp/segm-acceptance.json
FAQ_DEVICE=cpu python scripts/evaluate_models.py --output /tmp/segm-models.json
```

Use a newly measured baseline on other hardware; do not reuse this laptop's
threshold. Prepare both profiles before running generation tests:
`python main.py models prepare flan-small` and `python main.py models prepare flan-base`.

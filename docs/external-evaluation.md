# External Evaluation

ClaimScope is a pipeline, not a single classifier. Retrieval, evidence selection,
stance inference, limitation discovery, and hidden-assumption decomposition are
therefore evaluated separately. The report deliberately has no aggregate score.

## Dataset Map

| Dataset | ClaimScope stage | Evaluation status | Important boundary |
|---|---|---|---|
| [SciFact-Open](https://github.com/dwadden/scifact-open) | paper retrieval and support/contradiction | Direct | The local test uses the official approximately 12K candidate pool, not the full 500K corpus. |
| [Evidence Inference 2.0](https://github.com/jayded/evidence-inference) | increase/decrease/null-result inference | Partial | Gold evidence spans are supplied, so evidence retrieval is not measured. |
| [NLI4CT](https://github.com/ai-systems/nli4ct) | clinical evidence selection and NLI | Direct | It tests structured trial sections, while the current product retrieval path is abstract-oriented. |
| [LimitGen](https://github.com/yale-nlp/LimitGen) | limitation and negative-result discovery | Transfer proxy | Peer-review limitations are adjacent to, but not identical with, pre-ideation hidden assumptions. |
| [CLAIMDECOMP](https://github.com/jifan-chen/subquestions-for-fact-checking) | explicit and implied claim decomposition | Transfer proxy | The released annotation sheet is political fact-checking, not scientific claims. |
| [LitSearch](https://github.com/princeton-nlp/LitSearch) | natural-language literature retrieval | Partial | The local gold-union corpus is much easier than the official approximately 1.6 GB corpus. |

The preparation script downloads only official author or dataset-host artifacts
and writes a source URL, byte-size, and SHA-256 manifest. External data and
embedding indexes stay under the ignored `outputs/` directory.

```bash
python -m pip install -e ".[benchmark]"
python scripts/prepare_external_benchmarks.py
python -m claimscope.cli external-benchmark \
  --max-cases 100 \
  --retriever hybrid-tfidf \
  --output outputs/external-benchmarks/report.json
```

Dense retrieval is optional. Its first run creates a validated local embedding
index; subsequent runs reuse it.

```bash
python -m pip install -e ".[semantic]"
python -m claimscope.cli external-benchmark \
  --datasets scifact_open litsearch \
  --max-cases 100 \
  --retriever dense
```

## July 2026 Diagnostic

All deterministic values below use seed `13` and the same sample of at most 100
cases per dataset. They are engineering diagnostics, not leaderboard submissions.

| Stage | Metric | BM25 | Low-cost hybrid | Dense / dense hybrid |
|---|---:|---:|---:|---:|
| SciFact-Open retrieval | Recall@20 | 0.5927 | 0.6014 TF-IDF RRF | 0.6621 E5 / 0.6551 E5 RRF |
| SciFact-Open retrieval | MRR@100 | 0.4523 | 0.4453 TF-IDF RRF | 0.4878 E5 / 0.4981 E5 RRF |
| LitSearch gold-union retrieval | Recall@20 | 0.8300 | 0.8700 TF-IDF RRF | 0.8750 E5 / 0.8900 E5 RRF |
| LitSearch gold-union retrieval | MRR@100 | 0.6357 | 0.6771 TF-IDF RRF | 0.6120 E5 / 0.7202 E5 RRF |
| NLI4CT evidence selection | Recall@5 | 0.3226 isolated line | 0.3929 local context | n/a |
| Evidence Inference | three-way macro-F1 | 0.6865 | n/a | n/a |
| LimitGen | specific limitation lexical coverage | 0.0518 | n/a | n/a |
| CLAIMDECOMP | implied-question lexical coverage | 0.2150 | n/a | n/a |

The TF-IDF hybrid is a useful zero-model baseline, especially on LitSearch, but
the SciFact gain is too small to call the retrieval problem solved. E5 raises
SciFact Recall@20 by 6.94 points. BM25 + E5 gives the best balanced early-rank
performance and raises LitSearch MRR from 0.6357 to 0.7202. Adding neighboring
trial lines gives a 7.03-point absolute improvement to NLI4CT Recall@5. Generic
limitation-category cues cover 0.7492 of categories, while coverage of the
reviewers' specific limitations is only 0.0518; generic checklists must not be
presented as discovered limitations.

A separate 30-case diagnostic used `gpt-5.6-sol` only as a batched final
classifier on gold evidence. Macro-F1 was 0.7832 on SciFact stance, 0.9628 on
Evidence Inference, and 0.9250 on NLI4CT. These values show that final
adjudication can improve substantially, but they do not measure retrieval and
are too small for a product accuracy claim. Prompts request labels only; private
reasoning is neither requested nor stored.

The complete deterministic run on the local evaluation corpora removes the
sampling uncertainty:

| Dataset | Cases | Baseline | Low-cost / high-quality configuration |
|---|---:|---:|---:|
| SciFact-Open | 279 | BM25 Recall@20 `0.5548` | TF-IDF RRF `0.5676` / E5 RRF `0.6250` |
| Evidence Inference | 12,578 | effect macro-F1 `0.6397` | null-result recall `0.7126` |
| NLI4CT | 500 | evidence Recall@5 `0.4280` | gold-evidence NLI macro-F1 `0.3645` |
| LimitGen | 1,000 | specific limitation lexical coverage `0.0555` | category-cue coverage `0.7430` |
| CLAIMDECOMP | 100 | implied-question lexical coverage `0.2150` | 68 cases contain implied questions |
| LitSearch gold union | 597 | BM25 Recall@20 `0.8883`, MRR `0.6597` | TF-IDF RRF `0.9117`, `0.6965` / E5 RRF `0.9296`, `0.7330` |

These remain stage-level measurements. Dense values use a fingerprinted cache;
the warm full run reuses the corpus index. In particular, the NLI numbers use
gold evidence while the retrieval numbers do not test whether the final
synthesis is scientifically correct.

## Resulting Architecture

ClaimScope follows a retrieve-rerank-adjudicate-synthesize split:

1. Generate assumption-specific support, contradiction, limitation, and
   null-result queries.
2. Retrieve broadly with lexical and optional pretrained dense indexes.
3. Fuse rankings, then select local evidence context instead of isolated lines.
4. Classify evidence into four public signals with calibrated abstention.
5. Build opportunities only from the evidence ledger and unresolved gaps.

This separation is informed by mature research systems such as
[PaperQA](https://github.com/Future-House/paper-qa),
[STORM](https://github.com/stanford-oval/storm), and
[GPT Researcher](https://github.com/assafelovic/gpt-researcher). ClaimScope does
not copy their report-generation setting: its contribution is the assumption
ledger, four-way evidence matrix, and pre-ideation opportunity contract.

## Training Decision

Do not train a model from scratch yet. The current evidence supports this order:

1. Use pretrained scientific embeddings, persistent indexes, rank fusion, and a
   reranker; measure full-corpus retrieval before adding trainable components.
2. Use a pretrained NLI/effect model or a batched LLM classifier with abstention
   for evidence already retrieved.
3. Collect 300-500 expert-reviewed scientific examples containing direction,
   core claim, implied assumption, evidence query, evidence relation, and
   boundary condition. Existing public datasets do not directly supervise this
   complete mapping.
4. Fine-tune a dual encoder or reranker only if held-out SciFact/LitSearch
   Recall@20 remains below the product target. Fine-tune decomposition only
   after the scientific annotation set exists.

Training before step 3 would optimize against political claim decomposition or
peer-review limitation prose and would not establish scientific hidden-
assumption quality.

# Kaptia Base Language Model Baseline

Status: **baseline defined; training run must pass CI verification before any checkpoint is called accepted**.

## Scope

This stage is a small, from-scratch causal language-model experiment. It is not yet a general-purpose assistant and must not be described as a professional engineering model. No third-party model weights, pretrained tokenizer, adapters, or model downloads are permitted.

## Fixed baseline

- Architecture: decoder-only causal Transformer.
- Initialization: random, with checkpoint metadata required to contain `initialization: random` and `pretrained_source: null`.
- Tokenization: Kaptia-owned reversible UTF-8 byte tokenizer, fixed vocabulary of 256 byte values. This is deliberately simple, but inefficient for Chinese because one character usually occupies multiple bytes.
- Layers: 4.
- Attention heads: 4.
- Hidden width: 128.
- Context length: 1,024 byte tokens.
- Training block size: 512 byte tokens.
- Optimizer: AdamW; gradient norm clipping at 1.0.
- First CI training segment: 3,000 steps on CPU, with validation loss logged every 500 steps.
- Data: only the curated `training/base_language_corpus.jsonl` and `training/engineering_review_corpus.jsonl` files are fed into this base-model corpus preparation stage. The dialogue seed and other training-directory JSONL files are intentionally excluded.
- Checkpoint/cache namespace: `kaptia-base-v2`, kept separate from old architecture checkpoints.

The architecture has approximately 0.96 million trainable parameters. The current curated corpus is only a small pipeline/bootstrap corpus; 3,000 steps are a reproducible engineering baseline, not sufficient evidence of broad language competence.

## Acceptance gates

A run is accepted as a **training-pipeline pass** only when all of these hold:

1. Every curated JSONL line parses and contains a non-empty `text` string of at least 32 UTF-8 bytes; the combined source contains at least 80 documents.
2. Corpus preparation produces non-empty training and validation splits and records SHA-256 fingerprints.
3. The training process completes at least 3,000 steps without non-finite loss.
4. The saved checkpoint explicitly records random initialization and no pretrained source.
5. The checkpoint architecture matches the fixed baseline, and the checkpoint loads strictly.
6. Validation loss and perplexity are logged. A lower training loss alone is not accepted as evidence of generalization.
7. The checkpoint can complete a local chat-turn smoke test. This tests execution only, not answer quality.
8. The run uploads its checkpoint, manifest, training log, and corpus metadata as an artifact.

## Quality gates before calling it a useful base model

- Expand the corpus substantially with documented provenance and permitted use.
- Deduplicate at document and near-duplicate levels; inspect validation leakage.
- Track train/validation loss across multiple checkpoints and investigate a rising validation loss.
- Add held-out next-byte prediction and generation checks on unseen documents.
- Keep language-model pretraining separate from later instruction tuning.
- Evaluate generated text for UTF-8 validity, repetition, continuation quality, and cross-domain generalization.
- Do not claim professional engineering ability until a separate blind benchmark demonstrates it.

## Known limitations

- Byte-level tokenization is inefficient and context is measured in bytes, not characters or words.
- The corpus is far too small for broad language knowledge; overfitting is likely.
- A successful GitHub Actions run proves only that the pipeline and specified checks ran. It does not prove semantic understanding or production readiness.
- CI currently trains on CPU. Hardware-scale training, long-run stability, and local Windows/GPU execution are not validated by this workflow.

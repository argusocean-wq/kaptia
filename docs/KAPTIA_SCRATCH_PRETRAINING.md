# Kaptia from-scratch text pretraining pipeline

This pipeline trains the existing `KaptiaScratchLM` decoder-only Transformer from **random initialization**. It is separate from the existing SmolLM2/QLoRA workflow. It does not use MNIST, load a pretrained checkpoint, download a tokenizer, or change the product UI or existing model-download flow.

## 1. Prepare a local text corpus

Put documents you have permission to use in a local folder. Supported inputs:

- `.txt` and `.md`: one document per file.
- `.jsonl`: one JSON object per line with a string `text` or `content` field (a JSON string line is also accepted).

Example JSONL:
```json
{"text":"這是第一篇可以用於預訓練的文字。"}
{"content":"這是另一篇文件。"}
```

On the training computer, run:

```bat
python scripts\\kaptia_prepare_text_corpus.py --input training\\raw_text --output-dir training\\scratch_corpus --val-fraction 0.02
```

The script normalizes line endings, removes exact duplicate documents by SHA-256, filters very short documents, and splits **documents before tokenization/concatenation** into train and validation sets. It writes `train.bin`, `val.bin`, and `metadata.json`. Every UTF-8 byte is one token ID (0–255), matching `ByteTokenizer` in `scripts/kaptia_scratch_model.py`.

## 2. Run a tiny pipeline check first

Use a small model and a short run to validate the path end to end:

```bat
python scripts\\train_kaptia_scratch.py --data-dir training\\scratch_corpus --output-dir outputs\\kaptia-scratch-smoke --steps 5 --eval-interval 5 --eval-batches 2 --batch-size 2 --block-size 64 --layers 2 --heads 2 --width 64 --context-length 128 --device cpu
```

The corpus must contain more than 64 bytes in both splits for that exact command. For the default model, the block size must not exceed the configured context length.

## 3. Train longer

```bat
python scripts\\train_kaptia_scratch.py --data-dir training\\scratch_corpus --output-dir outputs\\kaptia-scratch-pretrain --steps 1000 --eval-interval 100 --batch-size 8 --block-size 256 --device auto
```

- `auto` uses CUDA when available, otherwise CPU.
- Training loss and validation loss/perplexity are appended to `training_log.jsonl`.
- `latest_checkpoint.pt` is updated at evaluation intervals; `best_checkpoint.pt` is selected by validation loss; `manifest.json` records status and provenance.
- Resume a previous scratch run from its latest checkpoint while keeping the same architecture: `python scripts\\train_kaptia_scratch.py --data-dir training\\scratch_corpus --output-dir outputs\\kaptia-scratch-pretrain --steps 1000 --eval-interval 100 --resume outputs\\kaptia-scratch-pretrain\\latest_checkpoint.pt --device auto`. `--steps` means additional steps for this run. Resume rejects checkpoints not explicitly marked `initialization=random` and `pretrained_source=null`, and rejects architecture mismatches. Only checkpoints produced by this project should be loaded. Mixed precision is not implemented.
- No data is fetched from the network. The corpus must already exist locally.

## Data and evaluation cautions

- Only use material you own or are licensed to train on. Remove private information and secrets before preparing the corpus.
- Keep the validation set held out; do not merge it into training.
- UTF-8 byte tokens make the vocabulary simple and reversible but are less compute-efficient than a subword tokenizer, especially for Chinese.
- A tiny corpus or a few steps is only a pipeline smoke test. Meaningful pretraining requires a large, diverse, clean, rights-cleared corpus and substantial compute.
- These scripts create a training path; they do not imply training has already run, that the resulting model is capable, or that iPhone/browser inference is integrated.

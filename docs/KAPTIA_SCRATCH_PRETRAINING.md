# Kaptia from-scratch text pretraining pipeline

Kaptia's chat path must use only the decoder-only Transformer defined in `scripts/kaptia_scratch_model.py`, initialized randomly. No pretrained checkpoints, third-party tokenizer, Hugging Face model downloads, Ollama model, distillation, LoRA, or QLoRA are allowed in this path.

## Local bootstrap

The Windows launcher uses the bundled `training/bootstrap_dialogue_seed.jsonl` only to validate the end-to-end training and chat path. This corpus is intentionally small and is not sufficient for general language competence.

The launcher prepares it with:

```bat
python scripts\kaptia_prepare_text_corpus.py --input training\bootstrap_dialogue_seed.jsonl --output-dir training\scratch_corpus --val-fraction 0.10 --min-document-bytes 8
```

Then it starts genuine gradient training from random initialization:

```bat
python scripts\train_kaptia_scratch.py --data-dir training\scratch_corpus --output-dir outputs\kaptia-scratch-pretrain --steps 1000 --eval-interval 100 --eval-batches 2 --batch-size 4 --block-size 128 --layers 2 --heads 2 --width 64 --context-length 256 --device auto
```

This small configuration is only a bootstrap. It is not a claim that the resulting model has useful general conversation skills.

## Larger corpus training

Put material you own or are licensed to use in a separate folder. Supported input files are `.txt`, `.md`, and `.jsonl`; JSONL records may use a string `text` or `content` field. Keep private information and secrets out of the corpus.

```bat
python scripts\kaptia_prepare_text_corpus.py --input training\raw_text --output-dir training\scratch_corpus --val-fraction 0.02
python scripts\train_kaptia_scratch.py --data-dir training\scratch_corpus --output-dir outputs\kaptia-scratch-pretrain --steps 1000 --eval-interval 100 --device auto
```

To resume from a Kaptia scratch checkpoint:

```bat
python scripts\train_kaptia_scratch.py --data-dir training\scratch_corpus --output-dir outputs\kaptia-scratch-pretrain --steps 1000 --eval-interval 100 --resume outputs\kaptia-scratch-pretrain\latest_checkpoint.pt --device auto
```

The training script records training/validation loss, perplexity, step counts, and checkpoint provenance. It rejects checkpoints unless `initialization=random` and `pretrained_source=null`. A small loss or successful training loop alone does not establish language understanding; use held-out prompts and manual evaluation.

## Tokenizer and limitations

Kaptia currently uses reversible UTF-8 byte IDs (256 tokens), not a pretrained tokenizer. This is simple but inefficient for Chinese and can generate invalid UTF-8 sequences when undertrained. A better tokenizer may be trained from Kaptia's own corpus later; it must not be copied from a finished model.

The product installer preserves model data under the local Kaptia data directory so application updates do not erase training progress. Reinstalling continues the same scratch checkpoint; it does not import external weights.

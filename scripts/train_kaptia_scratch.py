"""Train KaptiaScratchLM from random initialization on prepared UTF-8 byte corpora."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from kaptia_scratch_model import KaptiaScratchLM, ScratchConfig


def load_tokens(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing corpus file: {path}. Run scripts/kaptia_prepare_text_corpus.py first.")
    raw = path.read_bytes()
    if len(raw) < 2:
        raise ValueError(f"Corpus is too small: {path}")
    # ByteTokenizer maps every UTF-8 byte directly to one token ID.
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()


def sample_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    if data.numel() <= block_size:
        raise ValueError(f"Corpus has {data.numel()} tokens; need more than block_size={block_size}.")
    starts = torch.randint(0, data.numel() - block_size, (batch_size,))
    x = torch.stack([data[int(i):int(i) + block_size] for i in starts])
    return x.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_batches):
    model.eval()
    metrics = {}
    for name, data in (("train", train_data), ("validation", val_data)):
        losses = []
        for _ in range(eval_batches):
            x = sample_batch(data, batch_size, block_size, device)
            losses.append(float(model(x, labels=x)["loss"].item()))
        avg = sum(losses) / len(losses)
        metrics[name] = {"loss": avg, "perplexity": math.exp(min(avg, 20.0))}
    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="True from-scratch causal-LM pretraining for Kaptia.")
    parser.add_argument("--data-dir", type=Path, default=Path("training/scratch_corpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kaptia-scratch-pretrain"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a random-init Kaptia checkpoint; pretrained checkpoints are rejected.")
    args = parser.parse_args()
    if min(args.steps, args.eval_interval, args.eval_batches, args.batch_size, args.block_size) < 1:
        parser.error("steps, eval-interval, eval-batches, batch-size, and block-size must be positive")
    if args.block_size > args.context_length:
        parser.error("--block-size cannot exceed --context-length")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive and weight-decay non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    train_path = args.data_dir / "train.bin"
    val_path = args.data_dir / "val.bin"
    train_data = load_tokens(train_path)
    val_data = load_tokens(val_path)
    corpus_fingerprint = {
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(val_path.read_bytes()).hexdigest(),
    }
    config = ScratchConfig(layers=args.layers, heads=args.heads, width=args.width, context_length=args.context_length)
    config.validate()
    model = KaptiaScratchLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_val = float("inf")
    if args.resume is not None:
        if not args.resume.is_file():
            parser.error(f"--resume checkpoint does not exist: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("initialization") != "random" or checkpoint.get("pretrained_source") is not None:
            parser.error("Refusing checkpoint: expected random initialization and no pretrained source.")
        saved_config = checkpoint.get("config", {})
        if any(saved_config.get(key) != value for key, value in config.__dict__.items()):
            parser.error("Checkpoint architecture config does not match requested model config.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("best_validation_loss", checkpoint.get("metrics", {}).get("validation", {}).get("loss", float("inf"))))
        if checkpoint.get("corpus_fingerprint") != corpus_fingerprint:
            print("[WARN] Corpus fingerprint changed or missing; resetting best-validation comparison for the new corpus.")
            best_val = float("inf")
        if start_step < 0:
            parser.error("Checkpoint has an invalid negative step.")
        # Restore RNG states so resumed sampling follows the saved training trajectory.
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(f"[INFO] Resuming random-init training at step {start_step}; RNG state restored when available; no pretrained weights loaded.")
    log_path = args.output_dir / "training_log.jsonl"
    started = time.time()
    print(f"[INFO] Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Train tokens(bytes): {train_data.numel():,}; validation tokens(bytes): {val_data.numel():,}")
    print("[INFO] Model starts with random weights; no pretrained checkpoint is loaded.")

    with log_path.open("a", encoding="utf-8") as log:
        for step in range(start_step + 1, start_step + args.steps + 1):
            x = sample_batch(train_data, args.batch_size, args.block_size, device)
            result = model(x, labels=x)
            loss = result["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}; stopping.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % args.eval_interval == 0 or step == start_step + args.steps:
                metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, device, args.eval_batches)
                record = {"step": step, "elapsed_seconds": round(time.time() - started, 2), **metrics}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(f"[step {step} total; segment {step - start_step}/{args.steps}] train_loss={metrics['train']['loss']:.4f} val_loss={metrics['validation']['loss']:.4f} val_ppl={metrics['validation']['perplexity']:.2f}")
                checkpoint = {
                    "config": config.__dict__,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "best_validation_loss": best_val,
                    "metrics": metrics,
                    "seed": args.seed,
                    "pretrained_source": None,
                    "initialization": "random",
                    "tokenizer": "UTF-8 byte tokenizer (256 tokens)",
                    "corpus_fingerprint": corpus_fingerprint,
                    "python_rng_state": random.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }
                torch.save(checkpoint, args.output_dir / "latest_checkpoint.pt")
                if metrics["validation"]["loss"] < best_val:
                    best_val = metrics["validation"]["loss"]
                    torch.save(checkpoint, args.output_dir / "best_checkpoint.pt")
                (args.output_dir / "manifest.json").write_text(json.dumps({
                    "status": "TRAINING_COMPLETED" if step == start_step + args.steps else "TRAINING_IN_PROGRESS",
                    "initialization": "random",
                    "pretrained_source": None,
                    "step": step,
                    "requested_steps": start_step + args.steps,
                    "steps_this_run": args.steps,
                    "best_validation_loss": best_val,
                    "device": str(device),
                    "corpus_dir": str(args.data_dir),
                    "corpus_fingerprint": corpus_fingerprint,
                    "log": str(log_path),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Training segment finished ({args.steps} additional steps). Checkpoints/log: {args.output_dir.resolve()}")
    print("[NOTE] A small dataset or short run is only a pipeline validation, not evidence of useful general language ability.")


if __name__ == "__main__":
    main()

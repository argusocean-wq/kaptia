"""Kaptia scratch decoder: random initialization, no pretrained checkpoint or downloads.

This is a minimal causal Transformer scaffold, not a trained assistant. The default
configuration is intentionally small for architecture smoke tests. A production
model size must be selected after corpus, memory, and compute are known.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ScratchConfig:
    vocab_size: int = 256  # UTF-8 byte vocabulary; no external tokenizer required
    context_length: int = 512
    layers: int = 6
    heads: int = 6
    width: int = 384
    dropout: float = 0.0
    model_type: str = "kaptia-scratch-decoder"
    initialization: str = "random"
    pretrained_source: str | None = None

    def validate(self) -> None:
        if self.vocab_size < 2 or self.context_length < 2:
            raise ValueError("vocab_size and context_length must be >= 2")
        if self.layers < 1 or self.heads < 1 or self.width < 1:
            raise ValueError("layers, heads, and width must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class ByteTokenizer:
    """Reversible UTF-8 byte encoding. Every input byte maps to an ID 0..255."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(token) for token in ids).decode("utf-8", errors="replace")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
                .view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        mask = self.causal_mask[:, :, :length, :length]
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, config: ScratchConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.width)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class KaptiaScratchLM(nn.Module):
    """Decoder-only language model initialized from random weights."""

    def __init__(self, config: ScratchConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        # Re-tie after initialization to make the weight sharing explicit.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if length > self.config.context_length:
            raise ValueError(f"sequence length {length} exceeds context_length={self.config.context_length}")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels shape must match input_ids")
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length:]
            logits = self(context)["logits"][:, -1, :] / temperature
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


def save_initial_model(output_dir: Path, config: ScratchConfig, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = KaptiaScratchLM(config)
    torch.save({"config": asdict(config), "model_state_dict": model.state_dict(), "training_steps": 0},
               output_dir / "initial_random_weights.pt")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "model": "Kaptia Scratch LM",
        "status": "UNTRAINED_RANDOM_INITIALIZATION",
        "pretrained_source": None,
        "tokenizer": "UTF-8 byte tokenizer (fixed 256-token vocabulary)",
        "seed": seed,
        "training_steps": 0,
        "warning": "Random weights do not constitute a useful trained language model.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[Kaptia] Random initialization saved to: {output_dir.resolve()}")
    print(f"[Kaptia] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("[Kaptia] Training steps: 0; pretrained weights: none")


def smoke_test(config: ScratchConfig) -> None:
    model = KaptiaScratchLM(config)
    tokenizer = ByteTokenizer()
    ids = torch.tensor([tokenizer.encode("Kaptia 測試")], dtype=torch.long)
    result = model(ids, labels=ids)
    if result["logits"].shape != (1, ids.shape[1], config.vocab_size):
        raise AssertionError("Unexpected logits shape")
    if result["loss"] is None or not torch.isfinite(result["loss"]):
        raise AssertionError("Loss is missing or non-finite")
    result["loss"].backward()
    print("[PASS] forward shape, finite loss, and backward pass")
    print("[INFO] This smoke test uses synthetic text; no model training is performed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaptia from-scratch neural-network scaffold")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kaptia-scratch-init"))
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = ScratchConfig(
        layers=args.layers, heads=args.heads, width=args.width, context_length=args.context_length
    )
    config.validate()
    if args.smoke_test:
        smoke_test(config)
    else:
        save_initial_model(args.output_dir, config, args.seed)


if __name__ == "__main__":
    main()

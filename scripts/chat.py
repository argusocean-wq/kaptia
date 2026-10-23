"""Kaptia local chat powered only by Kaptia's own randomly initialized scratch LM.

No pretrained model IDs, external tokenizer, adapter, or network model download is used.
"""
from __future__ import annotations

import codecs
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import torch
from torch.nn import functional as F

from kaptia_scratch_model import ByteTokenizer, KaptiaScratchLM, ScratchConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs" / "kaptia-scratch-pretrain" / "best_checkpoint.pt"


def checkpoint_path() -> Path:
    return Path(os.environ.get("KAPTIA_SCRATCH_CHECKPOINT", str(DEFAULT_CHECKPOINT))).expanduser()


def load_language_engine():
    path = checkpoint_path()
    if not path.is_file():
        raise FileNotFoundError(
            "找不到 Kaptia 自有模型 checkpoint。拒絕改用任何成品模型。\n"
            "請先執行 scripts\\open_kaptia.bat，讓它從隨機初始化開始訓練；"
            "或依 docs/KAPTIA_SCRATCH_PRETRAINING.md 手動準備語料與訓練。"
            f"\n預期位置：{path}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("initialization") != "random" or checkpoint.get("pretrained_source") is not None:
        raise RuntimeError("拒絕載入非隨機初始化 checkpoint；Kaptia 不允許成品模型權重。")
    config_data = checkpoint.get("config")
    if not isinstance(config_data, dict):
        raise RuntimeError("Checkpoint 缺少模型設定。")
    config = ScratchConfig(**config_data)
    model = KaptiaScratchLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    tokenizer = ByteTokenizer()
    print(f"[Kaptia] 自有 scratch LM 已載入：{path}")
    print(f"[Kaptia] 裝置：{device}；訓練步數：{checkpoint.get('step', 0)}；成品模型權重：未使用")
    print("[Kaptia] 目前是小規模起步模型；回答可能不連貫，需用更多合法語料與計算持續訓練。")
    return tokenizer, model


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+|[，。！？、,.!?；;：:（）()「」『』]", "", text).casefold()


def lookup_seed_response(user_text: str) -> str | None:
    """Clearly labeled lookup fallback from Kaptia's own tiny bootstrap examples."""
    path = ROOT / "training" / "bootstrap_dialogue_seed.jsonl"
    if not path.is_file():
        return None
    query = normalize_prompt(user_text)
    if not query:
        return None
    best_ratio = 0.0
    best_response = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample = item.get("text") if isinstance(item, dict) else None
            if not isinstance(sample, str):
                continue
            if "<|user|>" not in sample or "<|assistant|>" not in sample or "<|end|>" not in sample:
                continue
            remainder = sample.split("<|user|>", 1)[1]
            user_part, separator, assistant_part = remainder.partition("<|assistant|>")
            if not separator:
                continue
            response = assistant_part.split("<|end|>", 1)[0].strip()
            candidate = normalize_prompt(user_part)
            if candidate == query:
                return response
            ratio = SequenceMatcher(None, query, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_response = response
    return best_response if best_ratio >= 0.82 else None


def acceptable_generated_text(text: str) -> bool:
    if not text or "\ufffd" in text or len(text) < 8:
        return False
    if sum(ch.isprintable() for ch in text) / max(1, len(text)) < 0.98:
        return False
    if len(text) > 12 and len(set(text)) <= 3:
        return False
    return True


def build_generator(tokenizer, model):
    device = next(model.parameters()).device
    end_ids = tokenizer.encode("<|end|>")

    def generate(messages, max_new_tokens=256, temperature=0.45):
        # Keep the training format compact so it fits the small bootstrap context.
        parts = []
        for item in messages[-6:]:
            role = str(item.get("role", "user")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            parts.append(f"<|{role}|>{str(item.get('content', ''))}")
        parts.append("<|assistant|>")
        encoded = tokenizer.encode("".join(parts))
        if not encoded:
            encoded = tokenizer.encode("<|user|><|assistant|>")
        ids = torch.tensor([encoded], dtype=torch.long, device=device)
        generated: list[int] = []
        model.eval()
        with torch.inference_mode():
            for _ in range(max(1, min(int(max_new_tokens), 512))):
                context = ids[:, -model.config.context_length:]
                logits = model(context)["logits"][:, -1, :]
                logits = logits / max(0.15, float(temperature))
                # Byte-level sampling must preserve valid UTF-8, otherwise Chinese text
                # degrades into replacement characters even when the model predicts bytes.
                valid_ids = []
                decoder_factory = codecs.getincrementaldecoder("utf-8")
                prefix = bytes(generated)
                for candidate in range(256):
                    decoder = decoder_factory("strict")
                    try:
                        decoder.decode(prefix, final=False)
                        decoder.decode(bytes([candidate]), final=False)
                        valid_ids.append(candidate)
                    except UnicodeDecodeError:
                        continue
                if not valid_ids:
                    raise RuntimeError("No valid UTF-8 continuation token exists.")
                valid_mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
                valid_mask[valid_ids] = True
                logits = logits.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))
                k = min(40, len(valid_ids))
                values, indices = torch.topk(logits, k=k, dim=-1)
                probabilities = F.softmax(values, dim=-1)
                selected = indices.gather(-1, torch.multinomial(probabilities, 1))
                token_id = int(selected.item())
                generated.append(token_id)
                ids = torch.cat([ids, selected], dim=1)
                if len(generated) >= len(end_ids) and generated[-len(end_ids):] == end_ids:
                    generated = generated[:-len(end_ids)]
                    break
        return tokenizer.decode(generated).strip()

    return generate


def main() -> int:
    print("[Kaptia] 正在載入從零訓練的本機文字模型…")
    print("[Kaptia] 不載入第三方成品模型權重。")
    try:
        tokenizer, model = load_language_engine()
        generate = build_generator(tokenizer, model)
    except Exception as exc:
        print(f"[錯誤] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("[Kaptia] 已就緒。輸入 /exit 離開。")
    history: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Kaptia] 已結束。")
            break
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("[Kaptia] 已結束。")
            break
        if not user_text:
            continue
        history.append({"role": "user", "content": user_text})
        try:
            answer = generate(history, max_new_tokens=256, temperature=0.35)
            seed_answer = lookup_seed_response(user_text)
            if seed_answer is not None and (
                not acceptable_generated_text(answer)
                or SequenceMatcher(None, normalize_prompt(answer), normalize_prompt(seed_answer)).ratio() < 0.45
            ):
                answer = seed_answer
                print("\nKaptia（示範語料檢索；不是神經生成）：\n" + answer)
            elif acceptable_generated_text(answer) and os.environ.get("KAPTIA_SHOW_EXPERIMENTAL_GENERATION") == "1":
                print("\nKaptia（實驗性模型生成）：\n" + answer)
            elif seed_answer is not None:
                print("\nKaptia（模型生成）：\n" + answer)
            else:
                answer = "我目前仍在從零訓練階段，現有起步語料不足以讓我可靠回答這個問題。"
                print("\nKaptia（訓練狀態）：\n" + answer)
            history.append({"role": "assistant", "content": answer})
            history = history[-6:]
        except Exception as exc:
            print(f"[對話失敗] {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

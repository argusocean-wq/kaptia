"""Kaptia local chat powered only by Kaptia's own randomly initialized scratch LM.

No Transformers, Hugging Face model IDs, pretrained checkpoints, adapters, or network model downloads.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from kaptia_scratch_model import ByteTokenizer, KaptiaScratchLM, ScratchConfig
from kaptia_pipeline import run_workflow

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
    print("[Kaptia] 注意：目前屬於小規模起步訓練，回答品質取決於實際語料與訓練進度。")
    return tokenizer, model


def build_generator(tokenizer, model):
    device = next(model.parameters()).device
    end_ids = tokenizer.encode("<|end|>")

    def generate(messages, max_new_tokens=256, temperature=0.8):
        parts = []
        for item in messages:
            role = str(item.get("role", "user")).strip().lower()
            content = str(item.get("content", ""))
            parts.append(f"<|{role}|>{content}\n")
        parts.append("<|assistant|>")
        prompt = "".join(parts)
        encoded = tokenizer.encode(prompt)
        if not encoded:
            encoded = tokenizer.encode("<|user|>\n<|assistant|>")
        ids = torch.tensor([encoded], dtype=torch.long, device=device)
        generated: list[int] = []
        model.eval()
        with torch.inference_mode():
            for _ in range(max(1, min(int(max_new_tokens), 512))):
                context = ids[:, -model.config.context_length:]
                logits = model(context)["logits"][:, -1, :]
                logits = logits / max(0.15, float(temperature))
                # Top-k sampling avoids spending all probability mass on the full byte vocabulary.
                k = min(40, logits.shape[-1])
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
    print("[Kaptia] 不載入 Hugging Face、Ollama 或任何成品模型權重。")
    try:
        tokenizer, model = load_language_engine()
        generate = build_generator(tokenizer, model)
    except Exception as exc:
        print(f"[錯誤] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("[Kaptia] 已就緒。輸入 /exit 離開。")
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
        try:
            result = run_workflow(generate, user_text)
            print("\nKaptia：\n" + result["content"])
            print("\n[流程紀錄]")
            for event in result["events"]:
                print(f"- {event}")
        except Exception as exc:
            print(f"[對話失敗] {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

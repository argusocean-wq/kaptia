"""Compare base Qwen and Kaptia adapter on the held-out project-knowledge set.

Writes one JSON object per eval prompt to outputs/kaptia-language-qlora/comparison.jsonl.
This is a reproducible output capture, not an automatic quality score.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "training" / "eval_dataset.jsonl"
ADAPTER_DIR = ROOT / "outputs" / "kaptia-language-qlora" / "adapter"
OUT_PATH = ROOT / "outputs" / "kaptia-language-qlora" / "comparison.jsonl"
MODEL_ID = os.environ.get("KAPTIA_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("KAPTIA_EVAL_MAX_NEW_TOKENS", "256"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows():
    rows = []
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("instruction"), str) or not item["instruction"].strip():
                raise ValueError(f"驗收資料第 {line_no} 行 instruction 無效")
            rows.append(item)
    if not rows:
        raise ValueError("驗收資料為空")
    return rows


def generate(tokenizer, model, instruction, extra_input=""):
    user_text = instruction
    if extra_input.strip():
        user_text += "\n\n補充資料：\n" + extra_input.strip()
    messages = [
        {"role": "system", "content": "你是 Kaptia 的專案知識助理，使用繁體中文。不得捏造程式實作、測試結果或來源；資料不足時要明確說明。"},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max(1, min(MAX_NEW_TOKENS, 512)),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def main():
    if not ADAPTER_DIR.is_dir():
        raise FileNotFoundError(f"找不到已訓練 adapter：{ADAPTER_DIR}；請先完成訓練。")
    if not (ADAPTER_DIR / "adapter_config.json").is_file():
        raise FileNotFoundError("adapter_config.json 不存在；輸出資料不是可辨識的 PEFT adapter。")
    rows = load_rows()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    common = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        common["device_map"] = "auto"

    print(f"[Kaptia Eval] base={MODEL_ID}; prompts={len(rows)}")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **common)
    base_model.eval()
    adapter_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    adapter_model.eval()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows, 1):
            base_answer = generate(tokenizer, base_model, row["instruction"], row.get("input", ""))
            adapter_answer = generate(tokenizer, adapter_model, row["instruction"], row.get("input", ""))
            record = {
                "index": index,
                "instruction": row["instruction"],
                "base_model": MODEL_ID,
                "base_answer": base_answer,
                "kaptia_adapter_answer": adapter_answer,
                "human_review": {"base_score": None, "adapter_score": None, "notes": ""},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] captured; manually review both answers")
    print(f"[Kaptia Eval] output={OUT_PATH}")
    print(f"[Kaptia Eval] eval_sha256={sha256(EVAL_PATH)}")
    print("[Kaptia Eval] 未自動判定 adapter 較好；請逐題人工評分並記錄理由。")


if __name__ == "__main__":
    main()

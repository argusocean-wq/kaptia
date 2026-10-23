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
EVAL_PATHS = [ROOT / "training" / "eval_dataset.jsonl", ROOT / "training" / "code_eval_dataset.jsonl"]
ADAPTER_DIR = ROOT / "outputs" / "kaptia-language-qlora" / "adapter"
OUT_PATH = ROOT / "outputs" / "kaptia-language-qlora" / "comparison.jsonl"
MODEL_ID = os.environ.get("KAPTIA_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("KAPTIA_EVAL_MAX_NEW_TOKENS", "256"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows():
    rows = []
    for eval_path in EVAL_PATHS:
        with eval_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item.get("instruction"), str) or not item["instruction"].strip():
                    raise ValueError(f"{eval_path.name} 第 {line_no} 行 instruction 無效")
                item["_eval_source"] = eval_path.name
                rows.append(item)
    if not rows:
        raise ValueError("驗收資料為空")
    return rows

def generate(tokenizer, model, instruction, extra_input=""):
    user_text = instruction
    if extra_input.strip():
        user_text += "\n\n補充資料：\n" + extra_input.strip()
    messages = [
        {"role": "system", "content": "你是 Kaptia，預設使用臺灣繁體中文。你是嚴謹的程式設計與專案助理；先理解需求、明確假設、給出可驗證方案，不捏造測試結果、來源或工具執行。中華民國與中華人民共和國不可混淆。"},
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

    # Run separate model instances sequentially so loading PEFT cannot contaminate
    # the base-model baseline. This also lowers peak memory usage.
    base_answers = []
    print("[Kaptia Eval] Pass 1/2: base model")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **common)
    base_model.eval()
    for index, row in enumerate(rows, 1):
        base_answers.append(generate(tokenizer, base_model, row["instruction"], row.get("input", "") + (("\\n驗收標準： " + row["expected_result"]) if row.get("expected_result") else "")))
        print(f"[base {index}/{len(rows)}]")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[Kaptia Eval] Pass 2/2: Kaptia adapter")
    adapter_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **common)
    adapter_model = PeftModel.from_pretrained(adapter_base, str(ADAPTER_DIR))
    adapter_model.eval()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for index, (row, base_answer) in enumerate(zip(rows, base_answers), 1):
            adapter_answer = generate(tokenizer, adapter_model, row["instruction"], row.get("input", "") + (("\\n驗收標準： " + row["expected_result"]) if row.get("expected_result") else ""))
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
    del adapter_model, adapter_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Kaptia Eval] output={OUT_PATH}")
    for eval_path in EVAL_PATHS:
        print(f"[Kaptia Eval] {eval_path.name}_sha256={sha256(eval_path)}")
    print("[Kaptia Eval] 未自動判定 adapter 較好；請逐題人工評分並記錄理由。")


if __name__ == "__main__":
    main()

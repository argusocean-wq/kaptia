"""Standalone local Kaptia chat using Transformers; no Ollama required."""
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID = os.environ.get("KAPTIA_BASE_MODEL", DEFAULT_MODEL)
ADAPTER_DIR = ROOT / "outputs" / "kaptia-qwen2.5-7b-qlora" / "adapter"
SYSTEM_PROMPT = """你是 Kaptia，一個以繁體中文（台灣用語）為預設的本機 AI 助理。
清楚、務實、直接，優先協助合理的技術、程式設計、分析、寫作與學習任務。
不得聲稱已經執行掃描、編譯、部署、訓練或測試，除非確實執行並取得結果。
回報時區分已執行檢查、尚未執行的檢查、已知問題與風險。
不確定時明確說明，不捏造檔案、API、輸出或成功結果。
對正常技術研究與程式開發避免無理由的泛化拒答；若有實際風險，指出具體問題並盡量提供安全替代方案。"""


def main():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[Kaptia] CUDA GPU: {device_name}")
        dtype = torch.float16
    else:
        print("[Kaptia] WARNING: CUDA GPU not detected; CPU inference may be very slow.")
        dtype = torch.float32

    print(f"[Kaptia] Loading base model: {MODEL_ID}")
    local_only = os.environ.get("KAPTIA_LOCAL_ONLY", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=True, local_files_only=local_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    )

    if ADAPTER_DIR.is_dir():
        try:
            from peft import PeftModel
            print(f"[Kaptia] Loading LoRA adapter: {ADAPTER_DIR}")
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Adapter folder exists but could not be loaded: {ADAPTER_DIR}. "
                "Check adapter/base-model compatibility and the PEFT installation."
            ) from exc

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("[Kaptia] Ready. Type /exit to quit. First launch may download model files.")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_history = 12
    while True:
        try:
            user_text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Kaptia] Bye.")
            break
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("[Kaptia] Bye.")
            break
        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})
        if len(history) > max_history + 1:
            history = [history[0]] + history[-max_history:]

        try:
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            input_device = model.get_input_embeddings().weight.device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=int(os.environ.get("KAPTIA_MAX_NEW_TOKENS", "512")),
                    do_sample=True,
                    temperature=0.35,
                    top_p=0.9,
                    repetition_penalty=1.08,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            print(f"\nKaptia：{answer}")
            history.append({"role": "assistant", "content": answer})
        except torch.cuda.OutOfMemoryError as exc:
            if history and history[-1]["role"] == "user":
                history.pop()
            print("[ERROR] GPU VRAM 不足。請關閉其他 GPU 程式或改用較小模型/較短輸出。", file=sys.stderr)
            torch.cuda.empty_cache()
        except Exception as exc:
            if history and history[-1]["role"] == "user":
                history.pop()
            print(f"[ERROR] 生成失敗：{type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

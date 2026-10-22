"""Standalone local Kaptia language assistant using Transformers; no Ollama required."""
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
# Internal implementation detail; never expose the provider/model ID in the chat UI.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID = os.environ.get("KAPTIA_BASE_MODEL", DEFAULT_MODEL)
ADAPTER_DIR = ROOT / "outputs" / "kaptia-language-qlora" / "adapter"
SYSTEM_PROMPT = """你是 Kaptia，一個以繁體中文（台灣用語）為預設的本機語言助理。
清楚、務實、直接，優先協助合理的技術、程式設計、分析、寫作與學習任務。
你只使用文字對話能力；不要聲稱自己具備未接入的影像、語音、網路搜尋或外部工具能力。
不得聲稱已經執行掃描、編譯、部署、訓練或測試，除非確實執行並取得結果。
回報時區分已執行檢查、尚未執行的檢查、已知問題與風險。
不確定時明確說明，不捏造檔案、API、輸出或成功結果。
對正常技術研究與程式開發避免無理由的泛化拒答；若有實際風險，指出具體問題並盡量提供安全替代方案。"""


def load_language_engine():
    local_only = os.environ.get("KAPTIA_LOCAL_ONLY", "0") == "1"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
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
    except Exception as exc:
        mode = "離線模式" if local_only else "模型載入"
        raise RuntimeError(
            f"{mode}失敗。請確認首次下載已完成、網路/磁碟空間正常，"
            "且模型檔案與目前環境相容。詳細原因："
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if ADAPTER_DIR.is_dir():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model, str(ADAPTER_DIR), local_files_only=True
            )
        except Exception as exc:
            raise RuntimeError(
                "本機微調資料載入失敗；請確認 adapter 與基礎語言模型相容。"
            ) from exc

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def main():
    print("[Kaptia] 正在啟動本機文字助理…")
    if torch.cuda.is_available():
        print("[Kaptia] 已偵測到相容的 GPU，準備載入語言引擎。")
    else:
        print("[Kaptia] 未偵測到 CUDA GPU，將嘗試使用 CPU；速度可能較慢。")

    try:
        tokenizer, model = load_language_engine()
    except Exception as exc:
        print(f"[錯誤] {exc}", file=sys.stderr)
        return 1

    print("[Kaptia] 已就緒。輸入 /exit 離開。")
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_history = 12

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
                    max_new_tokens=max(1, min(
                        int(os.environ.get("KAPTIA_MAX_NEW_TOKENS", "512")), 4096
                    )),
                    do_sample=True,
                    temperature=0.35,
                    top_p=0.9,
                    repetition_penalty=1.08,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if not answer:
                answer = "這次沒有產生文字回覆，請換個方式提問再試一次。"
            print(f"\nKaptia：{answer}")
            history.append({"role": "assistant", "content": answer})
        except torch.cuda.OutOfMemoryError:
            if history and history[-1]["role"] == "user":
                history.pop()
            print("[錯誤] GPU 記憶體不足。請關閉其他 GPU 程式或降低輸出長度。", file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            if history and history[-1]["role"] == "user":
                history.pop()
            print(f"[錯誤] 回覆生成失敗：{type(exc).__name__}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

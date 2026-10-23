"""Standalone local Kaptia assistant with a staged response and delivery workflow."""
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kaptia_pipeline import run_workflow

ROOT = Path(__file__).resolve().parents[1]
# Kaptia uses a replaceable local language backend; this default is not Kaptia-owned weights.
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_ID = os.environ.get("KAPTIA_BASE_MODEL", DEFAULT_MODEL)
ADAPTER_DIR = ROOT / "outputs" / "kaptia-smollm2-qlora" / "adapter"


def load_language_engine():
    local_only = os.environ.get("KAPTIA_LOCAL_ONLY", "0") == "1"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=True, local_files_only=local_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True, local_files_only=local_only,
        )
    except Exception as exc:
        mode = "離線模式" if local_only else "模型載入"
        raise RuntimeError(
            f"{mode}失敗。請確認模型下載、網路/磁碟空間與環境相容性。"
            f"詳細原因：{type(exc).__name__}: {exc}"
        ) from exc

    if ADAPTER_DIR.is_dir():
        try:
            import json
            config_path = ADAPTER_DIR / "adapter_config.json"
            if not config_path.is_file():
                raise RuntimeError("adapter_config.json 不存在；拒絕載入不完整的微調資料。")
            with config_path.open("r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            adapter_base = str(adapter_config.get("base_model_name_or_path", "")).casefold()
            if adapter_base and adapter_base != MODEL_ID.casefold():
                raise RuntimeError(f"微調 adapter 基礎模型不相容：{adapter_base}；目前模型：{MODEL_ID}。請重新訓練。")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), local_files_only=True)
        except Exception as exc:
            raise RuntimeError("本機微調資料載入失敗；請確認 adapter 與基礎語言模型相容。") from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def build_generator(tokenizer, model):
    def generate(messages, max_new_tokens=700, temperature=0.35):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_device = model.get_input_embeddings().weight.device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max(1, min(int(max_new_tokens), 1400)),
                do_sample=temperature > 0,
                temperature=max(0.1, float(temperature)),
                top_p=0.9, repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return generate


def main():
    print("[Kaptia] 正在啟動本機文字助理…")
    print("[Kaptia] 理解 → 技能指南 → 查證判斷 → 格式規劃 → 生成 → 驗證 → 交付")
    if torch.cuda.is_available():
        print("[Kaptia] 已偵測到相容的 GPU，準備載入語言引擎。")
    else:
        print("[Kaptia] 未偵測到 CUDA GPU，將嘗試使用 CPU；速度可能較慢。")
    try:
        tokenizer, model = load_language_engine()
        generate = build_generator(tokenizer, model)
    except Exception as exc:
        print(f"[錯誤] {exc}", file=sys.stderr)
        return 1

    print("[Kaptia] 已就緒。輸入 /exit 離開。")
    print("[Kaptia] 長內容會先列大綱再分段；本機自我審核不等於外部事實查證。")
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
        if user_text == "/memory list":
            try:
                from kaptia_memory import list_records
                records = list_records(limit=30)
                print("[專案記憶] " + (f"共 {len(records)} 筆" if records else "目前沒有紀錄"))
                for row in records:
                    print(f"- [{row['kind']}/{row['status']}] {row['title']}: {row['detail']}")
            except Exception as exc:
                print(f"[記憶讀取失敗] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if user_text.startswith("/memory add "):
            try:
                from kaptia_memory import append_record
                parts = user_text[len("/memory add "):].split("|", 2)
                if len(parts) != 3:
                    print("格式：/memory add decision|標題|內容")
                    print("kind 可用 decision、task、note、blocker；只會寫入你明確指定的專案資訊。")
                else:
                    record = append_record(parts[0].strip(), parts[1].strip(), parts[2].strip())
                    print(f"[專案記憶已保存] {record['kind']}: {record['title']}")
            except Exception as exc:
                print(f"[記憶寫入失敗] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if user_text == "/tasks list":
            try:
                from kaptia_tasks import list_tasks
                tasks = list_tasks(limit=50)
                print("[任務] " + (f"共 {len(tasks)} 筆" if tasks else "目前沒有任務"))
                for task in tasks:
                    print(f"- [{task['status']}] {task['id'][:8]} {task['title']}: {task['detail']}")
            except Exception as exc:
                print(f"[任務讀取失敗] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if user_text.startswith("/tasks add "):
            try:
                from kaptia_tasks import create_task
                parts = user_text[len("/tasks add "):].split("|", 1)
                if len(parts) != 2:
                    print("格式：/tasks add 標題|任務內容")
                else:
                    task = create_task(parts[0].strip(), parts[1].strip())
                    print(f"[任務已建立] {task['id']} [{task['status']}] {task['title']}")
            except Exception as exc:
                print(f"[任務建立失敗] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if user_text.startswith("/tasks set "):
            try:
                from kaptia_tasks import transition_task
                parts = user_text[len("/tasks set "):].split("|", 2)
                if len(parts) != 3:
                    print("格式：/tasks set 任務ID|新狀態|變更原因")
                    print("狀態：in_progress、blocked、completed、failed、cancelled")
                else:
                    task = transition_task(parts[0].strip(), parts[1].strip(), parts[2].strip())
                    print(f"[任務已更新] {task['id']} [{task['status']}] {task['title']}")
            except Exception as exc:
                print(f"[任務更新失敗] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        try:
            result = run_workflow(generate, user_text)
            print("\nKaptia：\n" + result["content"])
            print("\n[流程紀錄]")
            for event in result["events"]:
                print(f"  • {event}")
            review = result.get("review", {})
            print(f"[自我審核] {'通過' if review.get('pass') else '未能確認通過'}（非獨立事實查核）")
            for issue in review.get("issues", []):
                print(f"  - 待檢查：{issue}")
            if result["artifact_path"]:
                print(f"[交付檔案] {result['artifact_path']}")
        except torch.cuda.OutOfMemoryError:
            print("[錯誤] GPU 記憶體不足。請關閉其他 GPU 程式或降低任務長度。", file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[錯誤] 任務處理失敗：{type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""QLoRA fine-tuning for Kaptia; evaluates on a held-out JSONL set.

This script trains an adapter, not a standalone model. It requires a compatible
NVIDIA CUDA environment. Assistant completion tokens are the only loss targets.
"""
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "training" / "seed_dataset.jsonl"
EVAL_PATH = ROOT / "training" / "eval_dataset.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "kaptia-language-qlora"
MODEL_ID = os.environ.get("KAPTIA_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_LENGTH = int(os.environ.get("KAPTIA_MAX_LENGTH", "1024"))


def load_examples(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("instruction", "output"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"{path.name}:{line_no} 缺少非空欄位 {key}")
            user_text = item["instruction"]
            if item.get("input", "").strip():
                user_text += "\n\n補充資料：\n" + item["input"].strip()
            rows.append({"user_text": user_text, "answer": item["output"].strip()})
    if len(rows) < 2:
        raise ValueError(f"{path.name} 至少需要 2 筆有效資料")
    return rows


class CompletionOnlyCollator:
    """Pad token ids and labels independently; -100 labels are ignored by CE loss."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in features],
            padding=True, return_tensors="pt"
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            row = feature["labels"]
            labels.append(row + [-100] * (max_len - len(row)))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def tokenise_example(example, tokenizer):
    messages = [
        {"role": "system", "content": "你是 Kaptia，預設使用繁體中文。專案事實以提供的文件為準；誠實區分已驗證、推論與尚未執行的工作。"},
        {"role": "user", "content": example["user_text"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["answer"]}],
        tokenize=True, add_generation_prompt=False
    )
    # Some chat templates add a token at the boundary. Require a strict prefix so
    # prompt tokens can never silently become training targets.
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template 的 prompt/full token 邊界不一致；停止訓練以避免錯誤標籤")
    if len(full_ids) > MAX_LENGTH:
        full_ids = full_ids[:MAX_LENGTH]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    if not any(x != -100 for x in labels):
        raise ValueError("答案在截斷後沒有可訓練 token；請縮短範例或提高 KAPTIA_MAX_LENGTH")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("未偵測到 CUDA GPU；QLoRA 訓練需要相容的 NVIDIA CUDA 環境。")
    if not TRAIN_PATH.is_file() or not EVAL_PATH.is_file():
        raise FileNotFoundError("找不到訓練或驗收資料；請先執行 scripts/validate_training_data.py。")
    from validate_training_data import main as validate_data
    if validate_data() != 0:
        raise RuntimeError("訓練資料驗證失敗，停止訓練。")

    print("[Kaptia] QLoRA 訓練開始")
    print(f"[Kaptia] 基礎模型：{MODEL_ID}")
    print(f"[Kaptia] GPU：{torch.cuda.get_device_name(0)}")
    print(f"[Kaptia] max_length={MAX_LENGTH}")
    train_rows, eval_rows = load_examples(TRAIN_PATH), load_examples(EVAL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds = Dataset.from_list(train_rows).map(
        lambda row: tokenise_example(row, tokenizer),
        remove_columns=["user_text", "answer"], desc="Tokenizing train"
    )
    eval_ds = Dataset.from_list(eval_rows).map(
        lambda row: tokenise_example(row, tokenizer),
        remove_columns=["user_text", "answer"], desc="Tokenizing eval"
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=float(os.environ.get("KAPTIA_EPOCHS", "2")),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=CompletionOnlyCollator(tokenizer),
    )
    result = trainer.train()
    eval_metrics = trainer.evaluate()
    adapter_path = OUTPUT_DIR / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    trainer.save_state()
    metrics = {"base_model": MODEL_ID, "train_metrics": result.metrics, "eval_metrics": eval_metrics,
               "train_examples": len(train_rows), "eval_examples": len(eval_rows),
               "max_length": MAX_LENGTH, "epochs": args.num_train_epochs}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Kaptia] 訓練結束；請注意這不代表品質一定提升")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Kaptia] Adapter：{adapter_path}")
    print("[Kaptia] Adapter 不是獨立模型，仍需要相容的基礎模型。")


if __name__ == "__main__":
    main()

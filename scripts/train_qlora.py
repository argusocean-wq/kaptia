"""Initial QLoRA trainer for standalone Kaptia. Review data and dependencies before use."""
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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "training" / "seed_dataset.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "kaptia-language-qlora"
MODEL_ID = os.environ.get("KAPTIA_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_LENGTH = int(os.environ.get("KAPTIA_MAX_LENGTH", "1024"))


def load_examples(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是有效 JSON：{exc}") from exc
            for key in ("instruction", "output"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"第 {line_no} 行缺少非空欄位：{key}")
            user_text = item["instruction"]
            if item.get("input"):
                user_text += "\n\n補充資料：\n" + str(item["input"])
            rows.append({"messages": [
                {"role": "system", "content": "你是 Kaptia，預設使用繁體中文。誠實區分已驗證事實、推論與尚未執行的工作。"},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": item["output"]},
            ]})
    if len(rows) < 2:
        raise ValueError("至少需要 2 筆有效訓練範例。")
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("未偵測到 CUDA GPU；目前訓練流程需要相容的 NVIDIA CUDA 環境。")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到訓練資料：{DATA_PATH}")

    print("Kaptia 微調流程啟動")
    print(f"訓練資料：{DATA_PATH}")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"最大序列長度：{MAX_LENGTH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list(load_examples(DATA_PATH))

    def render_and_tokenize(row):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding=False)

    dataset = dataset.map(
        render_and_tokenize, remove_columns=["messages"], desc="Tokenizing"
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant_config,
        device_map="auto", torch_dtype=torch.float16
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=float(os.environ.get("KAPTIA_EPOCHS", "2")),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    result = trainer.train()
    adapter_path = OUTPUT_DIR / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    trainer.save_state()
    print("微調流程完成")
    print(f"訓練指標：{result.metrics}")
    print(f"Adapter 輸出位置：{adapter_path}")
    print("注意：Adapter 不是獨立模型，仍需要相容的基礎語言模型。")


if __name__ == "__main__":
    main()

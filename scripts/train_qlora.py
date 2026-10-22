"""Initial QLoRA trainer for Kaptia. Run only after reviewing data and dependencies."""
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
OUTPUT_DIR = ROOT / "outputs" / "kaptia-qwen2.5-7b-qlora"
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
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            for key in ("instruction", "output"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"Line {line_no} requires non-empty string field '{key}'")
            user_text = item["instruction"]
            if item.get("input"):
                user_text += "\n\n補充資料：\n" + str(item["input"])
            rows.append({"messages": [
                {"role": "system", "content": "你是 Kaptia，預設使用繁體中文。誠實區分已驗證事實、推論與尚未執行的工作。"},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": item["output"]},
            ]})
    if len(rows) < 2:
        raise ValueError("Need at least 2 valid examples for a training smoke test.")
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. This starter script expects an NVIDIA CUDA environment.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    print(f"Base model: {MODEL_ID}")
    print(f"Dataset: {DATA_PATH}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Max sequence length: {MAX_LENGTH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples(DATA_PATH)

    def render_and_tokenize(row):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding=False)

    dataset = Dataset.from_list(examples).map(
        render_and_tokenize,
        remove_columns=["messages"],
        desc="Formatting and tokenizing examples",
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
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
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "adapter"))
    trainer.save_state()
    print("TRAINING_FINISHED")
    print(f"Train metrics: {result.metrics}")
    print(f"Adapter output: {OUTPUT_DIR / 'adapter'}")
    print("Note: this is a LoRA adapter, not a standalone Ollama model. Evaluate before merging/conversion.")


if __name__ == "__main__":
    main()

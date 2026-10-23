"""Merge a trained Kaptia PEFT adapter into standalone Hugging Face weights.

This is a separate, memory-intensive export step. It does not convert to ONNX
or prove browser/iPhone compatibility.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "outputs" / "kaptia-language-qlora" / "adapter"
DEFAULT_OUTPUT = ROOT / "outputs" / "kaptia-standalone"


def main():
    parser = argparse.ArgumentParser(description="Export trained Kaptia adapter as standalone weights.")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=os.environ.get("KAPTIA_BASE_MODEL", ""))
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Only enable for a base model whose code you have reviewed.")
    args = parser.parse_args()

    adapter_dir = args.adapter.resolve()
    output_dir = args.output.resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"找不到已訓練的 adapter：{adapter_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"輸出目錄不是空目錄，為避免覆寫而停止：{output_dir}")

    adapter_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_model_id = args.base_model.strip() or adapter_config.base_model_name_or_path
    if not base_model_id:
        raise ValueError("無法確認 adapter 對應的基礎模型；請明確指定 --base-model。")
    if adapter_config.base_model_name_or_path and args.base_model and (
        adapter_config.base_model_name_or_path != args.base_model
    ):
        raise ValueError(
            "指定的基礎模型與 adapter_config.json 不一致："
            f"{args.base_model!r} != {adapter_config.base_model_name_or_path!r}"
        )

    print(f"[Kaptia Export] base_model={base_model_id}")
    print(f"[Kaptia Export] adapter={adapter_dir}")
    print(f"[Kaptia Export] output={output_dir}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # Merging requires loading full-precision-ish base weights, not 4-bit QLoRA weights.
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )
    tuned = PeftModel.from_pretrained(base, str(adapter_dir))
    tuned.eval()
    merged = tuned.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir), trust_remote_code=args.trust_remote_code
    )
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "product_name": "Kaptia",
        "model_kind": "merged_causal_lm",
        "base_model": base_model_id,
        "adapter_path": str(adapter_dir),
        "export_format": "Hugging Face Transformers safetensors",
        "browser_onnx_compatible": False,
        "training_or_export_does_not_prove_quality": True,
        "next_steps": [
            "Review base-model and adapter licenses before distribution.",
            "Run scripts/evaluate_adapter.py and manually score held-out answers.",
            "Treat browser/ONNX conversion as a separate validated step.",
        ],
    }
    (output_dir / "kaptia_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    weight_files = list(output_dir.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError("匯出後找不到 .safetensors 權重檔，拒絕回報成功。")
    print("[Kaptia Export] PASS: standalone weights and tokenizer saved")
    print(f"[Kaptia Export] shards={len(weight_files)}")
    print("[Kaptia Export] This is not an ONNX export and has not been tested in Safari.")


if __name__ == "__main__":
    main()

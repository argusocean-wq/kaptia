"""Preflight report for Kaptia QLoRA training; does not download or train a model."""
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "language_train": ROOT / "training" / "seed_dataset.jsonl",
    "language_eval": ROOT / "training" / "eval_dataset.jsonl",
    "code_train": ROOT / "training" / "code_seed_dataset.jsonl",
    "code_eval": ROOT / "training" / "code_eval_dataset.jsonl",
}


def count_rows(path):
    if not path.is_file():
        return {"exists": False, "rows": 0, "bytes": 0}
    with path.open("r", encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip()]
    return {"exists": True, "rows": len(rows), "bytes": path.stat().st_size}


def main():
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "datasets": {name: count_rows(path) for name, path in DATASETS.items()},
        "cuda": {"available": False, "device": None, "vram_bytes": None},
        "training_started": False,
    }
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda"]["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["cuda"].update({
                "device": props.name,
                "vram_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
            })
    except ImportError:
        report["torch"] = None
        report["warnings"] = ["PyTorch is not installed in this Python environment."]

    missing = [name for name, info in report["datasets"].items() if not info["exists"]]
    tiny = [name for name, info in report["datasets"].items() if info["rows"] < 20]
    warnings = report.setdefault("warnings", [])
    if missing:
        warnings.append("Missing datasets: " + ", ".join(missing))
    if tiny:
        warnings.append("Very small datasets (<20 rows): " + ", ".join(tiny))
    if not report["cuda"]["available"]:
        warnings.append("CUDA GPU not detected; train_qlora.py will stop without training.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # A preflight report is useful even when hardware is not ready; nonzero only
    # when required dataset files are missing, not merely when CUDA is unavailable.
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

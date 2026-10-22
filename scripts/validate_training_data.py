"""Validate Kaptia's JSONL training/evaluation data and prevent obvious split leakage."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training" / "seed_dataset.jsonl"
EVAL = ROOT / "training" / "eval_dataset.jsonl"


def load(path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到資料檔：{path}")
    rows, seen = [], set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no} JSON 格式錯誤：{exc}") from exc
            for key in ("instruction", "output"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"{path.name}:{line_no} 缺少非空字串欄位 {key}")
            if "input" in item and not isinstance(item["input"], str):
                raise ValueError(f"{path.name}:{line_no} input 必須是字串")
            signature = (item["instruction"].strip(), item.get("input", "").strip(), item["output"].strip())
            if signature in seen:
                raise ValueError(f"{path.name}:{line_no} 出現完全重複資料")
            seen.add(signature)
            rows.append(item)
    if len(rows) < 2:
        raise ValueError(f"{path.name} 至少需要 2 筆資料；目前 {len(rows)} 筆")
    return rows


def main():
    train, evaluation = load(TRAIN), load(EVAL)
    train_prompts = {(r["instruction"].strip(), r.get("input", "").strip()) for r in train}
    eval_prompts = {(r["instruction"].strip(), r.get("input", "").strip()) for r in evaluation}
    overlap = train_prompts & eval_prompts
    if overlap:
        print(f"[FAIL] train/eval 指令重疊 {len(overlap)} 筆：", file=sys.stderr)
        for instruction, _ in sorted(overlap):
            print(f"  - {instruction}", file=sys.stderr)
        return 1
    print("[PASS] JSONL 格式與必要欄位有效")
    print(f"[INFO] 訓練資料：{len(train)} 筆")
    print(f"[INFO] 驗收資料：{len(evaluation)} 筆")
    print("[PASS] 沒有完全重複資料，且 train/eval 指令未重疊")
    print("[WARN] 格式通過不代表內容正確；仍需人工審核來源、授權、隱私與答案品質")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

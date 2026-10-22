"""Validate Kaptia's code dataset schema, duplicate prompts, and train/eval leakage."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training" / "code_seed_dataset.jsonl"
EVAL = ROOT / "training" / "code_eval_dataset.jsonl"
REQUIRED = ("instruction", "output")
EVAL_REQUIRED = ("id", "language", "task_type", "difficulty", "instruction", "output", "expected_result", "review_status")


def load_rows(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"找不到資料集：{path}")
    rows = []
    signatures = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no} JSON 格式錯誤：{exc}") from exc
            missing = [key for key in required if not isinstance(row.get(key), str) or not row[key].strip()]
            if missing:
                raise ValueError(f"{path.name}:{line_no} 缺少必要非空字串欄位：{', '.join(missing)}")
            if "input" in row and not isinstance(row["input"], str):
                raise ValueError(f"{path.name}:{line_no} input 必須是字串")
            signature = (row["instruction"].strip(), row.get("input", "").strip())
            if signature in signatures:
                raise ValueError(f"{path.name}:{line_no} 重複指令與輸入")
            signatures.add(signature)
            rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path.name} 至少需要 2 筆資料；目前 {len(rows)} 筆")
    return rows


def main():
    train = load_rows(TRAIN, REQUIRED)
    evaluation = load_rows(EVAL, EVAL_REQUIRED)
    train_prompts = {(row["instruction"].strip(), row.get("input", "").strip()) for row in train}
    eval_prompts = {(row["instruction"].strip(), row.get("input", "").strip()) for row in evaluation}
    overlap = train_prompts & eval_prompts
    if overlap:
        print(f"[FAIL] 程式碼 train/eval 指令重疊 {len(overlap)} 筆：", file=sys.stderr)
        for instruction, _ in sorted(overlap):
            print(f"  - {instruction}", file=sys.stderr)
        return 1
    ids = [row["id"] for row in evaluation]
    if len(ids) != len(set(ids)):
        raise ValueError("程式碼驗收資料存在重複 id")
    print(f"[PASS] 程式碼訓練資料格式有效：{len(train)} 筆")
    print(f"[PASS] 程式碼驗收資料格式有效：{len(evaluation)} 筆")
    print("[PASS] 程式碼 train/eval 指令未重疊")
    print("[WARN] 這只驗證格式與明顯洩漏；未執行各語言程式，也未驗證參考答案正確性。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

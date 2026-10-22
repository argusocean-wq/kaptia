"""Validate the schema and IDs of Kaptia's multilingual code evaluation dataset."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "code_eval_dataset.jsonl"
REQUIRED = ("id", "language", "task_type", "difficulty", "instruction", "output", "expected_result", "review_status")


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"找不到資料集：{DATA}")
    seen = set()
    count = 0
    with DATA.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{DATA.name}:{line_no} JSON 格式錯誤：{exc}") from exc
            missing = [key for key in REQUIRED if not isinstance(row.get(key), str) or not row[key].strip()]
            if missing:
                raise ValueError(f"{DATA.name}:{line_no} 缺少必要字串欄位：{', '.join(missing)}")
            if row["id"] in seen:
                raise ValueError(f"{DATA.name}:{line_no} 重複 id：{row['id']}")
            seen.add(row["id"])
            count += 1
    if count < 2:
        raise ValueError(f"驗收資料至少需要 2 筆，目前 {count} 筆")
    print(f"[PASS] 程式碼驗收資料格式有效：{count} 筆")
    print("[WARN] 所有案例仍需人工審查；資料格式有效不代表參考答案正確或已執行測試。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Kaptia's staged response workflow.

The planner and writer use the same local language model. This module does not claim
web research, external tools, or automated policy guarantees that are not integrated.
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

PLANNER_PROMPT = """你是 Kaptia 的任務規劃器。只輸出一個有效 JSON 物件，不要 Markdown code fence。
根據使用者請求決定：
{
 "task_size":"short" 或 "long",
 "output_type":"text"|"document"|"presentation"|"code"|"table"|"other",
 "needs_research":true 或 false,
 "needs_file":true 或 false,
 "outline":["必要時的章節標題"],
 "assumptions":["重要且會影響結果的假設"],
 "checks":["生成期間要遵守的具體檢查"]
}
短問答、簡短改寫等選 short。長文件、多頁簡報、完整方案等選 long，先列大綱再分段生成。
只有在任務確實需要外部最新資料或來源查證時才標 needs_research=true。你沒有網路搜尋或外部工具，不能假裝已查證；計畫中應標明需要使用者提供來源或之後接入工具。
needs_file 只在使用者明確要檔案、長文件、簡報內容或可交付程式碼時為 true。
安全、版權、隱私與兒童安全考量要在每個生成階段持續套用，不是最後才補做。不要把政治中立誤解成避談政治；應公平呈現證據與觀點。"""

WRITER_RULES = """你是 Kaptia，本機繁體中文文字助理。
務實、直接、清楚；不要自稱 Claude 或其他模型。
你目前沒有已整合的網路搜尋、瀏覽器、檔案系統掃描或外部工具；不可聲稱使用了這些工具。
每一階段都必須避免逐字重製未提供的受版權保護長篇內容、避免協助有害行為、保護兒童安全與個人隱私。政治議題以事實、證據與多元觀點公平處理，不捏造平衡或虛構來源。
不要捏造執行結果、引用、網址、檔案或驗證狀態。若資料不足，清楚標示假設與限制。"""


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("規劃器沒有回傳 JSON")
    data = json.loads(text[start:end + 1])
    if data.get("task_size") not in {"short", "long"}:
        raise ValueError("規劃器的 task_size 無效")
    if data.get("output_type") not in {"text", "document", "presentation", "code", "table", "other"}:
        data["output_type"] = "text"
    for key in ("outline", "assumptions", "checks"):
        if not isinstance(data.get(key), list):
            data[key] = []
    data["needs_research"] = bool(data.get("needs_research", False))
    data["needs_file"] = bool(data.get("needs_file", False))
    return data


def plan_task(generate, user_text):
    """Understand request, choose output format, and create a plan before writing."""
    raw = generate(
        [{"role": "system", "content": PLANNER_PROMPT},
         {"role": "user", "content": user_text}],
        max_new_tokens=500,
        temperature=0.1,
    )
    try:
        plan = _extract_json(raw)
    except Exception:
        # Fail safe to a simple answer instead of executing an unparseable plan.
        plan = {
            "task_size": "short", "output_type": "text",
            "needs_research": False, "needs_file": False,
            "outline": [], "assumptions": [],
            "checks": ["遵守安全、版權、隱私與事實誠信原則"],
        }
    if plan["task_size"] == "long" and not plan["outline"]:
        plan["outline"] = ["目的與背景", "主要內容", "限制與注意事項", "結論"]
    return plan


def _write_section(generate, user_text, plan, title, previous_summary=""):
    checks = "\n".join(f"- {item}" for item in plan["checks"]) or "- 持續遵守安全、版權、隱私與事實誠信原則"
    prompt = f"""使用者原始需求：
{user_text}

輸出形式：{plan['output_type']}
本段標題：{title}
整體大綱：{json.dumps(plan['outline'], ensure_ascii=False)}
假設：{json.dumps(plan['assumptions'], ensure_ascii=False)}
前段摘要（維持一致性）：{previous_summary or "目前沒有前段內容"}

全程檢查：
{checks}

請只撰寫本段，內容要具體、避免重複其他段落。若任務需要外部查證，不能假裝查過；明確標示待查證項目。"""
    return generate(
        [{"role": "system", "content": WRITER_RULES},
         {"role": "user", "content": prompt}],
        max_new_tokens=1100 if plan["task_size"] == "long" else 700,
        temperature=0.35,
    ).strip()


def run_workflow(generate, user_text):
    """Return response, stage events, plan, and optional deliverable path."""
    events = ["理解需求", "選擇輸出形式"]
    plan = plan_task(generate, user_text)
    events.append("建立任務計畫")
    if plan["needs_research"]:
        events.append("需要外部查證，但目前沒有整合搜尋工具")
    sections = []
    if plan["task_size"] == "short":
        content = _write_section(generate, user_text, plan, "完整回覆")
        sections.append(content)
        events.append("一次完成短內容")
    else:
        events.append("先建立大綱，再逐段生成")
        summary = ""
        for title in plan["outline"][:12]:
            section = _write_section(generate, user_text, plan, title, summary)
            sections.append(f"## {title}\n\n{section}")
            summary = "\n".join(sections)[-1800:]
            events.append(f"完成段落：{title}")
    content = "\n\n".join(sections).strip()

    if plan["needs_research"]:
        caveat = "\n\n> 查證狀態：本次沒有可用的網路搜尋工具；涉及最新資訊或外部來源的內容需另行查證。"
        content += caveat
    events.append("生成期間持續套用安全、版權、隱私與事實誠信規則")

    artifact_path = None
    if plan["needs_file"] and content:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = {"presentation": "slides", "code": "code", "document": "document"}.get(plan["output_type"], "content")
        artifact_path = ARTIFACTS_DIR / f"kaptia-{suffix}-{stamp}.md"
        artifact_path.write_text(content + "\n", encoding="utf-8")
        events.append(f"已將交付內容寫入本機檔案：{artifact_path}")
    events.append("交付回覆與檔案位置")
    return {"content": content, "plan": plan, "events": events, "artifact_path": str(artifact_path) if artifact_path else None}

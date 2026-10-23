"""Kaptia's staged response workflow with explicit verification and local delivery.

The planner, writer and reviewer use the same local language model. This is not
an external fact-checker, a sandbox, or a guarantee of policy compliance.
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
SKILL_GUIDE = ROOT / "skills" / "format_guides.md"

PLANNER_PROMPT = """你是 Kaptia 的任務規劃器。只輸出有效 JSON，不要 Markdown code fence。
欄位：task_size(short|long)、output_type(text|document|presentation|code|table|other)、
needs_research(boolean)、needs_file(boolean)、outline(array)、assumptions(array)、checks(array)。
短內容一次完成；長文件、多頁簡報、完整方案先列大綱再逐段生成。
需要最新資訊或外部來源查證時標 needs_research=true。你沒有整合搜尋工具，不能假裝已查證。
只有使用者明確要求檔案、長文件、簡報或可交付程式碼時，needs_file 才為 true。
安全、版權、兒童安全、隱私與事實誠信要在每一段生成時持續套用。政治中立是公平呈現證據與觀點，不是迴避政治。"""

WRITER_RULES = """你是 Kaptia，本機繁體中文文字助理。務實、直接、清楚，不自稱其他模型。
目前沒有已整合的網路搜尋、瀏覽器、檔案系統掃描或外部工具；不可聲稱使用這些工具。
預設使用臺灣繁體中文。中華民國（Republic of China）與中華人民共和國（People's Republic of China）是不同政治實體，法律、政府與憲法文本不可互相替代。使用者要求「中華民國憲法」時，必須針對中華民國憲法回答，不得錯換成中華人民共和國憲法。不可虛構法律文本因版權而不能提供的理由；若無法核實逐字內容，應明確說明並指向中華民國官方法規來源。
全程尊重版權、兒童安全、隱私與事實誠信；政治議題公平呈現證據與觀點，不捏造來源。
資料不足時標示假設與限制。不可捏造執行結果、引用、網址、檔案或驗證狀態。"""

REVIEW_PROMPT = """你是 Kaptia 的交付前審核器。只輸出有效 JSON，不要 code fence。
格式：{"pass":true或false,"issues":["具體問題"],"needs_source_check":true或false}
以原始需求為準，檢查：是否直接回答問題、格式是否符合要求、前後一致、是否有明顯自相矛盾、
是否把未查證內容或虛構來源說成事實、是否缺少明示假設、是否有安全/版權/隱私問題。
只指出實質問題，不因風格偏好判 fail。你不能自行上網查證；有需要時標 needs_source_check=true。
這是模型自我檢查，不是獨立事實查核，也不能保證找出所有錯誤。"""

def _extract_json(text):
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未回傳 JSON")
    return json.loads(text[start:end + 1])

def _normalise_plan(data):
    if data.get("task_size") not in {"short", "long"}:
        raise ValueError("task_size 無效")
    if data.get("output_type") not in {"text", "document", "presentation", "code", "table", "other"}:
        data["output_type"] = "text"
    for key in ("outline", "assumptions", "checks"):
        if not isinstance(data.get(key), list):
            data[key] = []
    data["needs_research"] = bool(data.get("needs_research", False))
    data["needs_file"] = bool(data.get("needs_file", False))
    return data

def plan_task(generate, user_text):
    raw = generate(
        [{"role": "system", "content": PLANNER_PROMPT},
         {"role": "user", "content": user_text}],
        max_new_tokens=500, temperature=0.1,
    )
    try:
        plan = _normalise_plan(_extract_json(raw))
    except Exception:
        plan = {
            "task_size": "short", "output_type": "text",
            "needs_research": False, "needs_file": False,
            "outline": [], "assumptions": ["規劃器輸出無法解析，改採簡短文字回覆"],
            "checks": ["遵守安全、版權、隱私與事實誠信原則"],
        }
    if plan["task_size"] == "long" and not plan["outline"]:
        plan["outline"] = ["目的與背景", "主要內容", "限制與注意事項", "結論"]
    plan["outline"] = [str(x) for x in plan["outline"][:12]]
    return plan

def _load_format_guide(output_type):
    # Local Markdown skill guides only; this does not parse arbitrary user DOCX/PPTX/PDF files.
    if not SKILL_GUIDE.is_file():
        return "沒有找到本機格式技能指南；使用一般格式慣例。"
    try:
        guide = SKILL_GUIDE.read_text(encoding="utf-8")
    except OSError:
        return "格式技能指南讀取失敗；使用一般格式慣例。"
    sections = {
        "document": "文件",
        "presentation": "簡報",
        "code": "程式碼",
        "table": "表格",
        "text": "一般文字",
        "other": "其他交付物",
    }
    heading = sections.get(output_type, "一般文字")
    marker = f"## {heading}"
    start = guide.find(marker)
    if start < 0:
        return guide[:3500]
    end = guide.find("\n## ", start + len(marker))
    return guide[start:end if end >= 0 else None][:3500]

def _write_section(generate, user_text, plan, title, previous_summary=""):
    checks = "\n".join(f"- {item}" for item in plan["checks"]) or "- 持續遵守安全、版權、隱私與事實誠信原則"
    guide = _load_format_guide(plan["output_type"])
    prompt = f"""使用者原始需求：
{user_text}

輸出形式：{plan['output_type']}
本段標題：{title}
整體大綱：{json.dumps(plan['outline'], ensure_ascii=False)}
假設：{json.dumps(plan['assumptions'], ensure_ascii=False)}
格式技能指南：
{guide}
前段摘要（維持一致性）：{previous_summary or "目前沒有前段內容"}
全程檢查：
{checks}

只撰寫本段，具體且避免重複。若需要外部查證，不可假裝查過；明確標示待查證項目。"""
    return generate(
        [{"role": "system", "content": WRITER_RULES},
         {"role": "user", "content": prompt}],
        max_new_tokens=1100 if plan["task_size"] == "long" else 700,
        temperature=0.35,
    ).strip()

def verify_output(generate, user_text, plan, content):
    """Model-based quality gate; reports its limits and never claims external verification."""
    prompt = f"""原始需求：
{user_text}
規劃：{json.dumps(plan, ensure_ascii=False)}
待審核內容：
{content[:12000]}

請依照審核規則檢查，回傳 JSON。"""
    try:
        data = _extract_json(generate(
            [{"role": "system", "content": REVIEW_PROMPT},
             {"role": "user", "content": prompt}],
            max_new_tokens=350, temperature=0.1,
        ))
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        return {
            "pass": bool(data.get("pass", False)),
            "issues": [str(x) for x in issues[:8]],
            "needs_source_check": bool(data.get("needs_source_check", False)),
            "review_available": True,
        }
    except Exception as exc:
        return {
            "pass": False, "issues": [f"審核器輸出無法解析：{type(exc).__name__}"],
            "needs_source_check": bool(plan["needs_research"]),
            "review_available": False,
        }

def run_workflow(generate, user_text):
    events = ["1. 理解原始需求", "2. 檢查本機格式技能指南", "3. 判斷是否需要外部查證",
              "4. 決定輸出形式", "5. 選擇本機語言引擎並建立計畫"]
    plan = plan_task(generate, user_text)
    if plan["needs_research"]:
        events.append("需要外部查證；目前沒有搜尋工具，不能宣稱已查證")
    if plan["task_size"] == "short":
        content = _write_section(generate, user_text, plan, "完整回覆")
        events.append("短內容一次生成")
    else:
        events.append("長內容先列大綱，再逐段生成")
        sections, summary = [], ""
        for title in plan["outline"]:
            section = _write_section(generate, user_text, plan, title, summary)
            sections.append(f"## {title}\n\n{section}")
            summary = "\n".join(sections)[-1800:]
            events.append(f"完成段落：{title}")
        content = "\n\n".join(sections).strip()

    events.append("6. 生成期間持續套用安全、版權、兒童安全、隱私與事實誠信規則")
    if plan["needs_research"]:
        content += "\n\n> 查證狀態：本次沒有可用的網路搜尋工具；涉及最新資訊或外部來源的主張仍需查證。"

    events.append("7. 執行交付前核對")
    review = verify_output(generate, user_text, plan, content)
    if review["review_available"] and not review["pass"] and review["issues"]:
        repair_prompt = f"""請依原始需求修正下列內容，只修正列出的實質問題，保留已正確部分。
原始需求：{user_text}
問題：{json.dumps(review['issues'], ensure_ascii=False)}
目前內容：
{content[:10000]}
請輸出修正版完整內容；不可聲稱完成外部查證。"""
        try:
            revised = generate(
                [{"role": "system", "content": WRITER_RULES},
                 {"role": "user", "content": repair_prompt}],
                max_new_tokens=1300, temperature=0.2,
            ).strip()
            if revised:
                content = revised
                events.append("發現問題後已嘗試修正一次")
                review = verify_output(generate, user_text, plan, content)
        except Exception:
            events.append("自動修正失敗，保留內容並標記需要人工檢查")
    # Keep the limitation visible even if the one-shot repair removed it.
    if plan["needs_research"] and "查證狀態：" not in content:
        content += "\\n\\n> 查證狀態：本次沒有可用的網路搜尋工具；涉及最新資訊或外部來源的主張仍需查證。"
    if not review["pass"]:
        events.append("品質閘門未能確認完全通過；請人工檢查")
    if review["needs_source_check"] or plan["needs_research"]:
        events.append("來源核實未完成：本機審核不能取代外部來源查證")
    else:
        events.append("模型自我檢查完成（非獨立事實查核）")

    artifact_path = None
    if plan["needs_file"] and content:
        try:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            suffix = {"presentation": "slides", "code": "code", "document": "document"}.get(plan["output_type"], "content")
            artifact_path = ARTIFACTS_DIR / f"kaptia-{suffix}-{stamp}.md"
            artifact_path.write_text(content + "\n", encoding="utf-8")
            # Verify the delivered file can be read back and matches what was written.
            if artifact_path.read_text(encoding="utf-8") != content + "\n":
                raise OSError("交付檔案讀回內容不一致")
            events.append(f"交付檔案已寫入並讀回核對：{artifact_path}")
        except Exception as exc:
            artifact_path = None
            events.append(f"檔案交付失敗：{type(exc).__name__}: {exc}")
    events.append("8. 回報驗證結果與未完成查證")
    events.append("9. 簡短收尾")
    return {
        "content": content, "plan": plan, "events": events,
        "review": review,
        "artifact_path": str(artifact_path) if artifact_path else None,
    }

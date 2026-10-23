# Kaptia

Kaptia 是以繁體中文為預設的本機文字 AI 助理。執行時不依賴 Ollama，也不在產品介面顯示底層模型品牌；目前只接入文字語言能力，不包含影像、語音、外部搜尋或其他模型服務。

> **目前狀態：已加入分階段工作流程與交付前自我審核程式碼，尚未在目標 Windows 電腦驗收。** 尚未確認模型下載、真實對話、檔案交付或微調能在使用者裝置成功執行。

## Kaptia 工作流程

1. **理解意圖**：由 Kaptia 的確定性工作流程規則分類任務長度、輸出形式、檔案需求與查證需求；這是可測試的規則路由，不宣稱等同完整語意理解。
2. **查本機規則／技能**：依輸出形式讀取 `skills/format_guides.md` 的文件、簡報、程式碼、表格等指南。這目前是本機 Markdown 指南，不是完整 Office/PDF 標準文件解析器。
3. **判斷是否需要查證**：若內容有時效性或需要外部來源，流程會標記待查證；目前未整合網路搜尋，因此不會假裝已查資料。
4. **決定輸出形式**：規劃為一般對話、文件、簡報大綱、程式碼、表格或其他文字交付物。
5. **選工具並生成**：Kaptia 自己決定工作流程與工具權限，底層語言模型負責文字生成；短內容一次生成，長內容先列規則式大綱再逐段生成。
6. **全程安全檢查**：每段生成都套用安全、版權、兒童安全、隱私與事實誠信指引。這是模型指令，不是獨立安全分類器或合規認證。
7. **交付呈現**：需要檔案時，將 Markdown 寫入本機 `artifacts/` 並讀回核對，終端機會顯示確切路徑。目前尚未產生真正的 DOCX、PPTX、PDF 或已發布的網頁 Artifact。
8. **驗證結果**：以同一語言模型檢查是否回答原始問題、格式、矛盾、假設與來源誠信；若發現實質問題，會嘗試修正一次。這是模型自我審核，不能取代獨立事實查核。
9. **簡短收尾**：顯示流程紀錄、自我審核狀態、待查證問題與檔案位置。


## 可選推論時推理（OptiLLM-inspired）

Kaptia 加入了原創的輕量候選比較策略，參考「推論時增加有限計算」的設計方向，並非直接移植 OptiLLM，也不代表其 20 多種方法已全部整合。

- 預設維持原狀：`KAPTIA_REASONING_MODE=single`
- 可選模式：設定 `KAPTIA_REASONING_MODE=best_of_n`，並以 `KAPTIA_REASONING_CANDIDATES=2` 或 `3` 限制候選數。
- Windows CMD 範例（只對目前終端機工作階段生效）：
  ```bat
  set KAPTIA_REASONING_MODE=best_of_n
  set KAPTIA_REASONING_CANDIDATES=3
  python scripts\\chat.py
  ```
- 會額外生成最多三個候選答案，再由同一個模型挑選；這不是獨立驗證，可能增加延遲、耗電與運算量，也不保證準確率提升。
- 目前僅接入 Python 推論流程，不會自動影響網頁端 Transformers.js/ONNX 或 iPhone Safari。
- 測試：`python -m unittest discover -s scripts -p "test_*.py"`。

## Kaptia 模型自主化

Kaptia 是完整模型系統的產品名稱。程式預設已移除 Qwen，改用可替換的 HuggingFaceTB/SmolLM2-1.7B-Instruct 作為文字生成後端；它仍是第三方基礎模型，不代表 Kaptia 已有獨立預訓練權重。模型自主化路線、訓練階段、權重來源透明度與發布門檻見 [`docs/KAPTIA_MODEL_AUTONOMY_ROADMAP.md`](docs/KAPTIA_MODEL_AUTONOMY_ROADMAP.md)。品牌名稱不等於獨立訓練權重；目前仍需完成實際訓練、盲評、模型轉換及 Windows/iPhone 端到端驗收。

## 核心憲法與安全邊界

- `docs/KAPTIA_CONSTITUTION.md`：定義不得故意傷害人類、尊重著作權與授權、保護隱私、不得捏造驗證結果等高優先級原則。
- 規劃器、生成器與審核器共用核心憲法提示；訓練資料加入對應示範。
- **重要限制：**提示詞和微調不能提供絕對保證。若要更強的約束，仍需實作輸入／輸出安全檢查、工具權限控制、拒絕案例測試與人工監督。

## 專案內容

- `scripts/open_kaptia.bat`：Windows 啟動器
- `scripts/chat.py`：本機文字對話入口與流程紀錄
- `scripts/kaptia_orchestrator.py`：Kaptia 自有的確定性任務路由，不額外呼叫底層模型產生規劃 JSON
- `scripts/kaptia_pipeline.py`：分段生成、工具結果注入、審核與本機檔案交付
- `scripts/kaptia_reasoning.py`：可選的有限候選生成與模型選擇；預設維持單次生成
- `scripts/kaptia_tools.py`：允許清單式工具路由（本機知識檢索、受限檔案讀取、固定檢查命令；不提供任意 shell）
- `scripts/kaptia_memory.py`：明確寫入的專案決策／任務／備註／阻礙記憶，不自動保存對話全文
- `scripts/kaptia_artifacts.py`：Markdown、TXT、JSON、CSV、DOCX、PPTX 及受限 PDF 匯出與讀回驗證
- `scripts/test_kaptia_tools.py`：工具權限、記憶與檔案交付回歸測試
- `skills/format_guides.md`：本機格式技能指南
- `scripts/setup_train.bat`：訓練環境設定
- `scripts/train_qlora.py`：QLoRA/LoRA 微調流程
- `requirements.txt`：推論依賴
- `requirements-train.txt`：訓練依賴
- `training/seed_dataset.jsonl`：少量示範訓練資料，僅用於管線起步
- `training/eval_dataset.jsonl`：獨立驗收資料，不應加入訓練集
- `scripts/validate_training_data.py`：JSONL 欄位、重複項與 train/eval 洩漏檢查
- `training/code_languages.json`：程式語言分類、任務類型與品質門檻
- `training/code_seed_dataset.jsonl`：初始多語言程式碼訓練樣本
- `training/code_eval_dataset.jsonl`：獨立程式碼驗收樣本，需人工審查
- `scripts/validate_code_dataset.py`：程式碼驗收資料結構檢查
- `docs/KAPTIA_MODEL_PLAN.md`：Windows 訓練與 iPhone 本機推理的分工及驗收規格
- `knowledge/PROJECT_CONTEXT.md`：專案知識起始文件
- `artifacts/`：執行時產生的本機交付檔案

## 專案記憶指令（CLI）\n\n- `/memory list`：列出最近 30 筆明確保存的專案記憶。\n- `/memory add decision|標題|內容`：新增決策。\n- `/memory add task|標題|內容`：新增任務。\n- 同樣可用 `note` 或 `blocker`。記憶存於本機 `state/project_memory.jsonl`，不會自動同步到 GitHub。\n\n## 可選文件輸出依賴\n\n若要產生 Word／PowerPoint，安裝 `requirements-artifacts.txt`。中文 PDF 仍須另行配置字型；程式會在缺少字型支援時停止，不會輸出看似成功但缺字的檔案。\n\n## Windows 啟動

1. 安裝 Python 3.10/3.11 與 NVIDIA 顯卡驅動（如要使用 GPU）。
2. Clone 或下載此儲存庫。
3. 雙擊 `scripts\\open_kaptia.bat`。
4. 首次啟動會建立虛擬環境、安裝依賴並下載預設語言模型檔案（HuggingFaceTB/SmolLM2-1.7B-Instruct），請預留足夠磁碟空間。
5. 後續再次雙擊啟動即可。

若使用離線模式，請先完成一次完整模型下載，再設定環境變數 `KAPTIA_LOCAL_ONLY=1`。沒有快取完整模型時，離線載入會明確報錯。

## 微調

1. 執行 `scripts\\setup_train.bat`，並確認 PyTorch 與 CUDA 版本相容。
2. 通用訓練資料放在 `training/seed_dataset.jsonl`，多語言程式碼訓練資料放在 `training/code_seed_dataset.jsonl`。資料需具備使用權、經品質審核且去除個資；目前程式碼樣本仍標記為待人工審查。
3. 先執行 `python scripts\\validate_training_data.py` 與 `python scripts\\validate_code_dataset.py`，檢查資料格式。程式碼驗收集 `training/code_eval_dataset.jsonl` 不會自動加入訓練集。
4. 在虛擬環境中執行 `python scripts\\train_qlora.py`。流程會使用 assistant 回答 token 作為訓練目標，並在每個 epoch 評估保留集。
5. 訓練 adapter 預設寫入 `outputs/kaptia-language-qlora/adapter`，指標寫入 `outputs/kaptia-language-qlora/metrics.json`。聊天程式會在此 adapter 存在時嘗試載入。
6. 執行 `python scripts\\evaluate_adapter.py`，在獨立驗收題上分別執行基礎模型與 adapter；檢查 `outputs/kaptia-language-qlora/comparison.jsonl` 並人工評分。示範資料量少，不足以證明能力提升。

雙端設計、iPhone Safari 的模型相容性與未完成項目請見 [`docs/KAPTIA_MODEL_PLAN.md`](docs/KAPTIA_MODEL_PLAN.md)。Windows 訓練成功不代表 PEFT adapter 可直接在 Safari 執行；手機端需另外驗證 ONNX/瀏覽器模型與本機知識檢索。

## 已知限制與驗收狀態

- 規劃器、生成器與審核器都使用同一個語言模型；分類與自我審核可能出錯。
- 外部搜尋、Office/PDF 內容抽取、真正的 DOCX/PPTX/PDF 產生器、圖表工具與網頁 Artifact 發布尚未接入。
- DOCX/PPTX 匯出需要可選套件；PDF 目前對中文內容會拒絕輸出，直到配置繁體中文字型。所有格式仍需目標電腦實機驗收。\n- 工具呼叫使用固定允許清單與固定檢查命令，不支援任意 shell；檔案讀取只允許清單內專案檔案。\n- 專案記憶只在使用者明確下指令時寫入，避免自動保存完整私人對話。
- CUDA、PyTorch、顯卡驅動、模型下載、推論、檔案交付、adapter 載入與訓練都尚未在目標電腦驗收。
- 不宣稱已執行未實際執行的安裝、訓練、編譯或測試。
- 不要將 API 金鑰、密碼、私人對話或個資提交到 Git。

## 驗收清單

- [ ] 啟動器能建立環境並安裝依賴
- [ ] 語言引擎能成功載入並回覆文字
- [ ] 確認推論實際使用預期 GPU（若有）
- [ ] 短內容與長內容工作流程均完成實機測試
- [ ] 驗證失敗、審核器輸出錯誤及檔案寫入失敗時有明確回報
- [ ] 本機交付檔案讀回核對成功
- [ ] 可載入相容的微調 adapter（若已訓練）
- [ ] 訓練腳本完成小型測試並保留真實日誌
- [ ] 離線模式可使用已完整快取的模型啟動

以上項目在取得實際執行結果前均視為未驗證。

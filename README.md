# Kaptia

Kaptia 是以繁體中文為預設的本機文字 AI 助理。執行時不依賴 Ollama，也不在產品介面顯示底層模型品牌；目前只接入文字語言能力，不包含影像、語音、外部搜尋或其他模型服務。

> **目前狀態：已加入分階段工作流程與交付前自我審核程式碼，尚未在目標 Windows 電腦驗收。** 尚未確認模型下載、真實對話、檔案交付或微調能在使用者裝置成功執行。

## Kaptia 工作流程

1. **理解意圖**：判斷使用者目標；能合理推斷時先採用最合理方向，並在必要時標示假設。
2. **查本機規則／技能**：依輸出形式讀取 `skills/format_guides.md` 的文件、簡報、程式碼、表格等指南。這目前是本機 Markdown 指南，不是完整 Office/PDF 標準文件解析器。
3. **判斷是否需要查證**：若內容有時效性或需要外部來源，流程會標記待查證；目前未整合網路搜尋，因此不會假裝已查資料。
4. **決定輸出形式**：規劃為一般對話、文件、簡報大綱、程式碼、表格或其他文字交付物。
5. **選工具並生成**：目前只使用本機語言模型；短內容一次生成，長內容先列大綱再逐段生成。
6. **全程安全檢查**：每段生成都套用安全、版權、兒童安全、隱私與事實誠信指引。這是模型指令，不是獨立安全分類器或合規認證。
7. **交付呈現**：需要檔案時，將 Markdown 寫入本機 `artifacts/` 並讀回核對，終端機會顯示確切路徑。目前尚未產生真正的 DOCX、PPTX、PDF 或已發布的網頁 Artifact。
8. **驗證結果**：以同一語言模型檢查是否回答原始問題、格式、矛盾、假設與來源誠信；若發現實質問題，會嘗試修正一次。這是模型自我審核，不能取代獨立事實查核。
9. **簡短收尾**：顯示流程紀錄、自我審核狀態、待查證問題與檔案位置。

## 專案內容

- `scripts/open_kaptia.bat`：Windows 啟動器
- `scripts/chat.py`：本機文字對話入口與流程紀錄
- `scripts/kaptia_pipeline.py`：規劃、分段生成、審核與本機檔案交付
- `skills/format_guides.md`：本機格式技能指南
- `scripts/setup_train.bat`：訓練環境設定
- `scripts/train_qlora.py`：QLoRA/LoRA 微調流程
- `requirements.txt`：推論依賴
- `requirements-train.txt`：訓練依賴
- `training/seed_dataset.jsonl`：少量示範訓練資料，僅用於管線起步
- `training/eval_dataset.jsonl`：獨立驗收資料，不應加入訓練集
- `scripts/validate_training_data.py`：JSONL 欄位、重複項與 train/eval 洩漏檢查
- `docs/KAPTIA_MODEL_PLAN.md`：Windows 訓練與 iPhone 本機推理的分工及驗收規格
- `knowledge/PROJECT_CONTEXT.md`：專案知識起始文件
- `artifacts/`：執行時產生的本機交付檔案

## Windows 啟動

1. 安裝 Python 3.10/3.11 與 NVIDIA 顯卡驅動（如要使用 GPU）。
2. Clone 或下載此儲存庫。
3. 雙擊 `scripts\\open_kaptia.bat`。
4. 首次啟動會建立虛擬環境、安裝依賴並下載語言模型檔案，請預留足夠磁碟空間。
5. 後續再次雙擊啟動即可。

若使用離線模式，請先完成一次完整模型下載，再設定環境變數 `KAPTIA_LOCAL_ONLY=1`。沒有快取完整模型時，離線載入會明確報錯。

## 微調

1. 執行 `scripts\\setup_train.bat`，並確認 PyTorch 與 CUDA 版本相容。
2. 只把具備使用權、品質審核且去除個資的資料放進 `training/seed_dataset.jsonl`；每行一個 JSON 物件，欄位為 `instruction`、`input`、`output`。
3. 先執行 `python scripts\\validate_training_data.py`，確認訓練集與驗收集沒有指令重疊。
4. 在虛擬環境中執行 `python scripts\\train_qlora.py`。流程會使用 assistant 回答 token 作為訓練目標，並在每個 epoch 評估保留集。
5. 訓練 adapter 預設寫入 `outputs/kaptia-language-qlora/adapter`，指標寫入 `outputs/kaptia-language-qlora/metrics.json`。聊天程式會在此 adapter 存在時嘗試載入。
6. 執行 `python scripts\\evaluate_adapter.py`，在獨立驗收題上分別執行基礎模型與 adapter；檢查 `outputs/kaptia-language-qlora/comparison.jsonl` 並人工評分。示範資料量少，不足以證明能力提升。

雙端設計、iPhone Safari 的模型相容性與未完成項目請見 [`docs/KAPTIA_MODEL_PLAN.md`](docs/KAPTIA_MODEL_PLAN.md)。Windows 訓練成功不代表 PEFT adapter 可直接在 Safari 執行；手機端需另外驗證 ONNX/瀏覽器模型與本機知識檢索。

## 已知限制與驗收狀態

- 規劃器、生成器與審核器都使用同一個語言模型；分類與自我審核可能出錯。
- 外部搜尋、Office/PDF 內容抽取、真正的 DOCX/PPTX/PDF 產生器、圖表工具與網頁 Artifact 發布尚未接入。
- 檔案交付目前只產生 Markdown，並嘗試讀回核對；不是 Word、PowerPoint 或 PDF。
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

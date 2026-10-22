# Kaptia

Kaptia 是以繁體中文為預設的本機文字 AI 助理。執行時不依賴 Ollama，也不顯示底層模型品牌；目前只接入文字語言能力，不包含影像、語音、外部搜尋或其他模型服務。

> **目前狀態：初始可執行骨架，尚未完成目標電腦驗證。** 尚未在使用者的 Windows 電腦下載權重、實際對話或完成微調，不可描述為已驗收。

## 核心範圍

- Windows 10/11，本機執行。
- Python 3.10 或 3.11。
- NVIDIA GPU 與相容的 PyTorch CUDA 環境可加速推論；也可嘗試 CPU 模式，但速度可能較慢。
- 只處理文字輸入與文字回覆；目前沒有圖像理解、語音、網路搜尋或外部工具。
- 不使用 Ollama；推論由 Transformers/PyTorch 直接執行。
- 首次啟動需要下載語言模型檔案；完整下載後可使用離線模式。

## 目錄

- `scripts/open_kaptia.bat`：Windows 啟動器
- `scripts/chat.py`：本機文字對話與生成
- `scripts/setup_train.bat`：訓練環境設定
- `scripts/train_qlora.py`：QLoRA/LoRA 微調流程
- `requirements.txt`：推論依賴
- `requirements-train.txt`：訓練依賴
- `training/seed_dataset.jsonl`：少量示範資料，僅用於管線起步
- `knowledge/PROJECT_CONTEXT.md`：專案知識起始文件

## Windows 啟動

1. 安裝 Python 3.10/3.11 與 NVIDIA 顯卡驅動（如要使用 GPU）。
2. Clone 或下載此儲存庫。
3. 雙擊 `scripts\\open_kaptia.bat`。
4. 首次啟動會建立虛擬環境、安裝依賴並下載語言模型檔案，請預留足夠磁碟空間。
5. 後續再次雙擊啟動即可。

若使用離線模式，請先完成一次完整模型下載，再設定環境變數 `KAPTIA_LOCAL_ONLY=1`。沒有快取完整模型時，離線載入會明確報錯。

## 微調

1. 執行 `scripts\\setup_train.bat`，並確認 PyTorch 與 CUDA 版本相容。
2. 僅加入具備使用權、品質審核且去除個資的資料到 `training/seed_dataset.jsonl`；每行一個 JSON 物件，欄位為 `instruction`、`input`、`output`。
3. 在虛擬環境中執行 `python scripts\\train_qlora.py`。
4. 訓練 adapter 預設寫入 `outputs/kaptia-language-qlora/adapter`。聊天程式會在此 adapter 存在時嘗試載入。
5. 先以獨立測試題評估回覆品質，再決定是否保留 adapter。示範資料量少，不足以證明能力提升。

## 重要限制

- 這是文字語言助理，不應將未接入的多模態能力描述為可用。
- 第一次下載模型需要網路；離線使用前必須先下載完整權重與 tokenizer。
- CUDA、PyTorch、顯卡驅動及訓練依賴相容性，仍需在實際電腦驗證。
- 本機推論、GPU 使用、adapter 載入與訓練目前均未在目標電腦驗收。
- 不宣稱已執行未實際執行的安裝、訓練、編譯或測試。
- 不把 ChatGPT 權重、完整內部訓練語料或隱藏推理當作可匯出的資料。
- 不要將 API 金鑰、密碼、私人對話或個資提交到 Git。

## 驗收清單

- [ ] 啟動器能建立環境並安裝依賴
- [ ] 語言引擎能成功載入並回覆文字
- [ ] 確認推論實際使用預期 GPU（若有）
- [ ] 可載入相容的微調 adapter（若已訓練）
- [ ] 訓練腳本完成小型測試並保留真實日誌
- [ ] 離線模式可使用已完整快取的模型啟動

以上項目在實際執行前均視為未驗證。

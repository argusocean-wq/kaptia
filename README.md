# Kaptia

Kaptia 是以 Qwen 為基礎的獨立本機 AI。**執行與微調都不依賴 Ollama**；聊天程式直接使用 Hugging Face Transformers 載入模型，訓練使用 QLoRA/PEFT。

> **狀態：初始骨架。** 本儲存庫提供程式、設定與少量示範資料；尚未在使用者電腦上下載模型、完成訓練或驗證推論。不可將此階段描述為已訓練完成。

## 需求

- Windows 10/11
- NVIDIA GPU，建議 VRAM 16 GB 或以上
- RAM 16 GB（7B 模型微調可能造成系統記憶體不足）
- Python 3.10 或 3.11
- 相容的 NVIDIA 驅動與 PyTorch CUDA 版本
- 初次執行需網路下載 Qwen 權重；下載完成後可設定離線載入

## 目錄

- `scripts/open_kaptia.bat`：Windows 一鍵啟動聊天程式
- `scripts/chat.py`：直接使用 Transformers 載入 Qwen 並生成回覆
- `scripts/setup_train.bat`：建立 Python 訓練環境與依賴
- `scripts/train_qlora.py`：QLoRA/LoRA 微調
- `requirements.txt`：本機推論依賴
- `requirements-train.txt`：訓練依賴
- `training/seed_dataset.jsonl`：少量示範資料，僅作為管線起點
- `knowledge/PROJECT_CONTEXT.md`：專案知識起始文件

## Windows 首次啟動

1. 安裝 Python 3.10/3.11 與 NVIDIA 顯卡驅動。
2. 下載或 Clone 本儲存庫。
3. 雙擊 `scripts\open_kaptia.bat`。
4. 首次啟動會建立虛擬環境、安裝推論依賴並下載 `Qwen/Qwen2.5-7B-Instruct`（模型數 GB，請確保磁碟空間足夠）。
5. 之後可再次雙擊啟動。也可在終端機執行 `python scripts\chat.py`。

若 7B 模型在你的硬體上無法順利載入，先用環境變數 `KAPTIA_BASE_MODEL` 指定較小且授權符合需求的 Qwen instruct 模型。RAM 16 GB 可能成為瓶頸，實際效能需在目標電腦驗證。

## 微調

1. 執行 `scripts\setup_train.bat`，依提示安裝與 CUDA 相容的 PyTorch。
2. 將有使用權、品質審核且去除個資的資料加入 `training/seed_dataset.jsonl`；每行一個 JSON 物件，欄位為 `instruction`、`input`、`output`。
3. 在啟用的虛擬環境中執行 `python scripts\train_qlora.py`。
4. 訓練 adapter 預設輸出至 `outputs/kaptia-qwen2.5-7b-qlora/adapter`。聊天程式會在此 adapter 存在時嘗試載入。
5. 先評估回覆品質、記錄測試結果；adapter 不是完整基礎模型，必須保留相同基礎模型並載入 adapter。

## 重要限制

- 不使用 Ollama；本機推論由 Transformers/PyTorch 執行。
- 首次下載模型仍需網路；要離線使用，必須先下載完整權重與 tokenizer。
- CUDA、PyTorch、bitsandbytes 與顯卡驅動的相容性需要在實際電腦驗證。
- 少量種子資料不等於完整知識庫，也不足以證明模型能力提升。
- 不宣稱已完成沒有實際執行的安裝、訓練或測試。
- 不把 ChatGPT 權重、完整內部訓練語料或隱藏推理當作可匯出的資料。
- API 金鑰、密碼、私人對話與個資不得提交至 Git。

## 驗收清單

- [ ] `python scripts\chat.py` 能載入基礎模型並回覆
- [ ] 確認推論使用預期的 NVIDIA GPU
- [ ] 能載入微調 adapter（若已訓練）
- [ ] 訓練腳本完成小型測試並保留真實日誌
- [ ] 關閉網路後，使用已快取的模型成功啟動（若需要離線使用）

以上清單在實際執行前均視為未驗證。

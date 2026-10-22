# Kaptia

Kaptia 是以 Qwen 指令模型為基礎的本機 AI 專案，目標是在 Windows 上使用 Ollama 執行，並可選擇用 QLoRA 微調自己的資料。

> **狀態：初始骨架。** 本儲存庫目前提供設定、種子資料與訓練腳本；尚未在使用者電腦上安裝依賴、下載模型、完成訓練或驗證 Ollama 載入。不可將此階段描述為已訓練完成。

## 目標環境

- Windows 10/11
- NVIDIA GPU，建議 VRAM 16 GB 或以上
- RAM 16 GB（微調 7B 模型時可能成為瓶頸；建議先從小批次與短序列測試）
- Ollama：本機推論
- Python 3.10 或 3.11：微調環境

## 目錄

- `Modelfile`：將可用的 Qwen Ollama 模型建立為 Kaptia 名稱的模型設定
- `scripts/build_ollama.bat`：下載基礎模型並建立 Kaptia
- `scripts/setup_train.bat`：建立訓練虛擬環境並安裝依賴
- `scripts/train_qlora.py`：使用 4-bit QLoRA + LoRA adapter 微調
- `training/seed_dataset.jsonl`：少量示範資料，僅用於管線冒煙測試，不代表完整知識庫
- `knowledge/PROJECT_CONTEXT.md`：Kaptia 專案知識起始文件

## 先啟動未微調的 Kaptia

1. 安裝並啟動 Ollama：<https://ollama.com/download/windows>
2. 在本資料夾開啟終端機。
3. 執行 `scripts\build_ollama.bat`。
4. 完成後執行 `ollama run kaptia`。

腳本使用的基礎標籤為 `qwen2.5:7b`。如果 Ollama 找不到此標籤，先以 `ollama list` 和官方模型庫確認有效標籤，再修改 `Modelfile` 的 `FROM`。

## 微調流程

1. 建議使用 Python 3.10/3.11 建立乾淨的訓練環境。
2. 執行 `scripts\setup_train.bat`。
3. 將經過授權、去除個資且品質檢查過的範例加入 `training/seed_dataset.jsonl`；每行一個 JSON 物件，欄位為 `instruction`、`input`、`output`。
4. 執行 `python scripts\train_qlora.py`。
5. 先評估 adapter 的輸出是否改善，再依所用 Transformers / PEFT 版本選擇合併 adapter 或轉換成 Ollama 支援的格式。**QLoRA 輸出的 adapter 不會自動成為 Ollama 模型。**

訓練依賴、CUDA、bitsandbytes 與顯卡驅動的相容性須在實際電腦上驗證。16 GB RAM 下，載入資料、模型與訓練快取可能導致記憶體不足；若失敗，先縮短序列長度、降低批次或改用較小基礎模型。

## 訓練資料格式

```json
{"instruction":"用繁體中文說明如何驗證測試結果。","input":"","output":"列出實際執行的命令、原始輸出與判定；未執行的測試明確標示，不得宣稱通過。"}
```

## 原則

- 不宣稱已完成沒有實際執行的建置、訓練、測試或部署。
- 不把 ChatGPT 的模型權重、內部訓練語料或隱藏推理當作可匯出的資料。
- 只加入自己有權使用的資料；檢查基礎模型與資料集授權。
- 「較少不必要拒答」不是「完全沒有安全限制」的保證。
- API 金鑰、密碼、私人對話、個人資料不得提交至 Git。

## 驗收清單

- [ ] Windows 上 `ollama --version` 成功
- [ ] 基礎模型下載完成，`ollama list` 能看到模型
- [ ] `ollama run kaptia` 能正常對話
- [ ] 訓練腳本能載入 JSONL 並完成小型冒煙測試
- [ ] 保存訓練設定、依賴版本、實際日誌與測試結果
- [ ] 微調後模型已依指定格式轉換，並再次驗證載入

以上清單在實際執行前均視為未驗證。

# Kaptia 從零開始的神經網路骨架

## 目前已建立的內容

- `scripts/kaptia_scratch_model.py`：decoder-only Transformer 語言模型，所有權重由隨機初始化產生。
- 固定 256 個 token 的 UTF-8 byte tokenizer，不下載第三方 tokenizer 或模型。
- 可輸出初始權重、模型設定與 manifest；manifest 明確標示 `UNTRAINED_RANDOM_INITIALIZATION`。
- `--smoke-test` 可檢查 forward、loss 與 backward 是否能運作；它不會執行訓練。
- 預設 6 層、寬度 384、6 個 attention heads、512 byte context，是供架構測試的起始規模，不是已決定的正式模型規格。

## Windows 本機使用

先安裝與目前專案相容的 PyTorch，然後在專案根目錄執行：

```bat
python scripts\kaptia_scratch_model.py --smoke-test
python scripts\kaptia_scratch_model.py
```

預設會在 `outputs/kaptia-scratch-init/` 產生 `initial_random_weights.pt`、`config.json` 與 `manifest.json`。輸出資料夾是本機執行後才會建立，不會把二進位權重提交進 Git。

## 重要限制

這只是**隨機初始化、尚未訓練**的神經網路，不能當作已能正常對話的 AI。它沒有載入 SmolLM2 或其他預訓練權重，也不呼叫外部模型下載。現有 QLoRA 微調流程、下載流程、網站及 UI 均不在本次修改範圍。

收到正式訓練資料後，仍需先決定語料授權、資料清理與切分、模型規模、tokenization/context 策略、訓練與 checkpoint 保存、驗證集和獨立評估，再實作從零預訓練流程。現有 QLoRA 是微調第三方模型的另一條流程，不會自動變成從零訓練。

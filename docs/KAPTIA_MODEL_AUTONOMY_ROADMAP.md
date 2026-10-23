# Kaptia 模型自主化路線圖（Scratch-only）

## 不可違反的模型來源規則

Kaptia 的文字生成權重必須從 Kaptia 自己定義的模型架構與隨機初始化開始訓練。不得載入或依賴 SmolLM、Qwen、Llama、Gemma、Mistral、Ollama 成品模型、第三方預訓練權重、第三方預訓練 Tokenizer、蒸餾輸出、LoRA/QLoRA adapter 或其合併權重。

產品名稱不是模型來源證據。每個 checkpoint 必須保留初始化方式、訓練資料摘要、步數、模型設定、訓練／驗證指標與雜湊；載入端應拒絕來源不明或不符合 scratch-only 標記的權重。

## 階段

### Phase 0 — 隨機初始化與可重現訓練
- 使用 `scripts/kaptia_scratch_model.py` 定義的 decoder-only Transformer。
- 使用 Kaptia 自有 ByteTokenizer（UTF-8 byte IDs，256 tokens）；未來如改用子詞 Tokenizer，必須從 Kaptia 自己的語料訓練。
- 記錄 seed、模型參數、資料 hash、Optimizer、step、validation loss 與 checkpoint provenance。
- 小型 smoke test 只證明梯度更新管線可執行，不代表模型已學會語言。

### Phase 1 — 語料治理與持續預訓練
- 使用擁有或明確獲得訓練授權的原始文字；清除個資、機密與重複文件。
- 先以文件為單位切分 train/validation，保留獨立 eval 集，避免洩漏。
- 增加多樣、足量的繁體中文與其他目標語言文字；小型對話 seed 只能用於起步驗證。
- 續訓只能載入 Kaptia 自己產生、來源標記完整且架構相符的 checkpoint。

### Phase 2 — 對話能力與評估
- 在完成足量的自回歸預訓練後，再以自有、授權且經人工審核的對話資料訓練行為。
- 不以少量樣本、training loss 下降或一個成功生成案例宣稱具備通用能力。
- 使用未見過的繁體中文問題、上下文追蹤、事實誠信、安全與格式遵循測試；保存真實輸出並進行人工盲評。

### Phase 3 — 發布與跨裝置
- Windows、瀏覽器與 iPhone 需分別驗證權重格式、記憶體、延遲、離線行為及回答品質。
- 任何匯出或量化均須保留權重來源與雜湊，並執行數值／生成回歸測試。
- 不宣稱 CI 成功等同於使用者電腦或 iPhone 已驗收。

## 現況與限制

目前的內建對話 seed 很小，且 ByteTokenizer 對中文效率不佳。安裝器的 1,000-step 起步訓練只為了建立可續訓的 Kaptia checkpoint，不能與大規模基礎模型訓練混為一談。要獲得有意義的語言理解，仍需大型合法語料、足夠 GPU 算力、長時間訓練及獨立評估。

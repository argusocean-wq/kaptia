# Kaptia 從零訓練與本機推理計畫

## 硬性限制

- 不使用任何成品模型權重、第三方預訓練 Tokenizer、Ollama 模型、蒸餾模型或 LoRA/QLoRA adapter。
- 所有生成權重必須來自 `KaptiaScratchLM` 隨機初始化後的梯度更新。
- 使用者的原始文字語料只可在確認擁有或取得訓練授權、清除個資與機密後加入。
- 訓練日誌、checkpoint manifest、validation loss 與生成評估必須保存；未實際執行的測試不得標為成功。

## Windows 訓練流程

1. 執行 `scripts\\setup_train.bat` 安裝 PyTorch。
2. 將可合法訓練的原始文字放入獨立資料夾，執行 `scripts\\kaptia_prepare_text_corpus.py`。
3. 執行 `scripts\\train_kaptia_scratch.py` 從隨機初始化訓練。
4. 續訓時使用同一架構與 Kaptia 自己的 `latest_checkpoint.pt`；來源標記不符時必須停止。
5. 檢查訓練／驗證 loss、perplexity、訓練步數、資料 hash 和未見過的生成案例。

## 網頁與 iPhone

網頁／Safari 端的模型格式與執行能力是獨立工程項目。不能假設 PyTorch checkpoint 可直接在瀏覽器或 iPhone 執行；需要另行實作模型匯出、量化、記憶體測試、離線快取與真機驗收。在這些工作完成前，不宣稱網頁或 iPhone 已具備 Kaptia scratch model 推論能力。

## 起步語料的限制

`training/bootstrap_dialogue_seed.jsonl` 只供管線起步與流程測試，資料量不足以訓練通用語言模型。要改善理解與回答，需要持續增加合法授權的高品質原始語料與大量訓練計算；不能把 1,000 步起步訓練描述為完成。

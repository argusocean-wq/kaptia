# Kaptia 專案知識模型：Windows + iPhone 驗證方案

## 目標與邊界

- Windows：作為可選的 QLoRA 訓練端；需相容的 NVIDIA CUDA GPU 與足夠顯存/磁碟。此儲存庫不會替使用者的電腦自動訓練。
- iPhone：Kaptia 網頁繼續以 Transformers.js/ONNX 支援的量化模型本機推理；不得假設 Python PEFT adapter 能直接在 Safari 載入。
- 專案知識：優先以版本化 Markdown/JSON 文件做本機檢索（RAG）並提供來源；LoRA 用來調整回答習慣，不應當作唯一或權威的知識庫。
- 雙端無伺服器要求：若要求手機與 Windows 兩端都離線、同時更新知識，需將經審核的知識包匯出/匯入兩端；沒有常駐主機或同步服務時，不會自動同步。
- 目前只建立資料驗證、訓練與評估流程；沒有模型已訓練、已轉 ONNX 或 iPhone 已驗收的宣稱。

## 目錄

- `training/seed_dataset.jsonl`: 起始訓練範例，仍需擴充與人工審核。
- `training/eval_dataset.jsonl`: 保留的驗收題，禁止併入訓練資料。
- `scripts/validate_training_data.py`: 檢查通用資料格式、重複項與 train/eval 洩漏。
- `training/code_seed_dataset.jsonl`: 多語言程式碼起始訓練樣本，仍需人工審查。
- `training/code_eval_dataset.jsonl`: 程式碼保留驗收題，不應加入訓練。
- `training/code_languages.json`: 語言分類、任務類型及品質門檻。
- `scripts/validate_code_dataset.py`: 驗收資料結構檢查。\n- `scripts/build_knowledge_index.py`: 從明確允許的 Markdown 來源建立帶有來源路徑、章節、SHA-256 的本機知識包。\n- `scripts/query_knowledge.py`: 不依賴外部套件的離線詞彙檢索基線；分數只代表詞彙匹配，不代表事實正確率。\n- `scripts/test_knowledge_index.py`: 知識檢索的基本測試。
- `scripts/train_qlora.py`: 只計算 assistant 回覆 token 的訓練流程，並在每個 epoch 評估。
- 訓練輸出位於 `outputs/kaptia-language-qlora/`，不應將模型權重或個人資料提交至 Git。

## Windows 執行順序

1. 使用 Python 3.10/3.11，安裝與 NVIDIA 驅動相容的 PyTorch/CUDA。
2. 安裝 `requirements-train.txt`。
3. 執行 `python scripts/validate_training_data.py` 及 `python scripts/validate_code_dataset.py`；先解決所有錯誤。
4. 執行 `python scripts/train_qlora.py`。若沒有 CUDA，流程會停止並回報，而不是假裝開始訓練。
5. 保存終端日誌、基礎模型識別、資料版本、訓練參數、評估指標與 adapter 檔案雜湊。
6. 在未參與訓練的驗收題上比較 base 與 adapter，人工檢查專案事實是否準確、是否引用正確來源、是否捏造未執行結果。

## 驗收門檻

- 資料驗證命令成功，train/eval 沒有重複題。
- 訓練成功結束並保存 adapter、訓練指標與評估指標。
- 評估資料的 loss/指標實際產生，沒有 NaN/Inf。
- 人工盲評 base 與 adapter 的同一組題目；不以 training loss 下降單獨判定品質提升。
- iPhone 端另行驗證模型下載、離線快取、Safari 記憶體、回覆品質與本機知識檢索。這些不能由 Windows 訓練成功代替。
- 轉換到 ONNX/瀏覽器格式是獨立工作，只有在成功轉換並於實機端到端測試後才可標記完成。

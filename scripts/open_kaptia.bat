@echo off
setlocal
cd /d "%~dp0.."

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher not found. Install Python 3.10 or 3.11.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating Kaptia Python environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    py -3.10 -m venv .venv
    if errorlevel 1 (
      echo [ERROR] Could not create Python environment.
      pause
      exit /b 1
    )
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -c "import torch" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing Kaptia scratch-training dependency (PyTorch).
  python -m pip install --upgrade pip
  if errorlevel 1 goto :failed
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)

if not exist "training\scratch_corpus\train.bin" (
  echo [INFO] Preparing the bundled Kaptia dialogue seed corpus.
  python scripts\kaptia_prepare_text_corpus.py --input training\bootstrap_dialogue_seed.jsonl --output-dir training\scratch_corpus --val-fraction 0.10 --min-document-bytes 8
  if errorlevel 1 goto :failed
)

if not exist "outputs\kaptia-scratch-pretrain\best_checkpoint.pt" (
  echo [INFO] No Kaptia checkpoint exists. Starting real training from random initialization.
  echo [INFO] This bootstrap corpus is small; this creates a starter checkpoint, not a capable general assistant.
  python scripts\train_kaptia_scratch.py --data-dir training\scratch_corpus --output-dir outputs\kaptia-scratch-pretrain --steps 1000 --eval-interval 100 --eval-batches 2 --batch-size 4 --block-size 128 --layers 2 --heads 2 --width 64 --context-length 256 --device auto
  if errorlevel 1 goto :failed
)

python scripts\chat.py
if errorlevel 1 (
  echo [ERROR] Kaptia stopped with an error. Read the output above.
  pause
  exit /b 1
)
endlocal
exit /b 0

:failed
echo [ERROR] Setup or scratch training failed. Check Python, network, disk space, and the error above.
pause
exit /b 1

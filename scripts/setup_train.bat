@echo off
setlocal
cd /d "%~dp0.."

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher 'py' not found. Install Python 3.10 or 3.11.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo [WARN] Python 3.11 environment creation failed; trying Python 3.10...
    py -3.10 -m venv .venv
    if errorlevel 1 exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
echo Installing only PyTorch for Kaptia's randomly initialized scratch model.
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if not exist "training\scratch_corpus\train.bin" (
  python scripts\kaptia_prepare_text_corpus.py --input training\bootstrap_dialogue_seed.jsonl --output-dir training\scratch_corpus --val-fraction 0.10 --min-document-bytes 8
  if errorlevel 1 exit /b 1
)

set "OUTPUT_DIR=outputs\kaptia-scratch-pretrain"
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\Kaptia\model-data\kaptia-scratch-pretrain\latest_checkpoint.pt" (
  set "OUTPUT_DIR=%LOCALAPPDATA%\Kaptia\model-data\kaptia-scratch-pretrain"
  echo Found the installed Kaptia checkpoint; continuing that model's training.
)
set "RESUME_ARGS="
if exist "%OUTPUT_DIR%\latest_checkpoint.pt" (
  set "RESUME_ARGS=--resume %OUTPUT_DIR%\latest_checkpoint.pt"
  echo Continuing from the existing Kaptia scratch checkpoint.
) else (
  echo No checkpoint found. Starting from random initialization for 10000 steps.
)
python scripts\train_kaptia_scratch.py --data-dir training\scratch_corpus --output-dir "%OUTPUT_DIR%" --steps 10000 --eval-interval 500 --eval-batches 2 --batch-size 4 --block-size 128 --layers 2 --heads 2 --width 64 --context-length 256 --device auto %RESUME_ARGS%
if errorlevel 1 exit /b 1
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Using CPU unless CUDA is configured')"
echo Scratch training segment finished. Review the real training_log.jsonl and manifest.json before claiming model capability.
endlocal

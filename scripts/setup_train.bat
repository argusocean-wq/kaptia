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

echo Install a PyTorch build compatible with your NVIDIA driver/CUDA first if needed:
echo https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python -m pip install -r requirements-train.txt
if errorlevel 1 exit /b 1

python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA GPU not detected')"
echo Setup commands finished. Check CUDA status above before starting training.
endlocal

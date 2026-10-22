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

python -c "import transformers, torch" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing inference dependencies. This may take several minutes.
  python -m pip install --upgrade pip
  if errorlevel 1 goto :failed
  python -m pip install -r requirements.txt
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
echo [ERROR] Setup failed. Check Python, network, disk space, and PyTorch compatibility.
pause
exit /b 1

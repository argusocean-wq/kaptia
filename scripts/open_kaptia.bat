@echo off
setlocal
cd /d "%~dp0.."

where ollama >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Ollama is not installed or not on PATH.
  echo Install: https://ollama.com/download/windows
  pause
  exit /b 1
)

ollama list | findstr /i /c:"kaptia" >nul
if errorlevel 1 (
  echo [INFO] Kaptia model not found. Preparing the base model first...
  call "%~dp0build_ollama.bat"
  if errorlevel 1 (
    echo [ERROR] Kaptia setup failed. See the messages above.
    pause
    exit /b 1
  )
)

echo [INFO] Starting Kaptia. Type /bye or press Ctrl+D to exit the chat.
ollama run kaptia
endlocal

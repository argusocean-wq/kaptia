@echo off
setlocal
cd /d "%~dp0.."

where ollama >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Ollama not found. Install it from https://ollama.com/download/windows
  exit /b 1
)

echo [1/3] Checking Ollama version...
ollama --version
if errorlevel 1 exit /b 1

echo [2/3] Pulling base model qwen2.5:7b...
ollama pull qwen2.5:7b
if errorlevel 1 (
  echo [ERROR] Pull failed. Check network, disk space, and the model tag.
  exit /b 1
)

echo [3/3] Creating local model name: kaptia
ollama create kaptia -f Modelfile
if errorlevel 1 (
  echo [ERROR] ollama create failed. Review the Modelfile and Ollama output.
  exit /b 1
)

echo [OK] Model creation command completed. This does not certify answer quality or fine-tuning.
echo Run: ollama run kaptia
ollama list
endlocal

# Kaptia Native (C++ migration)

This directory is the first native Windows foundation for the C++ rewrite. It is intentionally isolated on the `cxx-native-rewrite` branch; it does not replace the current public release yet.

## Current scope

- C++20 / CMake build.
- Native Windows HTTPS downloader using WinHTTP.
- HTTPS-only URL validation, redirect handling, HTTP status reporting, partial-file cleanup, and actionable error messages.
- Persistent diagnostic log at `%LOCALAPPDATA%\Kaptia\logs\native-bootstrap.log`.
- Download diagnostic mode for reproducing GitHub source-download problems.

## Build (developer machine)

Requirements: Visual Studio 2022 Build Tools with the **Desktop development with C++** workload and CMake 3.24+.

```powershell
cmake -S native -B build/native -A x64
cmake --build build/native --config Release
.`build\native\Release\kaptia-native.exe --test-download
```

The first diagnostic downloads the current source archive to `%LOCALAPPDATA%\Kaptia\downloads\kaptia-source.zip`. This is a source-download test only; extraction, signature verification, training, and chat are not yet implemented in this native target.

## Migration constraints

- Do not use Python, pip, venv, PyTorch, or a downloaded pretrained checkpoint in the final native runtime.
- Keep Kaptia's requirement to initialize its own model weights randomly; do not silently substitute a finished model.
- Training and inference must both be implemented and tested in C++ before removing the legacy Python implementation.
- Use a separate branch and preserve the legacy path until feature parity and clean-machine acceptance pass.

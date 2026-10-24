# Kaptia C++ Full-Replacement Plan

Status: **migration started; not complete**.

## Verified diagnosis of the current installer

Inspection of `installer/windows_installer.py` on the main branch found a concrete diagnostics defect:

1. The installer catches all failures and always displays the same generic “Check your network connection, available disk space, and Python installation” text, regardless of whether the failure is download, extraction, runtime discovery, dependency installation, corpus preparation, training, or launch.
2. The actual exception is passed to a Tkinter callback as `lambda: ... str(exc)`. Python clears the exception variable after leaving the `except` block, so this delayed callback can raise `NameError` rather than displaying the original exception.
3. Subprocesses run without captured output and the installer is built with PyInstaller `--windowed`; therefore pip/training diagnostics may not be visible to the user.
4. The installer downloads a moving `main.zip` archive rather than pinning the app source to the same immutable release commit as the installer. This creates source/binary version skew and makes failures harder to reproduce.

These defects explain why the user sees a generic setup failure without a useful root-cause message. They do **not** prove which underlying installation stage failed on the user's PC. Python 3.11, network access to PyPI, venv creation, and basic PyTorch import/tensor execution have passed the user's reported tests.

## Target architecture

- **Native app and UI:** C++20, Win32 UI first; separate application/core libraries.
- **Network and updates:** WinHTTP, HTTPS-only, pinned release metadata, bounded retries, explicit timeouts, temporary files, checksum/signature validation, atomic replacement, useful persistent logs.
- **Archive handling:** safe extraction with traversal and symlink checks; validate required file manifest before install.
- **Model runtime:** C++ inference engine; evaluate llama.cpp/ggml only if its architecture and licensing fit the requirement. Do not load third-party pretrained weights.
- **From-scratch training:** own C++ training path including random initialization, tokenizer/data pipeline, forward/backward pass, optimizer, evaluation, checkpoints, resume, deterministic seed support, and integrity manifest. Do not claim this is done until implemented and tested.
- **Persistence:** model/checkpoint state separate from replaceable application binaries.
- **Installer UX:** one download, one install, one launch; no user-facing Python, pip, terminal, or venv steps.
- **Compatibility:** CPU baseline first; GPU backends only after hardware-specific tests.
- **Security:** HTTPS; pin the selected commit/release; verify SHA-256 and preferably signed manifests; reject untrusted archive paths; never execute partial or unverified downloads.

## Migration phases

1. **Foundation (current branch):** CMake project, WinHTTP downloader, diagnostic logging.
2. **Reproducible source install:** immutable commit metadata, archive verification, safe extraction, atomic install and rollback.
3. **Native desktop shell:** installer/app window, progress, cancellation, recovery, error details.
4. **C++ inference:** tokenizer, model format, CPU inference, prompt/context handling, streaming output.
5. **C++ scratch training:** numerical unit tests, gradient checks, tiny overfit test, checkpoint round-trip, interrupted resume.
6. **Feature parity:** memory, tools, artifacts, safety boundaries, audit/logging, update mechanism.
7. **Release acceptance:** clean Windows 10/11 VM, no Python installed, network interruption tests, low disk/permission tests, hash mismatch, corrupted archive, CPU-only smoke tests, uninstall/repair/upgrade tests.

## Required release gate

The old Python implementation must remain available until the C++ build passes automated tests and a clean Windows machine can install, start, train a tiny test model from random initialization, resume a checkpoint, and chat without Python or a pretrained checkpoint. No stage may be described as completed without recorded build/test output.

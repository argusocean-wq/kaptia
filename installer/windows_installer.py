"""Kaptia Windows bootstrapper. Built into Kaptia-Setup.exe with PyInstaller."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import urllib.request
import zipfile
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

OWNER_REPO = "argusocean-wq/kaptia"
SOURCE_ZIP = f"https://github.com/{OWNER_REPO}/archive/refs/heads/main.zip"
ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Kaptia"
ZIP_PATH = ROOT / "kaptia-source.zip"

BG = "#090B10"
PANEL = "#111620"
PANEL_2 = "#1B2330"
TEXT = "#F4F7FF"
MUTED = "#9AA7BC"
ACCENT = "#A9B9FF"
BORDER = "#293346"
ACCENT_2 = "#7C5CFF"
GREEN = "#77E4C2"

try:
    from build_info import BUILD_NUMBER, BUILD_TAG
except ImportError:
    BUILD_NUMBER, BUILD_TAG = 0, "development"


class Installer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kaptia Setup")
        self.geometry("720x520")
        self.minsize(660, 480)
        self.configure(bg=BG)
        self.status = tk.StringVar(value="Ready to set up your local AI workspace.")
        self.detail = tk.StringVar(value="The installer will download the app and prepare its runtime.")
        self.percent = tk.DoubleVar(value=0)
        self.phase = tk.StringVar(value="READY")
        self._build_ui()

    def _build_ui(self):
        self.title("Kaptia · 安裝程式")
        self.geometry("790x600")
        self.minsize(720, 560)
        self.configure(bg=BG)

        self.status = tk.StringVar(value="準備好建立你的本機 AI 工作空間")
        self.detail = tk.StringVar(value="安裝程式將下載 Kaptia、建立獨立環境，並準備本機模型。")
        self.percent = tk.DoubleVar(value=0)
        self.phase = tk.StringVar(value="待命")
        self._gradient_phase = 0

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Kaptia.Horizontal.TProgressbar",
            troughcolor=PANEL_2, background=ACCENT,
            bordercolor=PANEL_2, lightcolor=ACCENT,
            darkcolor=ACCENT_2, thickness=7
        )

        # Ambient background and restrained animated accent.
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.canvas.bind("<Configure>", self._paint_background)

        shell = tk.Frame(self, bg=BG)
        shell.pack(fill="both", expand=True, padx=34, pady=26)

        header = tk.Frame(shell, bg=BG)
        header.pack(fill="x")

        mark = tk.Canvas(header, width=46, height=46, bg=BG, highlightthickness=0)
        mark.pack(side="left")
        mark.create_oval(3, 3, 43, 43, fill="#1D2940", outline="#647DAD", width=1)
        mark.create_oval(11, 11, 35, 35, fill="#A9B9FF", outline="")
        mark.create_oval(18, 8, 29, 19, fill="#F4F7FF", outline="")

        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left", padx=(13, 0))
        tk.Label(brand, text="K A P T I A", bg=BG, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w")
        tk.Label(brand, text="LOCAL INTELLIGENCE  /  WINDOWS",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(3, 0))

        badge = tk.Label(header, text="  SECURE LOCAL SETUP  ", bg="#17231F",
                         fg=GREEN, font=("Segoe UI", 8, "bold"),
                         padx=10, pady=7)
        badge.pack(side="right", pady=7)

        tk.Frame(shell, bg=BORDER, height=1).pack(fill="x", pady=(22, 24))

        intro = tk.Frame(shell, bg=BG)
        intro.pack(fill="x")
        tk.Label(intro, text="你的工作空間，\n即將就緒。",
                 bg=BG, fg=TEXT, justify="left",
                 font=("Segoe UI", 29, "bold"), lineheight=36).pack(anchor="w")
        tk.Label(
            intro,
            text="在自己的電腦上準備 Kaptia。模型從隨機初始化開始訓練，安裝流程不會下載預訓練成品模型。",
            bg=BG, fg=MUTED, justify="left", wraplength=690,
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(11, 22))

        card = tk.Frame(shell, bg=PANEL, highlightbackground=BORDER,
                        highlightthickness=1, bd=0)
        card.pack(fill="x")
        inner = tk.Frame(card, bg=PANEL)
        inner.pack(fill="x", padx=22, pady=20)

        topline = tk.Frame(inner, bg=PANEL)
        topline.pack(fill="x")
        tk.Label(topline, text="安裝進度", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        phase_badge = tk.Label(topline, textvariable=self.phase, bg="#202943",
                               fg=ACCENT, padx=10, pady=4,
                               font=("Segoe UI", 8, "bold"))
        phase_badge.pack(side="right")

        tk.Label(inner, textvariable=self.status, bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 13, "bold"), anchor="w").pack(fill="x", pady=(17, 6))
        tk.Label(inner, textvariable=self.detail, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9), anchor="w", justify="left",
                 wraplength=650).pack(fill="x", pady=(0, 17))

        self.progress = ttk.Progressbar(
            inner, style="Kaptia.Horizontal.TProgressbar",
            mode="determinate", maximum=100, variable=self.percent
        )
        self.progress.pack(fill="x")

        meta = tk.Frame(inner, bg=PANEL)
        meta.pack(fill="x", pady=(12, 0))
        tk.Label(meta, text="本機環境 · Windows 10 / 11",
                 bg=PANEL, fg="#71809A", font=("Segoe UI", 8)).pack(side="left")
        tk.Label(meta, text="NO PRETRAINED WEIGHTS",
                 bg=PANEL, fg="#71809A", font=("Segoe UI", 8, "bold")).pack(side="right")

        footer = tk.Frame(shell, bg=BG)
        footer.pack(fill="x", pady=(21, 0))
        tk.Label(footer, text="K A P T I A     ·     LOCAL AI",
                 bg=BG, fg="#6D7B91", font=("Segoe UI", 8, "bold")).pack(side="left")
        self.button = tk.Button(
            footer, text="開始安裝   ↗", command=self.start,
            bg=ACCENT, fg="#0A0D14", activebackground="#C8D2FF",
            activeforeground="#0A0D14", relief="flat", bd=0,
            padx=23, pady=13, cursor="hand2",
            font=("Segoe UI", 10, "bold")
        )
        self.button.pack(side="right")
        self.button.bind("<Enter>", lambda _event: self.button.configure(bg="#C8D2FF"))
        self.button.bind("<Leave>", lambda _event: self.button.configure(bg=ACCENT))
        self._gradient_bar(shell, 3)
        self.after(80, self._animate_accent)

    def _paint_background(self, _event=None):
        # Low-cost ambient glow; no external UI dependencies.
        self.canvas.delete("ambient")
        width = max(1, self.winfo_width())
        for i in range(18):
            x = width - 110 + i * 14
            shade = 28 + i * 2
            color = f"#{shade:02x}{shade + 2:02x}{min(80, shade + 25):02x}"
            self.canvas.create_oval(
                x - 150, -190, x + 160, 130,
                fill=color, outline="", tags="ambient"
            )
        self.canvas.lower("ambient")

    def _animate_accent(self):
        if not self.winfo_exists():
            return
        self._gradient_phase = (self._gradient_phase + 1) % 100
        if hasattr(self, "accent_canvas"):
            self.accent_canvas.delete("line")
            width = max(1, self.accent_canvas.winfo_width())
            for x in range(width):
                t = ((x / width) + self._gradient_phase / 100) % 1.0
                strength = int(70 + 120 * (1 - abs(2 * t - 1)))
                color = f"#{min(255, strength):02x}{min(255, strength + 5):02x}{min(255, strength + 35):02x}"
                self.accent_canvas.create_line(x, 0, x, 3, fill=color, tags="line")
        self.after(100, self._animate_accent)

    def _gradient_bar(self, parent, height=3):
        self.accent_canvas = tk.Canvas(parent, height=height, bg=BG,
                                       highlightthickness=0, bd=0)
        self.accent_canvas.pack(fill="x", side="bottom", pady=(18, 0))
    def update_ui(self, status=None, detail=None, phase=None, percent=None, indeterminate=False):
        def apply():
            if status is not None:
                self.status.set(status)
            if detail is not None:
                self.detail.set(detail)
            if phase is not None:
                self.phase.set(phase)
            if percent is not None:
                self.percent.set(max(0, min(100, percent)))
            if indeterminate:
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate")
        self.after(0, apply)

    def start(self):
        self.button.configure(state="disabled", text="Setting up…")
        self.update_ui("Connecting to GitHub…", "Preparing a verified destination for the application files.",
                       "CONNECTING", 0, True)
        threading.Thread(target=self.install, daemon=True).start()

    def _download_source(self):
        """Download source atomically, retry transient failures, and validate each attempt."""
        import time

        ROOT.mkdir(parents=True, exist_ok=True)
        part_path = ZIP_PATH.with_suffix(".zip.part")
        attempts = 3
        last_error = None

        for attempt in range(1, attempts + 1):
            downloaded = 0
            try:
                part_path.unlink(missing_ok=True)
                request = urllib.request.Request(
                    SOURCE_ZIP,
                    headers={
                        "User-Agent": "Kaptia-Installer",
                        "Accept": "application/zip",
                        "Cache-Control": "no-cache",
                    },
                )
                self.update_ui(
                    "下載 Kaptia 原始檔…",
                    f"正在連線 GitHub（第 {attempt}/{attempts} 次）。",
                    "DOWNLOAD", 0, True,
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    status = getattr(response, "status", 200)
                    if status != 200:
                        raise RuntimeError(f"GitHub returned HTTP {status}")
                    total_header = response.headers.get("Content-Length")
                    total = int(total_header) if total_header and total_header.isdigit() else 0
                    if total > 1024 * 1024 * 1024:
                        raise RuntimeError("Source archive exceeds the 1 GiB safety limit.")

                    last_ui_update = 0
                    with part_path.open("wb") as out:
                        while True:
                            chunk = response.read(256 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            downloaded += len(chunk)
                            if downloaded > 1024 * 1024 * 1024:
                                raise RuntimeError("Download exceeded the 1 GiB safety limit.")
                            now = time.monotonic()
                            if now - last_ui_update >= 0.25:
                                if total > 0:
                                    pct = min(99.0, downloaded * 100 / total)
                                    detail = f"{downloaded / 1048576:.1f} / {total / 1048576:.1f} MB"
                                    self.update_ui("下載 Kaptia 原始檔…", detail, "DOWNLOAD", pct, False)
                                else:
                                    detail = f"已下載 {downloaded / 1048576:.1f} MB"
                                    self.update_ui("下載 Kaptia 原始檔…", detail, "DOWNLOAD", None, True)
                                last_ui_update = now
                        out.flush()
                        os.fsync(out.fileno())

                if downloaded == 0:
                    raise RuntimeError("GitHub returned an empty source archive.")
                if total > 0 and downloaded != total:
                    raise RuntimeError(
                        f"Incomplete download: received {downloaded} bytes, expected {total} bytes."
                    )

                # Validate before replacing the previous archive or extracting anything.
                with zipfile.ZipFile(part_path) as archive:
                    bad_member = archive.testzip()
                    if bad_member is not None:
                        raise RuntimeError(f"ZIP integrity check failed at {bad_member}")
                part_path.replace(ZIP_PATH)
                self.update_ui(
                    "下載完成",
                    f"已接收並驗證 {downloaded / 1048576:.1f} MB 原始檔。",
                    "DOWNLOAD", 100, False,
                )
                return
            except Exception as exc:
                last_error = exc
                part_path.unlink(missing_ok=True)
                if attempt < attempts:
                    self.update_ui(
                        "下載中斷，準備重試…",
                        f"第 {attempt} 次失敗：{type(exc).__name__}。1.5 秒後重試。",
                        "RETRY", 0, True,
                    )
                    time.sleep(1.5 * attempt)

        raise RuntimeError(
            f"Unable to download a valid Kaptia source archive after {attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _install_vscode_extension(self, app_dir):
        extension_dir = app_dir / "vscode-extension"
        required = ["package.json", "extension.js", "README.md"]
        if not all((extension_dir / name).is_file() for name in required):
            self.update_ui("Kaptia installed", "Editor extension source is missing; the desktop assistant can still run.", "EDITOR", 100, False)
            return

        manifest = """<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Language="en-US" Id="kaptia-ask-context" Version="0.1.0" Publisher="kaptia-local" />
    <DisplayName>Kaptia: Ask About Selected Code</DisplayName>
    <Description>Ask the local Kaptia scratch model to explain selected code.</Description>
    <Tags>AI,code,local,context</Tags>
    <Categories>Other</Categories>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="^1.80.0" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionManifest" Value="extension/package.json" />
      <Property Id="Microsoft.VisualStudio.Code.LocalizedLanguages" Value="en-US" />
    </Properties>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code" />
  </Installation>
  <Dependencies />
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" />
  </Assets>
</PackageManifest>
"""
        content_types = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="json" ContentType="application/json" />
  <Default Extension="js" ContentType="application/javascript" />
  <Default Extension="md" ContentType="text/markdown" />
  <Override PartName="/extension.vsixmanifest" ContentType="text/xml" />
</Types>
"""
        vsix_path = ROOT / "Kaptia-Ask-Context.vsix"
        with zipfile.ZipFile(vsix_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
            package.writestr("[Content_Types].xml", content_types)
            package.writestr("extension.vsixmanifest", manifest)
            for name in required:
                package.write(extension_dir / name, "extension/" + name)

        local = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        editors = [
            ("Cursor", shutil.which("cursor"), [
                local / "Programs" / "Cursor" / "resources" / "app" / "bin" / "cursor.cmd",
                local / "Programs" / "Cursor" / "bin" / "cursor.cmd",
            ]),
            ("VS Code", shutil.which("code"), [
                local / "Programs" / "Microsoft VS Code" / "bin" / "code.cmd",
                program_files / "Microsoft VS Code" / "bin" / "code.cmd",
            ]),
        ]
        detected = []
        for name, command, paths in editors:
            candidate = command or next((str(path) for path in paths if path.is_file()), None)
            if candidate and candidate not in [item[1] for item in detected]:
                detected.append((name, candidate))
        if not detected:
            self.update_ui("Kaptia installed", f"Cursor/VS Code was not detected. Extension package: {vsix_path}. To install later, use the editor's Extensions menu > Install from VSIX.", "EDITOR", 100, False)
            return

        installed = []
        failures = []
        for name, command in detected:
            self.update_ui("Installing editor integration…",
                           f"Adding '問問 Kaptia' to {name}. Code remains on this computer; no cloud model is called.",
                           "EDITOR", 100, True)
            try:
                result = subprocess.run([command, "--install-extension", str(vsix_path), "--force"],
                                         cwd=app_dir, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    installed.append(name)
                else:
                    detail = (result.stderr or result.stdout or "editor did not confirm installation").strip()[-350:]
                    failures.append(f"{name}: {detail}")
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if installed:
            detail = "已安裝至 " + "、".join(installed) + "。選取程式碼後按右鍵，選「問問 Kaptia」。答案仍是實驗性模型輸出，請自行核對。"
            if failures:
                detail += "；其他編輯器安裝失敗：" + " | ".join(failures)
            self.update_ui("Editor integration ready", detail, "EDITOR", 100, False)
        else:
            self.update_ui("Kaptia installed; editor integration needs attention",
                           f"Extension package: {vsix_path}. Use Cursor/VS Code Extensions > Install from VSIX. " + " | ".join(failures),
                           "EDITOR", 100, False)

    def _write_version_metadata(self):
        import json
        ROOT.mkdir(parents=True, exist_ok=True)
        data = {"product": "Kaptia", "version": BUILD_TAG, "build_number": BUILD_NUMBER}
        target = ROOT / "version.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp.replace(target)

    def _register_update_tasks(self, venv_python, app_dir):
        updater = app_dir / "scripts" / "kaptia_update.py"
        if not updater.is_file():
            return
        command = f'"{venv_python}" "{updater}" --check'
        tasks = [
            ["schtasks.exe", "/Create", "/SC", "ONLOGON", "/TN", "Kaptia Update Check (Logon)", "/TR", command, "/F", "/RL", "LIMITED", "/IT"],
            ["schtasks.exe", "/Create", "/SC", "HOURLY", "/MO", "6", "/TN", "Kaptia Update Check (6 Hours)", "/TR", command, "/F", "/RL", "LIMITED", "/IT"],
        ]
        failures = []
        for args in tasks:
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=20)
                if result.returncode != 0:
                    failures.append((result.stderr or result.stdout or "Task Scheduler rejected task").strip()[-200:])
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        if failures:
            self.update_ui("Kaptia installed", "Automatic update scheduling needs attention. You can run scripts\\kaptia_update.py manually. " + " | ".join(failures), "UPDATE", 100, False)
        else:
            self.update_ui("Automatic updates enabled", "Kaptia checks GitHub at sign-in and every 6 hours. New installers are SHA-256 verified and require your confirmation.", "UPDATE", 100, False)

    def install(self):
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            self._download_source()

            self.update_ui("Extracting application…", "Validating ZIP integrity and checking archive paths.",
                           "EXTRACT", 100, True)
            with zipfile.ZipFile(ZIP_PATH) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise RuntimeError(f"Downloaded archive failed integrity validation at: {bad_member}")
                root_resolved = ROOT.resolve()
                for member in archive.infolist():
                    target = (ROOT / member.filename).resolve()
                    if target != root_resolved and root_resolved not in target.parents:
                        raise RuntimeError(f"Unsafe archive path: {member.filename}")
                    if member.is_dir():
                        continue
                    mode = (member.external_attr >> 16) & 0o170000
                    if mode == 0o120000:
                        raise RuntimeError(f"Unsupported symbolic link in archive: {member.filename}")
                archive.extractall(ROOT)

            source_dirs = [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("kaptia-")]
            if not source_dirs:
                raise RuntimeError("GitHub archive structure was not recognized.")
            source = max(source_dirs, key=lambda p: p.stat().st_mtime)
            app_dir = ROOT / "app"
            if app_dir.exists():
                shutil.rmtree(app_dir)
            source.rename(app_dir)
            ZIP_PATH.unlink(missing_ok=True)

            self.update_ui("Checking Python runtime…", "Looking for Python 3.10 or 3.11.",
                           "RUNTIME", 100, True)
            python_cmd = None
            for candidate in (["py", "-3.11"], ["py", "-3.10"], ["python"], ["python3"]):
                try:
                    result = subprocess.run(candidate + ["--version"], capture_output=True,
                                            text=True, timeout=15)
                    if result.returncode == 0 and "Python 3." in (result.stdout + result.stderr):
                        python_cmd = candidate
                        break
                except Exception:
                    pass
            if python_cmd is None:
                raise RuntimeError("Python 3.10 or 3.11 was not found. Install Python and enable Add Python to PATH, then run setup again.")

            self.update_ui("Preparing local environment…", "Creating an isolated Python environment.",
                           "ENVIRONMENT", 100, True)
            venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                subprocess.run(python_cmd + ["-m", "venv", str(ROOT / ".venv")],
                               check=True, cwd=app_dir)

            self.update_ui("Installing AI components…",
                           "Dependency installation can take several minutes. Progress is indeterminate while pip resolves packages.",
                           "DEPENDENCIES", 100, True)
            subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                           check=True, cwd=app_dir)
            subprocess.run([str(venv_python), "-m", "pip", "install", "-r",
                            str(app_dir / "requirements.txt")], check=True, cwd=app_dir)

            # Prepare the bundled text corpus and train Kaptia's own scratch model.
            # The checkpoint is stored outside app_dir so reinstalling the app does not erase progress.
            model_dir = ROOT / "model-data" / "kaptia-scratch-pretrain"
            corpus_dir = app_dir / "training" / "scratch_corpus"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.update_ui("Preparing text corpus…",
                           "Using the bundled dialogue seed only. No pretrained model or tokenizer is downloaded.",
                           "CORPUS", 100, True)
            train_bin = corpus_dir / "train.bin"
            if not train_bin.is_file():
                subprocess.run([
                    str(venv_python), str(app_dir / "scripts" / "kaptia_prepare_text_corpus.py"),
                    "--input", str(app_dir / "training" / "bootstrap_dialogue_seed.jsonl"),
                    "--output-dir", str(corpus_dir), "--val-fraction", "0.10",
                    "--min-document-bytes", "8"
                ], check=True, cwd=app_dir)

            latest_checkpoint = model_dir / "latest_checkpoint.pt"
            best_checkpoint = model_dir / "best_checkpoint.pt"
            additional_steps = "1000" if latest_checkpoint.is_file() else "10000"
            train_command = [
                str(venv_python), str(app_dir / "scripts" / "train_kaptia_scratch.py"),
                "--data-dir", str(corpus_dir), "--output-dir", str(model_dir),
                "--steps", additional_steps, "--eval-interval", "500", "--eval-batches", "2",
                "--batch-size", "4", "--block-size", "128", "--layers", "2",
                "--heads", "2", "--width", "64", "--context-length", "256", "--device", "auto"
            ]
            if latest_checkpoint.is_file():
                train_command.extend(["--resume", str(latest_checkpoint)])
                self.update_ui("Continuing Kaptia training…",
                               "Resuming the previous random-initialization checkpoint for 1,000 additional steps.",
                               "TRAINING", 100, True)
            else:
                self.update_ui("Training Kaptia from scratch…",
                               "Initializing every model weight randomly for 10,000 steps. The small bundled corpus is only a starter dataset.",
                               "TRAINING", 100, True)
            subprocess.run(train_command, check=True, cwd=app_dir)

            self._write_version_metadata()
            self._register_update_tasks(venv_python, app_dir)
            self._install_vscode_extension(app_dir)
            self.update_ui("Launching Kaptia…",
                           "Loading Kaptia's own scratch-trained checkpoint. No finished model weights are used.",
                           "COMPLETE", 100, False)
            self.after(800, self.destroy)
            child_env = os.environ.copy()
            child_env["KAPTIA_SCRATCH_CHECKPOINT"] = str(best_checkpoint)
            subprocess.Popen([str(venv_python), str(app_dir / "scripts" / "chat.py")],
                             cwd=app_dir, env=child_env,
                             creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        except Exception as exc:
            self.update_ui("Setup could not finish",
                           "Check your network connection, available disk space, and Python installation.",
                           "ERROR", None, False)
            self.after(0, self.button.configure, {"state": "normal", "text": "Try again  →"})
            self.after(0, lambda: messagebox.showerror("Kaptia Setup failed", str(exc)))


if __name__ == "__main__":
    Installer().mainloop()

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

BG = "#080808"
PANEL = "#151515"
PANEL_2 = "#242424"
TEXT = "#FFFFFF"
MUTED = "#B8B8B8"
ACCENT = "#FFFFFF"
BORDER = "#3A3A3A"


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
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Kaptia.Horizontal.TProgressbar",
                        troughcolor=PANEL_2, background="#FFFFFF",
                        bordercolor=PANEL_2, lightcolor="#FFFFFF", darkcolor="#D8D8D8",
                        thickness=8)

        shell = tk.Frame(self, bg=BG)
        shell.pack(fill="both", expand=True, padx=28, pady=24)

        top = tk.Frame(shell, bg=BG)
        top.pack(fill="x")
        mark = tk.Frame(top, bg="#FFFFFF", width=42, height=42)
        mark.pack(side="left")
        mark.pack_propagate(False)
        tk.Label(mark, text="K", bg="#FFFFFF", fg="#080808",
                 font=("Segoe UI", 22, "bold")).pack(expand=True)
        brand = tk.Frame(top, bg=BG)
        brand.pack(side="left", padx=13)
        tk.Label(brand, text="K A P T I A", bg=BG, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w")
        tk.Label(brand, text="LOCAL AI WORKSPACE  /  WINDOWS SETUP",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(3, 0))
        tk.Label(top, text="SETUP  •  01", bg=BG, fg="#FFFFFF",
                 font=("Segoe UI", 9, "bold")).pack(side="right", pady=10)

        tk.Frame(shell, bg=BORDER, height=1).pack(fill="x", pady=(22, 22))

        tk.Label(shell, text="Your AI workspace,\nready on this PC.",
                 bg=BG, fg=TEXT, justify="left",
                 font=("Segoe UI", 27, "bold")).pack(anchor="w")
        tk.Label(shell, text="Install Kaptia locally. Your model runs on this computer after setup.",
                 bg=BG, fg=MUTED, justify="left",
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(10, 22))

        card = tk.Frame(shell, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x")
        inner = tk.Frame(card, bg=PANEL)
        inner.pack(fill="x", padx=18, pady=18)

        line = tk.Frame(inner, bg=PANEL)
        line.pack(fill="x")
        tk.Label(line, text="INSTALLATION STATUS", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        tk.Label(line, textvariable=self.phase, bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(side="right")
        tk.Label(inner, textvariable=self.status, bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", pady=(13, 5))
        tk.Label(inner, textvariable=self.detail, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9), anchor="w", justify="left",
                 wraplength=590).pack(fill="x", pady=(0, 14))
        self.progress = ttk.Progressbar(inner, style="Kaptia.Horizontal.TProgressbar",
                                         mode="determinate", maximum=100, variable=self.percent)
        self.progress.pack(fill="x")

        footer = tk.Frame(shell, bg=BG)
        footer.pack(fill="x", pady=(20, 0))
        tk.Label(footer, text="K A P T I A  A I  T E A M",
                 bg=BG, fg="#657083", font=("Segoe UI", 8)).pack(side="left")
        self.button = tk.Button(footer, text="Download Kaptia  →", command=self.start,
                                bg="#FFFFFF", fg="#080808", activebackground="#DADADA",
                                activeforeground="#080808", relief="flat", bd=0,
                                padx=22, pady=13, cursor="hand2",
                                font=("Segoe UI", 10, "bold"))
        self.button.pack(side="right")
        self._gradient_bar(shell, 4)

    def _gradient_bar(self, parent, height=4):
        # Tkinter has no native gradient widget; draw a monochrome horizontal gradient.
        canvas = tk.Canvas(parent, height=height, bg=BG, highlightthickness=0, bd=0)
        canvas.pack(fill="x", side="bottom", pady=(18, 0))
        width = 640
        for x in range(width):
            t = x / max(1, width - 1)
            shade = int(35 + 220 * (1 - abs(2 * t - 1)))
            color = f"#{shade:02x}{shade:02x}{shade:02x}"
            canvas.create_line(x, 0, x, height, fill=color)

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

    def install(self):
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(SOURCE_ZIP, headers={"User-Agent": "Kaptia-Installer"})
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                self.update_ui("Downloading Kaptia…", "Downloading application source from GitHub.",
                               "DOWNLOAD", 0, total <= 0)
                with ZIP_PATH.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 / total
                            self.update_ui(
                                "Downloading Kaptia…",
                                f"{downloaded / 1048576:.1f} MB of {total / 1048576:.1f} MB",
                                "DOWNLOAD", pct, False
                            )
                        else:
                            self.update_ui(
                                "Downloading Kaptia…",
                                f"{downloaded / 1048576:.1f} MB downloaded",
                                "DOWNLOAD", None, True
                            )

            self.update_ui("Extracting application…", "Checking archive paths before writing files.",
                           "EXTRACT", 100, True)
            with zipfile.ZipFile(ZIP_PATH) as archive:
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

            self.update_ui("Launching Kaptia…",
                           "First launch may download several gigabytes of model weights from the model host.",
                           "COMPLETE", 100, False)
            self.after(800, self.destroy)
            subprocess.Popen([str(venv_python), str(app_dir / "scripts" / "chat.py")],
                             cwd=app_dir, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        except Exception as exc:
            self.update_ui("Setup could not finish",
                           "Check your network connection, available disk space, and Python installation.",
                           "ERROR", None, False)
            self.after(0, self.button.configure, {"state": "normal", "text": "Try again  →"})
            self.after(0, lambda: messagebox.showerror("Kaptia Setup failed", str(exc)))


if __name__ == "__main__":
    Installer().mainloop()

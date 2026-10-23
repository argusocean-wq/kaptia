"""Kaptia Windows bootstrapper. Built into Kaptia-Setup.exe with PyInstaller."""
from __future__ import annotations
import os
import subprocess
import sys
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


class Installer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kaptia 安裝程式")
        self.geometry("560x340")
        self.minsize(500, 300)
        self.configure(bg="#090909")
        self.status = tk.StringVar(value="準備就緒。安裝需要網路與磁碟空間。")
        tk.Label(self, text="K A P T I A", bg="#090909", fg="#f5f5f5",
                 font=("Segoe UI", 18, "bold")).pack(pady=(24, 4))
        tk.Label(self, text="Windows 本機 AI 助理", bg="#090909", fg="#999999",
                 font=("Segoe UI", 10)).pack()
        tk.Label(self, textvariable=self.status, bg="#090909", fg="#dddddd",
                 wraplength=480, justify="center").pack(pady=22)
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=420)
        self.progress.pack(pady=4)
        self.button = tk.Button(self, text="下載並設定 Kaptia", command=self.start,
                                bg="#f3f3f3", fg="#111111", relief="flat",
                                padx=20, pady=10, font=("Segoe UI", 10, "bold"))
        self.button.pack(pady=18)
        tk.Label(self, text="原始碼：github.com/argusocean-wq/kaptia",
                 bg="#090909", fg="#777777", font=("Segoe UI", 8)).pack(side="bottom", pady=12)

    def set_status(self, value):
        self.after(0, self.status.set, value)

    def start(self):
        self.button.configure(state="disabled")
        self.progress.start(10)
        threading.Thread(target=self.install, daemon=True).start()

    def install(self):
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            self.set_status("正在從 GitHub 下載 Kaptia 程式檔案…")
            request = urllib.request.Request(SOURCE_ZIP, headers={"User-Agent": "Kaptia-Installer"})
            with urllib.request.urlopen(request, timeout=60) as response, ZIP_PATH.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            self.set_status("正在解壓縮程式檔案…")
            with zipfile.ZipFile(ZIP_PATH) as archive:
                archive.extractall(ROOT)
            source_dirs = [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("kaptia-")]
            if not source_dirs:
                raise RuntimeError("GitHub 下載檔案結構不正確，已停止。")
            source = max(source_dirs, key=lambda p: p.stat().st_mtime)
            app_dir = ROOT / "app"
            if app_dir.exists():
                import shutil
                shutil.rmtree(app_dir)
            source.rename(app_dir)
            ZIP_PATH.unlink(missing_ok=True)

            self.set_status("正在檢查 Python 3.10/3.11…")
            python_cmd = None
            for candidate in (["py", "-3.11"], ["py", "-3.10"], [sys.executable]):
                try:
                    result = subprocess.run(candidate + ["--version"], capture_output=True,
                                            text=True, timeout=15)
                    if result.returncode == 0 and "Python 3." in (result.stdout + result.stderr):
                        python_cmd = candidate
                        break
                except Exception:
                    pass
            if python_cmd is None:
                raise RuntimeError("找不到可用的 Python。請先安裝 Python 3.10 或 3.11，並勾選 Add Python to PATH，再重新執行安裝程式。")

            self.set_status("正在建立本機執行環境…")
            venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                subprocess.run(python_cmd + ["-m", "venv", str(ROOT / ".venv")],
                               check=True, cwd=app_dir)
            self.set_status("正在安裝 AI 執行依賴。下載量可能很大，請保持網路連線…")
            subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                           check=True, cwd=app_dir)
            subprocess.run([str(venv_python), "-m", "pip", "install", "-r",
                            str(app_dir / "requirements.txt")], check=True, cwd=app_dir)
            self.set_status("依賴已安裝。首次啟動將從模型託管來源下載模型權重，請預留數 GB 空間…")
            self.after(0, self.progress.stop)
            self.after(0, self.destroy)
            subprocess.Popen([str(venv_python), str(app_dir / "scripts" / "chat.py")],
                             cwd=app_dir, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        except Exception as exc:
            self.after(0, self.progress.stop)
            self.after(0, self.button.configure, {"state": "normal"})
            self.set_status("安裝未完成。請檢查網路、磁碟空間與 Python 安裝。")
            self.after(0, lambda: messagebox.showerror("Kaptia 安裝失敗", str(exc)))


if __name__ == "__main__":
    Installer().mainloop()

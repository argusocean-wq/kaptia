from __future__ import annotations
import hashlib, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

OWNER_REPO = "argusocean-wq/kaptia"
API_URL = f"https://api.github.com/repos/{OWNER_REPO}/releases/latest"
ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Kaptia"
VERSION_PATH = ROOT / "version.json"
STATE_PATH = ROOT / "update-state.json"
UA = "Kaptia-Update-Checker"

def read_json(path, default):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else default
    except (OSError, json.JSONDecodeError):
        return default

def save_state(data):
    ROOT.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(STATE_PATH)

def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as response:
        data = json.loads(response.read(2 * 1024 * 1024).decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Invalid release metadata")
    return data

def download(url, path, limit):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    total = 0
    with urllib.request.urlopen(req, timeout=90) as response, path.open("wb") as output:
        declared = int(response.headers.get("Content-Length", "0") or 0)
        if declared > limit:
            raise RuntimeError("Update asset exceeds size limit")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise RuntimeError("Update asset exceeds size limit")
            output.write(chunk)
    if not total:
        raise RuntimeError("Downloaded update is empty")

def offer_update(release, number, tag):
    assets = release.get("assets", [])
    exe = next((x for x in assets if x.get("name") == "Kaptia-Setup.exe"), None)
    checksum_asset = next((x for x in assets if x.get("name") == "Kaptia-Setup.exe.sha256"), None)
    if not exe or not checksum_asset:
        raise RuntimeError("Latest release does not contain installer and SHA-256 sidecar")
    version = read_json(VERSION_PATH, {}).get("version", "unknown")
    title = str(release.get("name") or tag)
    notes = str(release.get("body") or "No release notes provided.")[:1600]
    root = tk.Tk()
    root.withdraw()
    yes = messagebox.askyesno("Kaptia update available",
        f"A new Kaptia version is available.\n\nInstalled: {version}\nAvailable: {title}\n\nRelease notes:\n{notes}\n\nDownload and verify the installer? Installation will not run without your confirmation.",
        parent=root)
    root.destroy()
    if not yes:
        state = read_json(STATE_PATH, {})
        state.update({"defer_until": int(time.time()) + 86400, "last_deferred_tag": tag})
        save_state(state)
        return
    update_dir = ROOT / "updates"
    update_dir.mkdir(parents=True, exist_ok=True)
    installer = update_dir / f"Kaptia-Setup-{number}.exe"
    checksum = update_dir / f"Kaptia-Setup-{number}.sha256"
    try:
        download(exe["browser_download_url"], installer, 250 * 1024 * 1024)
        download(checksum_asset["browser_download_url"], checksum, 16384)
        expected = checksum.read_text(encoding="utf-8").strip().split()[0].lower()
        actual = hashlib.sha256(installer.read_bytes()).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
            raise RuntimeError("SHA-256 mismatch; refusing to launch the installer")
        save_state({"last_notified_tag": tag, "last_notified_at": int(time.time()), "defer_until": 0})
        subprocess.Popen([str(installer)], cwd=str(update_dir))
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Kaptia update ready", "The installer passed SHA-256 verification and has been opened. Follow its instructions to update.", parent=root)
        root.destroy()
    except Exception:
        installer.unlink(missing_ok=True)
        raise

def main():
    if os.name != "nt":
        print("[Kaptia updater] Desktop notifications currently support Windows only.")
        return 0
    try:
        current = read_json(VERSION_PATH, {})
        current_number = int(current.get("build_number", 0) or 0)
        release = get_json(API_URL)
        tag = str(release.get("tag_name", ""))
        match = re.fullmatch(r"kaptia-build-(\d+)", tag)
        if not match:
            raise RuntimeError(f"Unsupported release tag: {tag!r}")
        number = int(match.group(1))
        if number <= current_number:
            print(f"[Kaptia updater] Up to date (build {current_number}).")
            return 0
        state = read_json(STATE_PATH, {})
        now = int(time.time())
        if int(state.get("defer_until", 0) or 0) > now:
            return 0
        if state.get("last_notified_tag") == tag and now - int(state.get("last_notified_at", 0) or 0) < 21600:
            return 0
        offer_update(release, number, tag)
        return 0
    except Exception as exc:
        print(f"[Kaptia updater] Check skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0

if __name__ == "__main__":
    raise SystemExit(main())

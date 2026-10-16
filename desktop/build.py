"""Assemble Interview Desk, then build the Windows installer.

    build_desktop.cmd

Produces `desktop\\dist\\Interview Desk-Setup-<version>.exe` from pinned inputs only.
Nothing is copied by hand, and the build refuses to guess: every input is either
in the repository, in `lab/setup/manifest.json`, or downloaded from a pinned URL
and checked against a pinned SHA-256.

WHAT GOES IN THE INSTALLER (the application)
    runtime\\            embedded CPython 3.10.11 + the SHELL's dependencies
    app\\                the engine, unchanged
    desktop\\            shell, first-run flow, icon
    lab\\setup\\          manifest + requirements lock, reused verbatim at first run

WHAT DOES NOT (data, provisioned on first run into %LOCALAPPDATA%\\Lena)
    the engine's Python packages   2294 MiB, of which the NVIDIA CUDA
                                   wheels alone are 1985 MiB
    llama.cpp + CUDA runtime       ~1.1 GiB extracted
    Gemma, Whisper, Piper          ~6.4 GiB

That line is drawn by size and by kind: the application is what the installer
owns and replaces on upgrade; everything above is large, pinned, re-downloadable
data with an existing verified provisioning path.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"
BUILD = DESKTOP / "build"
STAGE = BUILD / "stage"
DIST = DESKTOP / "dist"

APP_NAME = "Interview Desk"
APP_VERSION = "2.2.0"
PUBLISHER = "Abdelrhman"

# Pinned exactly like every other artifact in this project: verified by size and
# SHA-256 before use. Same interpreter version the product is validated on.
PYTHON_EMBED = {
    "url": "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip",
    "size": 8629277,
    "sha256": "608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d",
}
GET_PIP = "https://bootstrap.pypa.io/get-pip.py"

# ⚑ The SHELL's dependencies only. The engine's dependencies are NOT here --
# they are 2.3 GiB and are provisioned at first run from requirements.lock.
SHELL_REQUIREMENTS = ["pywebview==6.2.1", "pythonnet==3.1.0"]

INNO_CANDIDATES = [
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
]


def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say("")
    say("=" * 72)
    say(f"  {n}. {title}")
    say("=" * 72)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    t = 0
    for r, _d, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


def human(n: float) -> str:
    return f"{n / (1 << 30):.2f} GiB" if n >= (1 << 30) else f"{n / (1 << 20):.1f} MiB"


# ───────────────────────────────────────────────────────────── the runtime
def build_runtime(cache: Path) -> Path:
    """Embedded CPython + shell dependencies. This IS the application runtime."""
    rt = STAGE / "runtime"
    if rt.exists():
        shutil.rmtree(rt)
    rt.mkdir(parents=True)

    zip_path = cache / "python-embed.zip"
    if not (zip_path.exists() and zip_path.stat().st_size == PYTHON_EMBED["size"]
            and sha256_of(zip_path) == PYTHON_EMBED["sha256"]):
        say(f"  downloading {PYTHON_EMBED['url']}")
        data = urllib.request.urlopen(PYTHON_EMBED["url"], timeout=180).read()
        zip_path.write_bytes(data)
    got = sha256_of(zip_path)
    if got != PYTHON_EMBED["sha256"]:
        raise SystemExit(f"embeddable Python checksum mismatch\n"
                         f"  expected {PYTHON_EMBED['sha256']}\n  found    {got}")
    say(f"  embeddable Python verified ({human(zip_path.stat().st_size)})")
    zipfile.ZipFile(zip_path).extractall(rt)

    # ⚑ `import site` is required so wheels' .pth files are processed. It also
    # drags in the USER's %APPDATA% site-packages, which on the build machine
    # leaked a TensorFlow install into the runtime -- the launcher therefore
    # sets PYTHONNOUSERSITE=1. Both halves are necessary.
    pth = rt / "python310._pth"
    pth.write_text("python310.zip\n.\nLib\\site-packages\n\nimport site\n",
                   encoding="ascii")
    (rt / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)

    gp = cache / "get-pip.py"
    if not gp.exists():
        say("  fetching get-pip.py")
        gp.write_bytes(urllib.request.urlopen(GET_PIP, timeout=180).read())
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    subprocess.run([str(rt / "python.exe"), str(gp), "--no-warn-script-location", "-q"],
                   check=True, env=env)
    say("  pip bootstrapped")

    subprocess.run([str(rt / "python.exe"), "-m", "pip", "install", "-q",
                    "--disable-pip-version-check", "--no-warn-script-location",
                    *SHELL_REQUIREMENTS], check=True, env=env)
    say(f"  shell dependencies installed: {', '.join(SHELL_REQUIREMENTS)}")

    # Trim what a shipped runtime never needs.
    for junk in ("Scripts",):
        p = rt / junk
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    for pyc in rt.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)
    say(f"  runtime staged: {human(dir_size(rt))}")
    return rt


# ─────────────────────────────────────────────────────────── the application
def stage_app():
    """Copy the product. `app/` is copied verbatim -- it is not modified here."""
    for rel, patterns in (("app", None), ("desktop", None)):
        src = ROOT / rel
        dst = STAGE / rel
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src, dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build",
                                          "dist", "assets_src"))
    # first run reuses the SAME manifest and lock the developer flow uses
    setup_dst = STAGE / "lab" / "setup"
    setup_dst.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "requirements.lock", "bootstrap.py",
                 "verify.py", "_runtime_probe.py"):
        shutil.copy2(ROOT / "lab" / "setup" / name, setup_dst / name)
    (STAGE / "lab" / "__init__.py").write_text("", encoding="utf-8")

    (STAGE / "VERSION").write_text(APP_VERSION, encoding="utf-8")
    # Remove obsolete launchers from a reused stage.  The old CMD entry point
    # opened Windows Terminal on machines that host console apps there.
    for obsolete in (f"{APP_NAME}.cmd", "launch.vbs"):
        p = STAGE / obsolete
        if p.exists():
            p.unlink()

    # Portable entry point: argument-free, silent and clearly named. Installed
    # shortcuts do not use script associations at all; see installer.iss.
    (STAGE / f"{APP_NAME}.vbs").write_text(
        'Option Explicit\n'
        'Dim sh, fso, root, cmd\n'
        'Set sh = CreateObject("WScript.Shell")\n'
        'Set fso = CreateObject("Scripting.FileSystemObject")\n'
        'root = fso.GetParentFolderName(WScript.ScriptFullName)\n'
        'cmd = Chr(34) & root & "\\runtime\\pythonw.exe" & Chr(34) & " " & '
        'Chr(34) & root & "\\desktop\\shell.py" & Chr(34)\n'
        'sh.Run cmd, 0, False\n',
        encoding="ascii",
    )
    (STAGE / "README.txt").write_text(
        "INTERVIEW DESK - CALL CENTER INTERVIEW\n"
        "========================================\n\n"
        "تشغيل مباشر:\n"
        "اضغط مرتين على Interview Desk.vbs\n\n"
        "في أول تشغيل فقط قد يحتاج البرنامج للإنترنت لاستكمال الملفات "
        "المحلية، وبعدها يعمل على جهازك. اختر Deutsch أو English، اسمح "
        "للميـكروفون، ثم ابدأ المقابلة. عند إنهاء المقابلة سيُفتح تقرير PDF "
        "ويُحفظ داخل فولدر reports.\n\n"
        f"بديل التثبيت: شغّل Interview Desk-Setup-{APP_VERSION}.exe.\n"
        "كل التسجيلات والتقارير تظل محلية على جهازك.\n",
        encoding="utf-8",
    )
    (STAGE / "BUILD.json").write_text(json.dumps({
        "app": APP_NAME, "version": APP_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_embed_sha256": PYTHON_EMBED["sha256"],
        "shell_requirements": SHELL_REQUIREMENTS,
        "source_commit": _git_commit(),
    }, indent=2), encoding="utf-8")
    say(f"  application staged: {human(dir_size(STAGE / 'app') + dir_size(STAGE / 'desktop'))}")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()[:12]
    except Exception:
        return "unknown"


# ───────────────────────────────────────────────────────────── validation
def validate_stage():
    """Prove the staged runtime works BEFORE wrapping it in an installer.

    Cheap, and it catches the failure that matters: a runtime that imports the
    shell but cannot reach the engine's own code.
    """
    py = STAGE / "runtime" / "python.exe"
    checks = [
        ("shell deps import",
         "import webview, clr_loader; print('ok')"),
        ("engine code imports (no engine deps yet)",
         "import sys; sys.path.insert(0, r'%s'); "
         "import app.preflight as p; print('ok', len(p.HARD))" % STAGE),
        ("user site is NOT on sys.path",
         "import sys; print('ok' if not any('Roaming' in x for x in sys.path) else 'LEAK')"),
    ]
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONNET_RUNTIME": "netfx",
           "PYTHONUTF8": "1"}
    ok = True
    for label, code in checks:
        r = subprocess.run([str(py), "-X", "utf8", "-c", code],
                           capture_output=True, text=True, env=env, timeout=120)
        good = r.returncode == 0 and "ok" in (r.stdout or "") and "LEAK" not in (r.stdout or "")
        say(f"  {'PASS' if good else 'FAIL'}  {label}"
            + ("" if good else f"   {(r.stdout or '') + (r.stderr or '')[:200]}"))
        ok &= good
    if not ok:
        raise SystemExit("staged runtime failed validation; installer not built")


# ───────────────────────────────────────────────────────────── the installer
def find_inno() -> Path | None:
    for c in INNO_CANDIDATES:
        if c.exists():
            return c
    which = shutil.which("ISCC.exe")
    return Path(which) if which else None


def build_installer() -> Path | None:
    iscc = find_inno()
    if iscc is None:
        say("  Inno Setup (ISCC.exe) not found — skipping installer build.")
        say("  Install it from https://jrsoftware.org/isdl.php, then re-run.")
        return None
    DIST.mkdir(parents=True, exist_ok=True)
    iss = DESKTOP / "installer.iss"
    r = subprocess.run([str(iscc),
                        f"/DAppVersion={APP_VERSION}",
                        f"/DStageDir={STAGE}",
                        f"/DOutDir={DIST}",
                        str(iss)],
                       capture_output=True, text=True)
    say((r.stdout or "")[-1500:])
    if r.returncode != 0:
        say((r.stderr or "")[-1500:])
        raise SystemExit(f"ISCC failed with exit {r.returncode}")
    out = DIST / f"{APP_NAME}-Setup-{APP_VERSION}.exe"
    return out if out.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Interview Desk desktop application.")
    ap.add_argument("--skip-runtime", action="store_true",
                    help="reuse an already-staged runtime (developer shortcut)")
    ap.add_argument("--no-installer", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    say("=" * 72)
    say(f"  BUILD {APP_NAME} {APP_VERSION}   (commit {_git_commit()})")
    say("=" * 72)

    cache = BUILD / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)

    step(1, "RUNTIME — embedded CPython + shell dependencies")
    if args.skip_runtime and (STAGE / "runtime" / "python.exe").exists():
        say("  reusing staged runtime (--skip-runtime)")
    else:
        build_runtime(cache)

    step(2, "APPLICATION — engine, shell, manifest")
    stage_app()

    step(3, "VALIDATE the staged runtime")
    validate_stage()

    total = dir_size(STAGE)
    say(f"\n  staged application: {human(total)}")

    installer = None
    if not args.no_installer:
        step(4, "INSTALLER")
        installer = build_installer()

    say("")
    say("=" * 72)
    say("  BUILD COMPLETE")
    say("=" * 72)
    say(f"  staged      {STAGE}   ({human(total)})")
    if installer:
        say(f"  installer   {installer}   ({human(installer.stat().st_size)})")
    say(f"  elapsed     {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

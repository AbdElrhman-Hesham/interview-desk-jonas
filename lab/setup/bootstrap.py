"""First-run bootstrap: clean Windows machine -> a working `start.cmd`.

    setup.cmd

⚑ THIS FILE MUST RUN ON A BARE SYSTEM PYTHON. It executes before `lab/.venv`
exists, so it may import the standard library and nothing else. No requests, no
tqdm, no huggingface_hub. urllib, hashlib, zipfile, venv, subprocess.

STAGES, cheapest failure first — the point is that nothing expensive is
attempted until the thing it depends on has been proven:

  1 preflight   OS, Python, GPU, driver, VRAM, disk, audio      seconds
  2 venv        create lab/.venv, install requirements.lock     ~1.5 GB
  3 runtime     import the native stack; CUDA DLLs resolve      seconds
  4 binaries    llama.cpp + CUDA runtime, then --list-devices   ~0.6 GB
  5 models      Gemma, Piper voice, Whisper snapshot            ~6.9 GB
  6 integrity   re-verify every artifact against the manifest   seconds
  7 verify      real generation, real transcription, real synth ~40 s

Stage 4 ends by asking the freshly extracted llama-server to enumerate CUDA
devices. That is deliberate: it is the last cheap moment at which "this build
cannot talk to this GPU" can be discovered, and it costs nothing compared with
finding out after a 4.8 GiB model download.

Rerunning is safe and cheap. Artifacts that are present and verify are skipped,
downloads resume from `.part` files, and a verified-state cache keeps a re-run
from re-hashing 4.8 GiB for nothing.

Isolation for testing (never point these at the production tree):
    --venv PATH             put the virtualenv somewhere else
    --artifacts-root PATH   put lab/models and lab/bin somewhere else
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETUP_DIR = ROOT / "lab" / "setup"
MANIFEST = SETUP_DIR / "manifest.json"
REQUIREMENTS = SETUP_DIR / "requirements.lock"
UA = "lena-setup/1.0 (+local first-run bootstrap)"

GiB = 1 << 30


# ═════════════════════════════════════════════════════════════════ reporting
class Abort(RuntimeError):
    """A failure the user can act on. The message IS the remedy."""


_t0 = time.perf_counter()


def say(msg: str = ""):
    print(msg, flush=True)


def stage(n: int, title: str):
    say("")
    say("=" * 74)
    say(f"  STAGE {n}  {title}")
    say("=" * 74)


def ok(label: str, detail: str = ""):
    say(f"  [ ok ] {label}" + (f"   {detail}" if detail else ""))


def warn(label: str, detail: str = ""):
    say(f"  [warn] {label}" + (f"   {detail}" if detail else ""))


def info(label: str, detail: str = ""):
    say(f"  [ -- ] {label}" + (f"   {detail}" if detail else ""))


def human(n: float) -> str:
    return f"{n / GiB:.2f} GiB" if n >= GiB else f"{n / (1 << 20):.1f} MiB"


# ═══════════════════════════════════════════════════════════════ verified state
# Hashing 4.8 GiB costs ~20 s. After the first successful verification we record
# (size, mtime_ns, sha256); a re-run trusts that pair and re-hashes only when it
# does not match. `--reverify` always re-hashes. Size is ALWAYS checked, because
# it is free.
class State:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def trusted(self, f: Path, sha: str) -> bool:
        rec = self.data.get(str(f))
        if not rec or rec.get("sha256") != sha:
            return False
        try:
            st = f.stat()
        except OSError:
            return False
        return rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns

    def remember(self, f: Path, sha: str):
        try:
            st = f.stat()
        except OSError:
            return
        self.data[str(f)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                             "sha256": sha, "verified_at": time.time()}
        self.flush()

    def flush(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception:
            pass


def sha256_of(path: Path, label: str = "") -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    last = 0.0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            done += len(chunk)
            if label and total > 256 * (1 << 20) and time.perf_counter() - last > 3.0:
                last = time.perf_counter()
                say(f"         hashing {label}: {100 * done / total:5.1f}%")
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════ stage 1: preflight
# HARD  = setup cannot succeed without it -> abort
# WARN  = the product may still work, but something measured is out of range
# INFO  = recorded for the report, no judgement
def preflight(mf: dict, args) -> dict:
    stage(1, "PREFLIGHT — cheap checks before anything is downloaded")
    found = {}

    # ---- operating system (HARD) ----------------------------------------
    if sys.platform != "win32":
        raise Abort(
            f"This bootstrap targets Windows and this is '{sys.platform}'.\n"
            f"  The pinned llama.cpp build and the CUDA DLL workaround in\n"
            f"  app/providers.py are Windows-specific.")
    ok("windows", f"{platform.system()} {platform.version()}")
    found["os"] = f"{platform.system()} {platform.version()}"

    # ---- disk (HARD) -----------------------------------------------------
    need = mf["disk"]["required_free_bytes"]
    rec = mf["disk"]["recommended_free_bytes"]
    target = Path(args.artifacts_root)
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    found["disk_free"] = free
    if free < need:
        raise Abort(
            f"Not enough free disk space on {target.drive or target}.\n"
            f"  Required : {human(need)}\n"
            f"  Available: {human(free)}\n"
            f"  Downloads alone are {human(mf['disk']['download_bytes'])} and the\n"
            f"  installed stack occupies about {human(mf['disk']['final_on_disk_bytes'])}.\n"
            f"  Free up {human(need - free)} and run setup.cmd again.")
    if free < rec:
        warn("disk space", f"{human(free)} free; {human(rec)} recommended")
    else:
        ok("disk space", f"{human(free)} free on {target.drive or target}")

    # ---- path length (HARD) ----------------------------------------------
    check_path_length(mf, args)

    # ---- python for the venv (HARD) --------------------------------------
    py = find_python(mf, args)
    found["python"] = str(py)

    # ---- GPU (HARD: the architecture is CUDA-only) -----------------------
    gpu = probe_gpu()
    found["gpu"] = gpu
    if not gpu.get("present"):
        raise Abort(
            "No NVIDIA GPU was detected (nvidia-smi did not report one).\n"
            f"  {gpu.get('detail', '')}\n"
            "  This product runs the LLM and Whisper on CUDA; there is no CPU\n"
            "  fallback path in the current architecture. Verified on a\n"
            f"  {mf['gpu']['verified_device']} (compute {mf['gpu']['verified_compute_capability']}).\n"
            "  If you do have an NVIDIA GPU, install the current NVIDIA driver\n"
            "  and check that `nvidia-smi` runs from a terminal.")
    ok("nvidia gpu", f"{gpu['name']}  driver {gpu['driver']}  "
                     f"{gpu['vram_mb']} MiB  compute {gpu.get('cc', '?')}")

    # driver (WARN, not HARD -- the minimum is inferred, and rejecting a
    # machine on an inferred bound is exactly what we were told not to do)
    try:
        cur = tuple(int(x) for x in str(gpu["driver"]).split(".")[:2])
        low = tuple(int(x) for x in str(mf["gpu"]["min_driver"]).split(".")[:2])
        if cur < low:
            warn("driver version",
                 f"{gpu['driver']} is below the {mf['gpu']['min_driver']} assumed "
                 f"for CUDA 12.4. Setup continues; stage 4 will prove whether "
                 f"llama.cpp can actually use this GPU.")
        else:
            ok("driver version", f"{gpu['driver']} >= {mf['gpu']['min_driver']}")
    except Exception:
        info("driver version", "could not be compared")

    # VRAM (WARN -- measured peak is 4469 MB at HEAD, gate is 7000)
    vram_need = mf["gpu"]["vram_required_mb"]
    if gpu.get("vram_mb") and gpu["vram_mb"] < vram_need:
        warn("vram", f"{gpu['vram_mb']} MiB total; {vram_need} MiB is the E0 gate "
                     f"for the LLM with Whisper co-resident. Setup continues, "
                     f"but expect out-of-memory during real use.")
    elif gpu.get("vram_mb"):
        ok("vram", f"{gpu['vram_mb']} MiB total, gate is {vram_need} MiB")

    # ---- audio (WARN -- setup does not need it; the product does) --------
    # Deliberately not HARD: a machine can be set up now and have a headset
    # plugged in later, and enumerating devices from a bare Python without
    # sounddevice installed is not possible anyway. Stage 7 checks properly.
    info("audio devices", "checked after the venv exists (stage 7)")

    return found


def check_path_length(mf: dict, args):
    """Refuse a location where the venv's own files exceed MAX_PATH.

    ⚑ This check exists because the failure it prevents is unactionable. A venv
    under a long path installs every package successfully and then dies at
    import with "DLL load failed ... The filename or extension is too long" and
    "libportaudio64bit.dll: error 0xce" -- messages that point at the library
    rather than at the real cause. Windows resolves DLL dependencies against
    MAX_PATH even when LongPathsEnabled is 1; that was verified, not assumed.
    """
    cfg = mf["path_length"]
    longest, cap, margin = (cfg["venv_longest_relative"], cfg["max_path"],
                            cfg["warn_margin"])
    budget = cap - 1 - longest                 # max length of the venv root

    venv = Path(args.venv).resolve()
    n = len(str(venv))
    if n > budget:
        raise Abort(
            f"The install location is too deep for Windows.\n"
            f"  Virtualenv : {venv}\n"
            f"  That path is {n} characters; the maximum workable here is {budget}.\n"
            f"  Some installed files are up to {longest} characters long on their\n"
            f"  own, and Windows caps the total at {cap} for loading DLLs -- even\n"
            f"  with long paths enabled in the registry.\n"
            f"  If setup continued, every package would install correctly and\n"
            f"  then fail at import with 'DLL load failed ... the filename or\n"
            f"  extension is too long', which points at the wrong thing entirely.\n"
            f"  FIX: move the project folder somewhere shorter -- C:\\Lena is\n"
            f"  ideal -- and run setup.cmd again from there. Nothing else needs\n"
            f"  changing; no downloads are lost.")
    if budget - n < margin:
        warn("path length",
             f"venv root is {n} chars, only {budget - n} below the {budget} limit. "
             f"Moving the project to a shorter path (e.g. C:\\Lena) would remove "
             f"a fragile dependency on folder depth.")
    else:
        ok("path length", f"venv root {n} chars, limit {budget}")


def find_python(mf: dict, args) -> Path:
    """Locate an interpreter matching the pinned major.minor."""
    want = mf["python"]["required_major_minor"]

    if args.python:
        cand = Path(args.python)
        v = python_version(cand)
        if not v:
            raise Abort(f"--python {cand} is not a working Python interpreter.")
        if not v.startswith(want + "."):
            warn("python override",
                 f"{cand} is {v}, not {want}.x. Continuing because you asked "
                 f"explicitly, but only {mf['python']['verified']} is verified.")
        else:
            ok("python (override)", f"{v} at {cand}")
        return cand

    seen = []
    for cand in candidate_pythons(want):
        v = python_version(cand)
        if not v:
            continue
        seen.append(f"{v} ({cand})")
        if v.startswith(want + "."):
            ok("python", f"{v} at {cand}")
            return cand

    raise Abort(
        f"No Python {want}.x interpreter was found, and this project pins {want}.x.\n"
        f"  Found: {', '.join(seen) if seen else 'no Python installations at all'}\n"
        f"  Why it is pinned: {' '.join(mf['python']['why'])}\n"
        f"  Install it from:\n"
        f"    {mf['python']['install_url']}\n"
        f"  Tick 'Add python.exe to PATH' in the installer, then run setup.cmd\n"
        f"  again. If it is already installed somewhere unusual, point at it:\n"
        f"    setup.cmd {mf['python']['override_flag']}")


def candidate_pythons(want: str):
    """Interpreters worth trying, most authoritative first."""
    # The Windows launcher knows about every registered installation.
    # ⚑ Match the path with a regex, not line.split(): "C:\Program Files\..."
    # contains a space and splitting produces two invalid fragments.
    import re
    try:
        out = subprocess.run(["py", "-0p"], capture_output=True, text=True,
                             timeout=30)
        for line in (out.stdout or "").splitlines():
            m = re.search(r"([A-Za-z]:\\.*?python\.exe)", line, re.IGNORECASE)
            if m:
                yield Path(m.group(1))
    except Exception:
        pass
    try:
        out = subprocess.run(["py", f"-{want}", "-c",
                              "import sys;print(sys.executable)"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            yield Path(out.stdout.strip())
    except Exception:
        pass
    for name in (f"python{want}", "python", "python3"):
        p = shutil.which(name)
        if p:
            yield Path(p)
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    if local.is_dir():
        for d in sorted(local.iterdir()):
            exe = d / "python.exe"
            if exe.exists():
                yield exe
    yield Path(sys.executable)


def python_version(exe: Path) -> str | None:
    try:
        out = subprocess.run([str(exe), "-c",
                              "import sys;print('%d.%d.%d'%sys.version_info[:3])"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def probe_gpu() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        for c in (Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                  / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
                  Path(r"C:\Windows\System32\nvidia-smi.exe")):
            if c.exists():
                exe = str(c)
                break
    if not exe:
        return {"present": False, "detail": "nvidia-smi was not found on PATH."}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        line = (out.stdout or "").strip().splitlines()
        if out.returncode != 0 or not line:
            return {"present": False,
                    "detail": f"nvidia-smi exited {out.returncode}: "
                              f"{(out.stderr or '').strip()[:200]}"}
        parts = [p.strip() for p in line[0].split(",")]
        vram = None
        if len(parts) > 2:
            digits = "".join(ch for ch in parts[2] if ch.isdigit())
            vram = int(digits) if digits else None
        return {"present": True, "name": parts[0],
                "driver": parts[1] if len(parts) > 1 else "?",
                "vram_mb": vram, "cc": parts[3] if len(parts) > 3 else None}
    except Exception as e:
        return {"present": False, "detail": f"{type(e).__name__}: {e}"}


# ═════════════════════════════════════════════════════════════ stage 2: venv
def make_venv(py: Path, venv: Path, args) -> Path:
    stage(2, "PYTHON ENVIRONMENT — pinned dependencies")
    vpy = venv / "Scripts" / "python.exe"

    if vpy.exists() and not args.recreate_venv:
        v = python_version(vpy)
        ok("venv exists", f"{venv}  (python {v})")
    else:
        if venv.exists() and args.recreate_venv:
            info("removing existing venv", str(venv))
            shutil.rmtree(venv, ignore_errors=True)
        info("creating venv", str(venv))
        r = subprocess.run([str(py), "-m", "venv", str(venv)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not vpy.exists():
            raise Abort(
                f"Could not create the virtual environment at {venv}.\n"
                f"  python: {py}\n"
                f"  {(r.stderr or r.stdout or '').strip()[:600]}\n"
                f"  Check that the folder is writable and not synced-locked, "
                f"then run setup.cmd again.")
        ok("venv created", str(venv))

    if not REQUIREMENTS.exists():
        raise Abort(f"Missing dependency lock file: {REQUIREMENTS}")

    # `pip install --report` records the resolved URL and hash of every wheel
    # it installs -- the supply-chain record, obtained for free.
    report = SETUP_DIR / "pip-report.json"
    info("installing", f"{REQUIREMENTS.name} (this takes a few minutes)")
    cmd = [str(vpy), "-m", "pip", "install", "--disable-pip-version-check",
           "--only-binary=:all:", "-r", str(REQUIREMENTS),
           "--report", str(report)]
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        raise Abort(
            f"Installing the pinned dependencies failed (pip exit {r.returncode}).\n"
            f"  Lock file: {REQUIREMENTS}\n"
            f"  The output above names the package that failed.\n"
            f"  Most common causes:\n"
            f"    - no internet connection, or a proxy blocking pypi.org\n"
            f"    - a Python that is not {json.loads(MANIFEST.read_text(encoding='utf-8'))['python']['required_major_minor']}.x "
            f"(no matching wheels are published)\n"
            f"  Fix the cause and run setup.cmd again; already-installed\n"
            f"  packages are skipped.")
    ok("dependencies installed", f"report -> {report.name}")
    record_wheel_provenance(report, vpy)
    return vpy


def record_wheel_provenance(report: Path, vpy: Path):
    """Distil pip's resolution into the record worth keeping.

    `pip --report` names, for every wheel, the exact URL it came from and that
    file's SHA-256 — the supply-chain answer to "what is this product actually
    built out of".

    ⚑ On a machine where everything is already satisfied, the INSTALL report's
    `install` list is empty, and an earlier version of this function happily
    overwrote a good provenance file with zero rows. So: use the install report
    when it actually installed something, and otherwise ask pip to resolve the
    lock file without installing (~10 s, no wheels downloaded). Either way the
    record reflects the pinned set rather than the accident of what this
    particular run had to do.
    """
    rows = _provenance_rows(report)
    source = "install"
    if not rows:
        dry = SETUP_DIR / "pip-dryrun.json"
        r = subprocess.run(
            [str(vpy), "-m", "pip", "install", "--disable-pip-version-check",
             "--only-binary=:all:", "--dry-run", "--ignore-installed",
             "-r", str(REQUIREMENTS), "--report", str(dry)],
            capture_output=True, text=True)
        if r.returncode == 0:
            rows = _provenance_rows(dry)
            source = "resolve"
        try:
            dry.unlink(missing_ok=True)
        except Exception:
            pass

    if not rows:
        warn("wheel provenance", "could not be determined; existing record kept")
        return

    missing = [r["name"] for r in rows if not r["sha256"]]
    out = SETUP_DIR / "wheel-provenance.json"
    try:
        out.write_text(json.dumps(
            {"_comment": "Generated by bootstrap.py from pip's own report. "
                         "Every wheel this product depends on, with the URL it "
                         "came from and that file's SHA-256.",
             "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "derived_from": source,
             "count": len(rows),
             "without_hash": missing,
             "wheels": rows}, indent=2), encoding="utf-8")
        ok("wheel provenance", f"{len(rows)} wheels recorded ({source})"
                               + (f", {len(missing)} without a hash" if missing else ""))
    except Exception:
        pass


def _provenance_rows(report: Path) -> list[dict]:
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    for item in data.get("install", []):
        md = item.get("metadata", {})
        dl = item.get("download_info", {}) or {}
        hashes = ((dl.get("archive_info") or {}).get("hashes") or {})
        rows.append({"name": md.get("name"), "version": md.get("version"),
                     "url": dl.get("url"), "sha256": hashes.get("sha256")})
    rows.sort(key=lambda r: (r["name"] or "").lower())
    return rows


# ═════════════════════════════════════════════════════════ stage 3: native stack
def verify_runtime(vpy: Path):
    """Prove the native libraries actually LOAD before downloading ~7 GB.

    `pip` returning 0 means files were copied. It does not mean ctranslate2 can
    find cuBLAS -- on this stack that failure appears only at the first GEMM,
    which historically looked like a broken GPU rather than a packaging gap.
    """
    stage(3, "NATIVE RUNTIME — imports and CUDA libraries, before the big files")
    probe = SETUP_DIR / "_runtime_probe.py"
    r = subprocess.run([str(vpy), "-X", "utf8", str(probe)],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    say((r.stdout or "").rstrip())
    if r.returncode != 0:
        raise Abort(
            "The Python stack installed but could not initialise.\n"
            f"  {(r.stderr or '').strip()[:800]}\n"
            "  Nothing large has been downloaded yet, which is the point of\n"
            "  checking here. Fix the cause above and run setup.cmd again.")
    ok("native runtime", "imports and CUDA libraries resolve")


# ═════════════════════════════════════════════════════ download + verification
def fetch(art: dict, dest: Path, state: State, args) -> None:
    """Download `art` to `dest`, resuming and verifying. Idempotent."""
    size, sha = art["size"], art["sha256"]
    label = art["label"]

    if dest.exists():
        actual = dest.stat().st_size
        if actual == size and (state.trusted(dest, sha) and not args.reverify):
            ok(f"{label}", "present, verified earlier")
            return
        if actual == size:
            info(f"{label}", f"present, verifying {human(size)}…")
            got = sha256_of(dest, label)
            if got == sha:
                state.remember(dest, sha)
                ok(f"{label}", "present and verified")
                return
            handle_corrupt(dest, sha, got, args)
        else:
            info(f"{label}", f"present but {human(actual)} != {human(size)}; refetching")
            dest.unlink(missing_ok=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    download(art["url"], part, size, label)

    info(f"{label}", "verifying checksum…")
    got = sha256_of(part, label)
    if got != sha:
        part.unlink(missing_ok=True)
        raise Abort(
            f"{label}: checksum mismatch after download.\n"
            f"  File     : {dest}\n"
            f"  Expected : {sha}\n"
            f"  Found    : {got}\n"
            f"  Source   : {art['url']}\n"
            f"  The partial file was deleted. This usually means the download\n"
            f"  was corrupted or intercepted. Run setup.cmd again to refetch it.")
    part.replace(dest)
    state.remember(dest, sha)
    ok(f"{label}", f"downloaded and verified ({human(size)})")


def handle_corrupt(dest: Path, expected: str, got: str, args):
    if args.keep_corrupt:
        aside = dest.with_suffix(dest.suffix + f".corrupt-{int(time.time())}")
        dest.replace(aside)
        warn("integrity failure", f"moved to {aside.name}")
    else:
        dest.unlink(missing_ok=True)
        warn("integrity failure", f"deleted {dest.name}; it will be re-downloaded")
    say(f"         expected {expected}")
    say(f"         found    {got}")


def download(url: str, part: Path, size: int, label: str):
    """Resumable GET into `part`. Never silently accepts a short read."""
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink(missing_ok=True)
        have = 0
    if have == size:
        return
    if have:
        info(f"{label}", f"resuming at {human(have)} of {human(size)}")
    else:
        info(f"{label}", f"downloading {human(size)}")

    headers = {"User-Agent": UA}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if have and e.code in (416, 200):        # server refused the range
            part.unlink(missing_ok=True)
            return download(url, part, size, label)
        raise Abort(
            f"{label}: download failed with HTTP {e.code}.\n"
            f"  URL: {url}\n"
            f"  {'The pinned revision may have been withdrawn upstream.' if e.code == 404 else ''}\n"
            f"  Check the connection and run setup.cmd again; the download resumes.")
    except Exception as e:
        raise Abort(
            f"{label}: download failed ({type(e).__name__}: {e}).\n"
            f"  URL: {url}\n"
            f"  Check the internet connection and run setup.cmd again; the\n"
            f"  download resumes from where it stopped.")

    mode = "ab" if (have and resp.status == 206) else "wb"
    if mode == "wb":
        have = 0
    t0 = time.perf_counter()
    last = 0.0
    with resp, part.open(mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            now = time.perf_counter()
            if now - last > 1.0:
                last = now
                rate = have / max(now - t0, 1e-6) / (1 << 20)
                pct = 100 * have / size if size else 0
                say(f"         {pct:5.1f}%  {human(have)} / {human(size)}  {rate:5.1f} MiB/s")

    got = part.stat().st_size
    if got != size:
        raise Abort(
            f"{label}: incomplete download.\n"
            f"  Expected {size} bytes, received {got}.\n"
            f"  The partial file was KEPT at {part.name}, so running setup.cmd\n"
            f"  again resumes instead of starting over.")


# ═════════════════════════════════════════════════════════ stage 4: binaries
def install_binaries(mf: dict, state: State, args):
    stage(4, "BINARIES — llama.cpp and its CUDA runtime")
    root = Path(args.artifacts_root)
    arts = [a for a in mf["artifacts"] if a["stage"] == "binaries"]

    for a in arts:
        archive = root / a["archive"]
        target = root / a["extract_to"]
        sentinel = target / a["sentinel"]
        if sentinel.exists() and not args.reverify:
            ok(a["label"], f"already extracted ({a['sentinel']} present)")
            continue
        fetch(a, archive, state, args)
        info(a["label"], f"extracting -> {target}")
        target.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as z:
                # Zip-slip guard: never write outside the target directory.
                for m in z.namelist():
                    p = (target / m).resolve()
                    if not str(p).startswith(str(target.resolve())):
                        raise Abort(f"{a['label']}: archive contains an unsafe "
                                    f"path ({m}). Refusing to extract.")
                z.extractall(target)
        except zipfile.BadZipFile:
            archive.unlink(missing_ok=True)
            raise Abort(
                f"{a['label']}: the archive is not a valid zip and was deleted.\n"
                f"  Run setup.cmd again to download it fresh.")
        if not sentinel.exists():
            raise Abort(
                f"{a['label']}: extracted, but {a['sentinel']} is not in {target}.\n"
                f"  The release layout may have changed. Nothing further can be\n"
                f"  verified automatically here.")
        ok(a["label"], f"extracted to {target.name}")

    # ⚑ The last cheap moment to discover that this build cannot use this GPU.
    exe = root / "lab" / "bin" / "llama124" / "llama-server.exe"
    r = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                       timeout=120)
    ver = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    ok("llama-server", ver[0] if ver else "(no version output)")

    r = subprocess.run([str(exe), "--list-devices"], capture_output=True,
                       text=True, timeout=180)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0:
        raise Abort(
            f"llama-server could not start (exit {r.returncode}).\n"
            f"  {out[:600]}\n"
            f"  This is the CUDA runtime failing, not a model problem -- no\n"
            f"  model has been downloaded yet. Check the NVIDIA driver version.")
    if "CUDA" not in out.upper():
        raise Abort(
            "llama.cpp started but reports NO CUDA DEVICE:\n"
            f"  {out[:600]}\n"
            "  The 4.8 GiB model download has been skipped on purpose -- it\n"
            "  would not have run. Update the NVIDIA driver and try again.")
    for line in out.splitlines():
        if line.strip().startswith("CUDA"):
            ok("llama.cpp sees the GPU", line.strip())


# ═══════════════════════════════════════════════════════════ stage 5: models
def install_models(mf: dict, vpy: Path, state: State, args):
    stage(5, "MODELS — the large downloads")
    root = Path(args.artifacts_root)
    for a in [x for x in mf["artifacts"] if x["stage"] == "models"]:
        fetch(a, root / a["dest"], state, args)

    for snap in mf.get("hf_snapshots", []):
        info(snap["label"], f"snapshot {snap['revision'][:12]} "
                            f"(~{human(snap['approx_bytes'])})")
        code = (
            "import sys\n"
            "from huggingface_hub import snapshot_download\n"
            f"p = snapshot_download(repo_id={snap['repo']!r},"
            f" revision={snap['revision']!r})\n"
            "print(p)\n"
        )
        r = subprocess.run([str(vpy), "-X", "utf8", "-c", code],
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode != 0:
            raise Abort(
                f"{snap['label']}: download failed.\n"
                f"  repo    : {snap['repo']}\n"
                f"  revision: {snap['revision']}\n"
                f"  {(r.stderr or '').strip()[:600]}\n"
                f"  huggingface_hub caches partial downloads, so running\n"
                f"  setup.cmd again resumes rather than starting over.")
        ok(snap["label"], (r.stdout or "").strip().splitlines()[-1][:70])


# ═══════════════════════════════════════════════════════ stage 6: integrity
def integrity(mf: dict, state: State, args) -> list[str]:
    stage(6, "INTEGRITY — every artifact re-checked against the manifest")
    root = Path(args.artifacts_root)
    problems = []
    for a in mf["artifacts"]:
        dest = root / (a.get("dest") or a["archive"])
        if not dest.exists():
            problems.append(f"{a['label']}: missing at {dest}")
            continue
        if dest.stat().st_size != a["size"]:
            problems.append(f"{a['label']}: size {dest.stat().st_size} != {a['size']}")
            continue
        if args.reverify or not state.trusted(dest, a["sha256"]):
            got = sha256_of(dest, a["label"])
            if got != a["sha256"]:
                problems.append(f"{a['label']}: sha256 {got} != {a['sha256']}")
                continue
            state.remember(dest, a["sha256"])
        ok(a["label"], "size + sha256 match")
    return problems


# ═══════════════════════════════════════════════════════ stage 7: verification
def verify_app(vpy: Path, args) -> int:
    stage(7, "VERIFICATION — the real stack, briefly")
    r = subprocess.run([str(vpy), "-X", "utf8", str(SETUP_DIR / "verify.py")],
                       text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return r.returncode


# ═══════════════════════════════════════════════════════════════════════ main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="First-run bootstrap for Lena (Windows).")
    ap.add_argument("--python", help="interpreter to build the venv from")
    ap.add_argument("--venv", default=str(ROOT / "lab" / ".venv"))
    ap.add_argument("--artifacts-root", default=str(ROOT),
                    help="root under which lab/models and lab/bin are placed")
    ap.add_argument("--recreate-venv", action="store_true")
    ap.add_argument("--reverify", action="store_true",
                    help="re-hash every artifact instead of trusting the cache")
    ap.add_argument("--keep-corrupt", action="store_true",
                    help="move failed artifacts aside instead of deleting them")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip stage 7 (does not make the run a success)")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    say("=" * 74)
    say("  LENA — FIRST-RUN SETUP")
    say("  Local German conversation partner. Nothing leaves this machine")
    say("  except the downloads listed below.")
    say("=" * 74)

    if not MANIFEST.exists():
        say(f"\n  FAILED: manifest not found at {MANIFEST}")
        return 1
    mf = json.loads(MANIFEST.read_text(encoding="utf-8"))
    state = State(SETUP_DIR / "state.json")

    try:
        found = preflight(mf, args)
        if args.preflight_only:
            say("\n  preflight only — stopping here as requested.")
            return 0

        vpy = make_venv(Path(found["python"]), Path(args.venv), args)
        verify_runtime(vpy)
        install_binaries(mf, state, args)
        install_models(mf, vpy, state, args)

        problems = integrity(mf, state, args)
        if problems:
            say("")
            for p in problems:
                say(f"  [FAIL] {p}")
            raise Abort(
                "Integrity verification failed for the artifacts listed above.\n"
                "  Run setup.cmd again: files that fail their checksum are\n"
                "  re-downloaded, files that pass are left alone.")

        rc = 0 if args.skip_verify else verify_app(vpy, args)
        if args.skip_verify:
            warn("verification skipped", "setup is NOT proven; run setup.cmd --verify-only")

    except Abort as e:
        say("")
        say("=" * 74)
        say("  SETUP FAILED")
        say("=" * 74)
        for line in str(e).splitlines():
            say(f"  {line}")
        say("")
        say(f"  Nothing was left half-installed that a re-run cannot finish.")
        say(f"  Elapsed {time.perf_counter() - _t0:.0f}s")
        return 1
    except KeyboardInterrupt:
        say("\n\n  Interrupted. Run setup.cmd again — downloads resume.")
        return 130

    say("")
    say("=" * 74)
    if rc == 0 and not args.skip_verify:
        say("  SETUP COMPLETE — VERIFIED")
        say("=" * 74)
        say("")
        say("  Start a conversation:")
        say("      start.cmd")
        say("")
        say("  Full regression check (~4 min, needs speakers and a quiet room):")
        say("      regress.cmd")
    elif args.skip_verify:
        say("  SETUP COMPLETE — NOT VERIFIED")
        say("=" * 74)
        say("  Artifacts are installed and their checksums match, but the")
        say("  stack was not exercised. Run:  setup.cmd --verify-only")
    else:
        say("  SETUP INCOMPLETE — VERIFICATION FAILED")
        say("=" * 74)
        say("  The artifacts are installed and verified, but the stack did not")
        say("  pass stage 7. The output above names the component.")
    say(f"\n  Elapsed {time.perf_counter() - _t0:.0f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

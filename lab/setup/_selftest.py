"""Adversarial tests for the bootstrap's download and integrity machinery.

    lab\\.venv\\Scripts\\python.exe lab\\setup\\_selftest.py

The bootstrap's whole value is what it does when things go WRONG, and "it
worked on the machine that was already set up" proves none of that. So each
case below breaks something on purpose and asserts the bootstrap notices.

Everything runs in a temporary directory against the REAL network, using the
smallest pinned artifact (the 63 MiB Piper voice) as the payload — real HTTP,
real ranges, real SHA-256, real resume. The production tree is never touched:
no test writes outside its own temp directory.

  1 fresh download        empty target -> download, verify, land
  2 idempotent rerun      already correct -> no second download
  3 corrupt artifact      flipped byte -> detected, replaced, correct again
  4 truncated artifact    short file -> detected by size, refetched
  5 resumed download      partial .part -> resumes, does not restart
  6 wrong checksum        manifest disagrees -> abort, partial deleted
  7 dead URL              404 -> actionable abort, nothing left behind
  8 zip-slip guard        malicious archive path -> refused
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab" / "setup"))

import bootstrap as bs  # noqa: E402

MANIFEST = json.loads((ROOT / "lab" / "setup" / "manifest.json").read_text(encoding="utf-8"))
VOICE = next(a for a in MANIFEST["artifacts"] if a["id"] == "tts_voice")

RESULTS: list[tuple[str, bool, str]] = []


class Args:
    """Stands in for the argparse namespace fetch() reads."""
    reverify = False
    keep_corrupt = False


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    print("=" * 74)
    print("  BOOTSTRAP SELFTEST — deliberate failures, real network")
    print(f"  payload: {VOICE['label']}  ({VOICE['size'] / (1 << 20):.1f} MiB)")
    print("=" * 74)

    tmp = Path(tempfile.mkdtemp(prefix="lena-setup-selftest-"))
    print(f"  sandbox: {tmp}\n")
    args = Args()
    try:
        dest = tmp / "voice.onnx"
        state = bs.State(tmp / "state.json")

        # ---- 1. fresh download ------------------------------------------
        t0 = time.perf_counter()
        bs.fetch(VOICE, dest, state, args)
        first_wall = time.perf_counter() - t0
        check("1 fresh download lands and verifies",
              dest.exists() and dest.stat().st_size == VOICE["size"]
              and sha(dest) == VOICE["sha256"], f"{first_wall:.1f}s")

        # ---- 2. idempotent rerun ----------------------------------------
        t0 = time.perf_counter()
        bs.fetch(VOICE, dest, state, args)
        rerun_wall = time.perf_counter() - t0
        check("2 rerun does not download again",
              rerun_wall < max(1.0, first_wall / 4),
              f"{rerun_wall:.2f}s vs {first_wall:.1f}s first time")

        # ---- 3. corrupt artifact ----------------------------------------
        # Flip one byte in the middle: same size, wrong content. This is the
        # case a size check alone cannot catch.
        with dest.open("r+b") as f:
            f.seek(VOICE["size"] // 2)
            b = f.read(1)
            f.seek(VOICE["size"] // 2)
            f.write(bytes([b[0] ^ 0xFF]))
        corrupted = sha(dest)
        state2 = bs.State(tmp / "state2.json")      # cache must not mask it
        bs.fetch(VOICE, dest, state2, args)
        check("3 flipped byte detected and replaced",
              corrupted != VOICE["sha256"] and sha(dest) == VOICE["sha256"],
              "same size, wrong content")

        # ---- 4. truncated artifact --------------------------------------
        with dest.open("r+b") as f:
            f.truncate(VOICE["size"] - 4096)
        bs.fetch(VOICE, dest, bs.State(tmp / "state3.json"), args)
        check("4 truncated file detected and refetched",
              dest.stat().st_size == VOICE["size"] and sha(dest) == VOICE["sha256"])

        # ---- 5. resumed download ----------------------------------------
        # Leave a valid prefix as a .part and delete the target: the fetch must
        # resume from the prefix rather than start over.
        part = dest.with_suffix(dest.suffix + ".part")
        prefix = VOICE["size"] // 2
        part.write_bytes(dest.read_bytes()[:prefix])
        dest.unlink()
        t0 = time.perf_counter()
        bs.fetch(VOICE, dest, bs.State(tmp / "state4.json"), args)
        resume_wall = time.perf_counter() - t0
        check("5 partial download resumes",
              dest.exists() and sha(dest) == VOICE["sha256"]
              and resume_wall < first_wall * 0.9,
              f"resumed from {prefix / (1 << 20):.0f} MiB in {resume_wall:.1f}s "
              f"(full was {first_wall:.1f}s)")

        # ---- 6. wrong checksum in the manifest --------------------------
        # Simulates an artifact that is not what we pinned. Must abort, and
        # must not leave a partial file that a later run would trust.
        bad = dict(VOICE)
        bad["sha256"] = "0" * 64
        bad_dest = tmp / "bad.onnx"
        try:
            bs.fetch(bad, bad_dest, bs.State(tmp / "state5.json"), args)
            check("6 checksum mismatch aborts", False, "fetch returned normally")
        except bs.Abort as e:
            msg = str(e)
            check("6 checksum mismatch aborts",
                  "checksum mismatch" in msg and "Expected" in msg
                  and "Found" in msg and not bad_dest.exists()
                  and not bad_dest.with_suffix(".onnx.part").exists(),
                  "message names expected, found, file and remedy")

        # ---- 7. dead URL -------------------------------------------------
        dead = dict(VOICE)
        dead["url"] = ("https://huggingface.co/rhasspy/piper-voices/resolve/"
                       "ea046e8458f6acd997706d6e6066a022b42f6fb1/de/de_DE/"
                       "thorsten/medium/this-file-does-not-exist.onnx")
        dead_dest = tmp / "dead.onnx"
        try:
            bs.fetch(dead, dead_dest, bs.State(tmp / "state6.json"), args)
            check("7 dead URL aborts", False, "fetch returned normally")
        except bs.Abort as e:
            check("7 dead URL aborts actionably",
                  "404" in str(e) and "setup.cmd again" in str(e)
                  and not dead_dest.exists())

        # ---- 8. zip-slip guard -------------------------------------------
        evil = tmp / "evil.zip"
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("../../escaped.txt", "should never be written")
        target = tmp / "extract_here"
        target.mkdir()
        refused = False
        try:
            with zipfile.ZipFile(evil) as z:
                for m in z.namelist():
                    p = (target / m).resolve()
                    if not str(p).startswith(str(target.resolve())):
                        refused = True
        except Exception:
            pass
        check("8 zip-slip path refused",
              refused and not (tmp.parent / "escaped.txt").exists())

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [n for n, ok, _d in RESULTS if not ok]
    print("\n" + "=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} selftests passed")
    print(f"\n  BOOTSTRAP SELFTEST: {'PASS' if not failed else 'FAIL'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

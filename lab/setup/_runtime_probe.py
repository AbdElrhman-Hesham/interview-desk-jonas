"""Stage 3 — prove the native stack LOADS, before ~7 GB is downloaded.

`pip install` returning 0 means files were copied. It does not mean the
extension modules can resolve their CUDA dependencies. On this stack that
distinction is not academic:

    ctranslate2 4.8.1 on Windows ships no cuBLAS. With no CUDA Toolkit
    installed, the Whisper model ALLOCATES 1670 MB of VRAM successfully and
    then dies at the first GEMM with "cublas64_12.dll is not found". The fix
    is the nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels PLUS an explicit
    os.add_dll_directory() registration, because Python 3.8+ deliberately
    stopped resolving extension-module DLLs through PATH.

Importing app.providers performs that registration, so this probe exercises
the same path the product uses rather than a reimplementation of it.

Runs inside the venv. No model files are needed and none are touched.
"""
from __future__ import annotations

import ctypes
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []


def check_import(mod: str, why: str):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "")
        print(f"         import {mod:<16} ok {v}")
        return m
    except Exception as e:
        FAIL.append(f"import {mod} failed ({type(e).__name__}: {e}) — {why}")
        print(f"         import {mod:<16} FAILED  {type(e).__name__}: {e}")
        return None


def main() -> int:
    print(f"         python {sys.version.split()[0]}  {sys.executable}")

    for mod, why in (
        ("numpy", "array plumbing for every stage"),
        ("sounddevice", "microphone capture and playback"),
        ("soundfile", "reading the learner recordings"),
        ("psutil", "telemetry resource probe"),
        ("ctranslate2", "the Whisper inference engine"),
        ("faster_whisper", "STT"),
        ("onnxruntime", "Piper inference"),
        ("piper", "TTS"),
        ("huggingface_hub", "fetching the pinned Whisper snapshot"),
    ):
        check_import(mod, why)

    # ⚑ Registers site-packages/nvidia/*/bin on the DLL search path. This import
    # is the product's own bootstrap, not a test fixture.
    try:
        from app import providers
        dirs = getattr(providers, "_CUDA_DIRS", [])
        print(f"         CUDA DLL dirs registered: {len(dirs)}")
        for d in dirs[:4]:
            print(f"           {d}")
    except Exception as e:
        FAIL.append(f"app.providers could not be imported ({type(e).__name__}: {e})")
        print(f"         app.providers FAILED {type(e).__name__}: {e}")
        dirs = []

    # The exact DLL whose absence produced a load-succeeds / inference-fails bug.
    for dll, required in (("cublas64_12.dll", True),
                          ("cublasLt64_12.dll", True),
                          ("cudnn64_9.dll", False)):
        try:
            ctypes.CDLL(dll)
            print(f"         {dll:<20} loadable")
        except OSError as e:
            msg = f"{dll} is not loadable ({e})"
            print(f"         {dll:<20} NOT LOADABLE")
            if required:
                FAIL.append(
                    msg + " — ctranslate2 needs it and the Windows wheel does "
                          "not bundle it. Expected from the nvidia-cublas-cu12 "
                          "wheel, registered via os.add_dll_directory().")

    # Does the machine expose a CUDA device to ctranslate2 at all? Capability
    # queries lie on this stack (they reported all seven compute types on a
    # configuration where none could execute), so treat a 0 here as fatal but
    # a non-zero as merely encouraging — stage 7 does the real work.
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        print(f"         ctranslate2 CUDA devices: {n}")
        if n < 1:
            FAIL.append("ctranslate2 reports 0 CUDA devices — Whisper cannot "
                        "run on GPU and there is no CPU path in this design.")
    except Exception as e:
        FAIL.append(f"ctranslate2 CUDA query failed ({type(e).__name__}: {e})")

    if FAIL:
        print("\n         RUNTIME PROBE FAILED:")
        for f in FAIL:
            print(f"           - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

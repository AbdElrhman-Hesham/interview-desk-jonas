"""Stage 7 — exercise the real stack briefly, so "installed" means something.

    setup.cmd --verify-only

The deterministic success condition for setup. It runs the SHIPPING classes —
`providers.LlamaCppProvider`, `providers.STT`, `providers.TTS` — not
reimplementations, so a pass says the real path works:

  LLM   llama-server starts on the real model and produces real tokens.
        `/health` returning 200 is NOT proof of life here: a known llama-server
        failure answers /health for hours while every completion times out, so
        LlamaCppProvider.start() requires a real generation and this inherits
        that.
  STT   the pinned Whisper snapshot loads on CUDA and transcribes. This is the
        step that fails when cuBLAS is missing — the model allocates VRAM
        happily and dies at the first GEMM.
  TTS   Piper loads the pinned voice and synthesises audible-length PCM.
  AUDIO devices are enumerated and reported. Missing devices are a WARNING:
        setup can legitimately run before a headset is plugged in.

Deliberately NOT here: playback continuity, hands-free lifecycle, endpointing,
the evaluation path. Those are `regress.cmd` — ~4 minutes, speakers, and a
quiet room. Making setup depend on room acoustics would be a bad trade.

Takes roughly 40 s, almost all of it loading models.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
WARNINGS: list[str] = []


def record(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def check_files() -> bool:
    """Nothing below can work if the artifacts are not where config expects."""
    all_ok = True
    for label, p in (("LLM model", config.LLM_MODEL_PATH),
                     ("llama-server", config.LLAMA_BUILD_DIR / "llama-server.exe"),
                     ("Piper voice", Path(config.TTS_VOICE)),
                     ("Piper voice config", Path(str(config.TTS_VOICE) + ".json"))):
        exists = p.exists()
        all_ok &= exists
        record(f"{label} present", exists, str(p) if not exists else p.name)
    return all_ok


def check_llm() -> bool:
    from app.providers import LlamaCppProvider
    llm = LlamaCppProvider(config.LLM_MODEL_PATH, config.LLAMA_BUILD_DIR,
                           config.LLM_PORT, config.LLM_CTX, config.LLM_NGL,
                           config.LLM_SERVER_ARGS, config.REQ_TIMEOUT_S)
    try:
        t0 = time.perf_counter()
        llm.start()                       # includes a real 8-token generation
        load = time.perf_counter() - t0
        record("llama-server starts and generates", True,
               f"load {load:.1f}s, generation {llm.generation}")

        # One real German turn through the shipping streaming path.
        t0 = time.perf_counter()
        text, first_ms, toks = "", None, 0
        for delta, meta in llm.stream_chat(
                [{"role": "user", "content": "Sag bitte einen kurzen Satz auf Deutsch."}],
                max_tokens=40):
            text += delta
            if first_ms is None and meta.get("first_token_ms"):
                first_ms = meta["first_token_ms"]
            toks = meta.get("tokens", toks)
        wall = (time.perf_counter() - t0) * 1000
        good = bool(text.strip())
        record("LLM streams German text", good,
               f"first token {first_ms} ms, {toks} tokens, {wall:.0f} ms")
        if good:
            print(f"        -> {text.strip()[:110]}")
        return good
    except Exception as e:
        record("llama-server starts and generates", False,
               f"{type(e).__name__}: {e}")
        return False
    finally:
        try:
            llm.stop()
        except Exception:
            pass


def check_stt() -> bool:
    """Loads the pinned snapshot and runs a real decode on the GPU."""
    try:
        from app.providers import STT
        t0 = time.perf_counter()
        stt = STT()                       # constructor warms the model
        load = time.perf_counter() - t0
        record("Whisper loads on CUDA", True,
               f"{config.STT_MODEL} {config.STT_COMPUTE}, {load:.1f}s")
    except Exception as e:
        record("Whisper loads on CUDA", False, f"{type(e).__name__}: {e}")
        return False

    try:
        # Two seconds of shaped noise: no transcript is asserted, because the
        # audio is not speech. What is asserted is that the CUDA decode path
        # RUNS and returns a well-formed result — that is the step which fails
        # when cuBLAS is missing.
        rng = np.random.default_rng(7)
        audio = (rng.standard_normal(2 * config.RECORD_SR) * 0.02).astype(np.float32)
        t0 = time.perf_counter()
        out = stt.transcribe(audio)
        wall = (time.perf_counter() - t0) * 1000
        good = isinstance(out, dict) and "text" in out and "latency_ms" in out
        record("Whisper decodes on the GPU", good,
               f"{wall:.0f} ms, beam {config.STT_BEAM_SIZE}")
        return good
    except Exception as e:
        record("Whisper decodes on the GPU", False, f"{type(e).__name__}: {e}")
        return False


def check_tts() -> bool:
    try:
        from app.providers import TTS
        t0 = time.perf_counter()
        tts = TTS(config.TTS_VOICE)       # constructor warms Piper
        load = time.perf_counter() - t0
        record("Piper loads the pinned voice", True,
               f"{Path(config.TTS_VOICE).name}, {tts.sr} Hz, {load:.1f}s")
    except Exception as e:
        record("Piper loads the pinned voice", False, f"{type(e).__name__}: {e}")
        return False

    try:
        lv = config.LEVELS[config.DEFAULT_LEVEL]
        t0 = time.perf_counter()
        pcm, first = tts.synth("Hallo, schön dass du da bist.", lv["length_scale"])
        wall = (time.perf_counter() - t0) * 1000
        secs = len(pcm) / tts.sr if tts.sr else 0
        good = secs > 0.5
        record("Piper synthesises German", good,
               f"{secs:.2f}s of audio in {wall:.0f} ms (first chunk {first} ms)")
        return good
    except Exception as e:
        record("Piper synthesises German", False, f"{type(e).__name__}: {e}")
        return False


def check_audio():
    """WARNING, not FAIL — a machine can be set up before the headset arrives."""
    try:
        import sounddevice as sd
    except Exception as e:
        WARNINGS.append(f"sounddevice unavailable: {type(e).__name__}: {e}")
        return
    for kind in ("input", "output"):
        try:
            d = sd.query_devices(kind=kind)
            print(f"  [ ok ] default {kind:<6} {d['name']}")
        except Exception as e:
            WARNINGS.append(
                f"no default {kind} device ({e}). "
                + ("Lena cannot hear you until a microphone is available."
                   if kind == "input" else
                   "Lena cannot speak until an output device is available."))


def main() -> int:
    print("=" * 74)
    print("  VERIFICATION — the real stack")
    print("=" * 74)
    t0 = time.perf_counter()

    if not check_files():
        print("\n  Required files are missing. Run setup.cmd to fetch them.")
        return 1

    llm_ok = check_llm()
    stt_ok = check_stt()
    tts_ok = check_tts()
    check_audio()

    wall = time.perf_counter() - t0
    failed = [n for n, ok, _d in RESULTS if not ok]

    print("\n" + "=" * 74)
    for w in WARNINGS:
        print(f"  [warn] {w}")
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed in {wall:.0f}s")

    ok = llm_ok and stt_ok and tts_ok and not failed
    print(f"\n  VERIFICATION: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  The LLM, Whisper and Piper all ran on this machine.")
        print("  start.cmd should work. For the full product invariants: regress.cmd")
    else:
        print("  The checks above name the component that did not work.")

    try:
        out = ROOT / "lab" / "setup" / "last-verify.json"
        out.write_text(json.dumps({
            "at": time.time(), "pass": ok, "wall_s": round(wall, 1),
            "python": sys.version.split()[0],
            "checks": [{"name": n, "pass": o, "detail": d} for n, o, d in RESULTS],
            "warnings": WARNINGS,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

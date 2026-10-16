"""Headless end-to-end smoke test of the MVP loop.

Runs the real pipeline (streaming LLM -> sanitizer -> Piper) with playback
stubbed out, so the pipeline is exercised without needing a microphone or
speakers. Verifies streaming chunking, latency, telemetry and containment.

⚑ THIS IS A SYNTHETIC RUN AND SAYS SO. The session is marked `synthetic: true`
and every turn carries `input_source: "synthetic"`, because the "learner turns"
are hardcoded strings, not speech. Nothing here is evidence about a real user:
not the word ratio, not the speaking time, and above all not STT latency.

⚑ STT IS NOT EXERCISED HERE, so `stt_latency_ms` is reported as **None**, not 0.
The previous version injected `{"latency_ms": 0}`, which propagated into
`median_stt_ms: 0` in every session file and was mistaken for a real
measurement. An unmeasured stage is null. Forever.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

from app import config  # noqa: E402
from app.engine import Engine  # noqa: E402

# stub playback -- we measure synthesis, not speakers
_played = {"n": 0, "samples": 0}


def _play(pcm, sr):
    _played["n"] += 1
    _played["samples"] += len(pcm)


sd.play = _play
sd.wait = lambda: None
sd.stop = lambda: None


class _NullStream:
    """Stands in for sd.OutputStream so the smoke test needs no audio device.

    Reports underflowed=False always. ⚑ That means this harness can NEVER
    validate playback continuity -- only a real device can. Continuity is
    measured by lab/experiments/e04_tts_continuity/measure_gaps.py.
    """

    def __init__(self, *a, **kw):
        self.sr = kw.get("samplerate")

    def start(self):
        pass

    def write(self, data):
        _played["n"] += 1
        _played["samples"] += len(data)
        return False

    def stop(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass


sd.OutputStream = _NullStream

events = []


def on_event(e):
    events.append(e)
    if e["kind"] in ("boot", "ready", "first_audio", "turn_done", "recovering",
                     "turn_error", "boot_failed"):
        print(f"  [{e['kind']}] " + " ".join(
            f"{k}={v}" for k, v in e.items() if k not in ("kind", "t")))


def main():
    print("=" * 68)
    print("  MVP SMOKE TEST  —  SYNTHETIC INPUT, STUBBED PLAYBACK")
    print("  (not evidence about a real user; STT is not exercised)")
    print("=" * 68)
    e = Engine(level="B1", on_event=on_event, synthetic=True)
    t0 = time.perf_counter()
    e.start()
    boot_s = time.perf_counter() - t0
    print(f"\n  boot (LLM+STT+TTS, all warmed): {boot_s:.1f}s")
    print(f"  opener chunks played: {_played['n']}")
    print(f"  state after opener: {e.state}")

    turns = [
        "Ich habe heute Fußball gespielt.",
        "Mit Freunden, nur zum Spaß.",
        "Ich war gestern in der pharmacy.",
    ]
    for i, text in enumerate(turns, 1):
        print(f"\n  --- turn {i}: {text!r}")
        before = _played["n"]
        e.turn += 1
        e.messages.append({"role": "user", "content": text})
        t = time.perf_counter()
        # stt_latency_ms deliberately absent -> stored as None, never 0
        e._speak_streamed(t, text, 3.0,
                          {"latency_ms": None, "avg_logprob": None,
                           "n_pauses": None, "total_pause_s": None},
                          None, None)
        rec = e.log.turns[-1]
        print(f"      first_audio={rec['e2e_speech_end_to_first_audio_ms']}ms  "
              f"first_token={rec['llm_first_token_ms']}ms  "
              f"tok/s={rec['llm_tok_per_s']}  "
              f"tts_first_chunk={rec['tts_first_chunk_ms']}ms  "
              f"chunks={_played['n']-before}")
        print(f"      stt_latency_ms={rec['stt_latency_ms']}  (None = not exercised)")
        print(f"      sanitizer={rec['sanitizer_removed'] or 'clean'}")
        print(f"      LENA: {rec['ai_text_spoken'][:140]}")

    summary = e.log.close(e.llm.stalls, reason="user_ended")
    e.stop(reason="user_ended")

    print("\n" + "=" * 68)
    print("  RESULT")
    print("=" * 68)
    print(f"  synthetic             {summary['synthetic']}")
    print(f"  turns                 {summary['turns']}  (real: {summary['turns_real']})")
    print(f"  product metrics valid {summary['product_metrics_valid']}"
          f"   <- False is CORRECT for a synthetic run")
    print(f"  median STT            {summary['median_stt_ms']}"
          f"  (n={summary['n_stt_ms']})   <- None/0 samples is CORRECT here")
    print(f"  median first-audio    {summary['median_e2e_first_audio_ms']} ms"
          f"  (n={summary['n_e2e_first_audio_ms']})")
    print(f"  median TTS 1st chunk  {summary['median_tts_first_chunk_ms']} ms"
          f"  (n={summary['n_tts_first_chunk_ms']})")
    print(f"  median tok/s          {summary['median_tok_per_s']}"
          f"  (n={summary['n_tok_per_s']})")
    print(f"  sanitizer total       {summary['sanitizer_removed_total'] or 'clean'}")
    print(f"  stalls / restarts     {summary['n_stalls']} / {summary['n_server_restarts']}")
    print(f"  end reason            {summary['end_reason']} (clean={summary['clean_end']})")
    print(f"  VRAM peak             {summary['vram_peak_mb']} MB")
    print(f"  session file          {e.log.path.name}")

    ok = (summary["turns"] == len(turns)
          and summary["n_stalls"] == 0
          and summary["median_stt_ms"] is None          # must NOT be 0
          and summary["product_metrics_valid"] is False
          and summary["clean_end"] is True
          and (summary["median_e2e_first_audio_ms"] or 9e9) < 4000)
    print(f"\n  SMOKE TEST: {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

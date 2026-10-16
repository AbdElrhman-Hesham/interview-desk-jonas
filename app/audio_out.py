"""Continuous speech playback.

⚑ WHY THIS EXISTS (measured, not assumed). The first real user session produced
the report "Lena's speech sometimes sounds CHOPPED / SEGMENTED ... like several
generated pieces rather than one natural continuous utterance."

E4 (`lab/experiments/e04_tts_continuity`) instrumented the shipped playback path
and found the cause exactly. The old loop was:

    synth(chunk N) -> sd.play(N) -> sd.wait() -> synth(N+1) -> sd.play(N+1) ...

`sd.wait()` blocks until chunk N has finished playing, and only THEN is chunk
N+1 synthesised. So the silence between two chunks equals the synthesis time of
the next one. Measured over 8 boundaries of real Lena replies:

    gap_before_ms   229.7  144.9  155.0  112.3  159.6  107.1  190.9   57.6
    synth_ms        229.6  144.5  154.9  112.2  159.4  106.9  190.8   57.5

The two columns agree to within 0.3 ms. The chopping was a structural hole in
the middle of every sentence, median 149.9 ms, not a prosody artefact.

THE FIX, in two parts:
  1. synthesis runs AHEAD of playback on its own thread, so chunk N+1 is already
     PCM by the time chunk N ends;
  2. one continuous `sd.OutputStream` is written to, instead of starting and
     stopping the device per chunk.

Result on the same script: 0 device underflows, median gap 2.1 ms.

`OutputStream.write()` returning `underflowed=True` is the authoritative signal
that the device actually ran out of samples -- wall-clock gap arithmetic is only
a model, this is the hardware reporting a real audible hole. It is recorded per
chunk so regressions are detectable from telemetry alone.
"""
from __future__ import annotations

import time

import numpy as np
import sounddevice as sd


class SpeechPlayer:
    """One continuous output stream for a single Lena turn.

    Usage:
        with SpeechPlayer(sr) as p:
            p.write(pcm)          # as chunks become available
        p.stats                   # after the turn
    """

    def __init__(self, samplerate: int, device=None):
        self.sr = samplerate
        self.device = device
        self.stream = None
        self.chunks: list[dict] = []
        self.n_underflows = 0
        self.first_write_at: float | None = None
        self._prev_audio_end: float | None = None
        self._aborted = False

    # ------------------------------------------------------------ lifecycle
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def open(self):
        self.stream = sd.OutputStream(
            samplerate=self.sr, channels=1, dtype="int16",
            device=self.device, blocksize=0, latency="low")
        self.stream.start()
        return self

    def close(self):
        """Drain remaining audio and release the device."""
        if self.stream is None:
            return
        try:
            if not self._aborted:
                self.stream.stop()          # lets queued audio finish
            self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None

    def abort(self):
        """Cut playback immediately -- used when the learner interrupts."""
        self._aborted = True
        if self.stream is None:
            return
        try:
            self.stream.abort()             # discards queued audio, unlike stop()
        except Exception:
            pass

    # --------------------------------------------------------------- write
    def write(self, pcm: np.ndarray, text: str = "", synth_ms: float | None = None,
              boundary: str = "") -> bool:
        """Append PCM to the continuous stream. Returns False if aborted.

        Blocks while the device consumes audio, which is what keeps synthesis
        from running arbitrarily far ahead. Records whether the device starved.
        """
        if self.stream is None or self._aborted or pcm.size == 0:
            return not self._aborted

        t0 = time.perf_counter()
        if self.first_write_at is None:
            self.first_write_at = t0
        gap_ms = (None if self._prev_audio_end is None
                  else round((t0 - self._prev_audio_end) * 1000, 1))
        try:
            underflowed = bool(self.stream.write(pcm.reshape(-1, 1)))
        except Exception:
            return False
        dur = len(pcm) / self.sr
        start_audio = t0 if self._prev_audio_end is None else max(t0, self._prev_audio_end)
        self._prev_audio_end = start_audio + dur
        if underflowed:
            self.n_underflows += 1
        self.chunks.append({
            "text": text, "boundary": boundary,
            "synth_ms": None if synth_ms is None else round(synth_ms, 1),
            "audio_ms": round(dur * 1000, 1),
            "gap_before_ms": max(0.0, gap_ms) if gap_ms is not None else None,
            "underflow": underflowed,
        })
        return not self._aborted

    # --------------------------------------------------------------- stats
    @property
    def stats(self) -> dict:
        gaps = [c["gap_before_ms"] for c in self.chunks
                if c["gap_before_ms"] is not None]
        return {
            "n_chunks": len(self.chunks),
            "n_underflows": self.n_underflows,
            "gap_ms_max": round(max(gaps), 1) if gaps else None,
            "gap_ms_median": round(float(np.median(gaps)), 1) if gaps else None,
            "audio_ms_total": round(sum(c["audio_ms"] for c in self.chunks), 1)
            if self.chunks else None,
            "aborted": self._aborted,
            "chunks": self.chunks,
        }

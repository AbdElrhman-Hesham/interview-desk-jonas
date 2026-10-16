"""Inference providers: LLM, STT, TTS.

Thin interfaces so the LLM can be swapped without touching the engine. The MVP
ships one implementation of each; the abstraction costs ~30 lines and keeps the
provisional model decision reversible.

⚑ CUDA DLL note: importing this module registers site-packages/nvidia/*/bin on
the Windows DLL search path. Python 3.8+ ignores PATH for extension-module DLL
resolution, so faster-whisper cannot find cuBLAS without it. This MUST happen
before faster_whisper is imported.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import config


# --------------------------------------------------------- CUDA DLL bootstrap
def _register_cuda_dlls() -> list[str]:
    if sys.platform != "win32":
        return []
    added = []
    for sp in map(Path, sys.path):
        nv = sp / "nvidia"
        if not nv.is_dir():
            continue
        for sub in sorted(nv.iterdir()):
            for cand in (sub / "bin", sub / "lib"):
                if cand.is_dir() and any(cand.glob("*.dll")):
                    try:
                        os.add_dll_directory(str(cand))
                        added.append(str(cand))
                    except (OSError, AttributeError):
                        pass
        if added:
            break
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    return added


_CUDA_DIRS = _register_cuda_dlls()


# ------------------------------------------------------------------- LLM base
class LLMProvider:
    """Interface. stream_chat yields text deltas."""

    def start(self): ...
    def stop(self): ...
    def stream_chat(self, messages: list[dict], **kw): ...
    @property
    def info(self) -> dict: return {}


class StallError(RuntimeError):
    pass


class LlamaCppProvider(LLMProvider):
    def __init__(self, model_path: Path, build_dir: Path, port: int, ctx: int,
                 ngl: int, server_args: tuple, req_timeout: int):
        self.model_path, self.build_dir = Path(model_path), Path(build_dir)
        self.port, self.ctx, self.ngl = port, ctx, ngl
        self.server_args, self.req_timeout = tuple(server_args), req_timeout
        self.proc: subprocess.Popen | None = None
        self.generation = 0
        self.stalls: list[dict] = []
        self.load_s = None

    # -------------------------------------------------------------- lifecycle
    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _port_free(self) -> bool:
        s = socket.socket(); s.settimeout(0.8)
        try:
            s.connect(("127.0.0.1", self.port)); return False
        except Exception:
            return True
        finally:
            s.close()

    def start(self, timeout: float = 300.0):
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        exe = self.build_dir / "llama-server.exe"
        if not exe.exists():
            raise FileNotFoundError(f"llama-server not found: {exe}")
        if not self._port_free():
            raise RuntimeError(f"port {self.port} occupied before start")

        self.generation += 1
        cmd = [str(exe), "-m", str(self.model_path), "-c", str(self.ctx),
               "-ngl", str(self.ngl), "--port", str(self.port),
               "--host", "127.0.0.1", "-np", "1", *self.server_args]
        t0 = time.perf_counter()
        # llama-server is a console-subsystem executable. On Windows 11 with
        # Windows Terminal as the default console host, spawning it normally
        # opens a large blank Terminal window even though stdout/stderr are
        # discarded. CREATE_NO_WINDOW keeps the model process fully headless.
        creationflags = (subprocess.CREATE_NO_WINDOW
                         if sys.platform == "win32" else 0)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     creationflags=creationflags)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                    if r.status == 200:
                        self.load_s = round(time.perf_counter() - t0, 2)
                        if not self._smoke():
                            raise RuntimeError("server failed smoke test")
                        return self
            except RuntimeError:
                raise
            except Exception:
                time.sleep(0.4)
        self.stop()
        raise TimeoutError("llama-server never became healthy")

    def _smoke(self) -> bool:
        """/health 200 is NOT proof of life -- the known hang answers /health
        for hours while every completion times out. Require a real generation."""
        try:
            body = json.dumps({"messages": [{"role": "user", "content": "Sag Hallo."}],
                               "max_tokens": 8, "temperature": 0.1}).encode()
            req = urllib.request.Request(f"{self.base}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            return bool((d.get("choices") or [{}])[0].get("message", {}).get("content", "").strip())
        except Exception:
            return False

    def stop(self):
        if not self.proc:
            return
        try:
            self.proc.terminate(); self.proc.wait(timeout=20)
        except Exception:
            try:
                self.proc.kill(); self.proc.wait(timeout=8)
            except Exception:
                pass
        self.proc = None
        time.sleep(1.5)

    def restart(self):
        self.stop()
        self.start()

    # ------------------------------------------------------------- generation
    def stream_chat(self, messages: list[dict], max_tokens: int = 220, **kw):
        """Yields (delta_text, meta). Raises StallError on timeout so the engine
        can restart the server and retry -- never a silent frozen conversation."""
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "stream": True, **config.LLM_SAMPLING, **kw}
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first_token_ms = None
        n_tok = 0
        try:
            with urllib.request.urlopen(req, timeout=self.req_timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if not delta:
                        continue
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - t0) * 1000, 1)
                    n_tok += 1
                    yield delta, {"first_token_ms": first_token_ms}
        except Exception as e:
            rec = {"generation": self.generation, "waited_ms": round((time.perf_counter()-t0)*1000, 1),
                   "error": f"{type(e).__name__}: {e}", "at": time.time()}
            self.stalls.append(rec)
            raise StallError(json.dumps(rec)) from e
        total = (time.perf_counter() - t0)
        yield "", {"first_token_ms": first_token_ms, "tokens": n_tok,
                   "total_ms": round(total * 1000, 1),
                   "tok_per_s": round(n_tok / total, 2) if total > 0 else 0}

    @property
    def info(self) -> dict:
        return {"provider": "llamacpp", "model": self.model_path.name,
                "generation": self.generation, "ctx": self.ctx,
                "server_args": list(self.server_args), "load_s": self.load_s}


# ------------------------------------------------------------------------ STT
class STT:
    def __init__(self, language: str | None = None):
        from faster_whisper import WhisperModel
        self.language = language or config.STT_LANGUAGE
        self.model = WhisperModel(config.STT_MODEL, device=config.STT_DEVICE,
                                  compute_type=config.STT_COMPUTE)
        self.warm()

    def set_language(self, language: str):
        """Switch the pinned Whisper model between German and English."""
        if language not in ("de", "en"):
            raise ValueError("Whisper language must be 'de' or 'en'")
        self.language = language
        self.warm()

    def warm(self):
        # Cold first transcribe measured at 3648 ms vs ~390 ms warm. Without
        # this the FIRST turn of every session is ~3 s slower than the rest.
        list(self.model.transcribe(np.zeros(16000, dtype=np.float32),
                                   language=self.language, beam_size=1)[0])

    def transcribe(self, audio16k: np.ndarray) -> dict:
        """⚑ beam_size comes from config and is 5, not 1.

        E8b, measured on 17 real learner recordings:
          - beam 1 vs beam 5 disagreed on 8/17 transcripts, and beam 5 was
            consistently the better reading ("das wird ein Praxisches" ->
            "das ist eine Praxis"; "Welche erfahrungen man" -> "Welche
            Erfahrung man"), with correct capitalisation and punctuation.
          - beam 5 is DETERMINISTIC: two identical runs differed 0/17, against
            1/17 for greedy. Reproducibility matters here because the Mistakes
            Panel judges morphology; a transcript that changes between runs
            makes corrections unreproducible.
          - cost is +28 ms/turn (270 -> 298 ms), which the user has explicitly
            said is acceptable in exchange for quality.
        beam_size=1 was inherited from E0, where it was chosen for latency and
        never evaluated for accuracy on this user's German.

        ⚑ condition_on_previous_text stays False: it stops Whisper carrying
        context across turns and "repairing" learner German toward fluent
        German. Transcription is Whisper's job; correction is the panel's.
        """
        t0 = time.perf_counter()
        audio16k, input_gain = self._prepare_audio(audio16k)
        segs, info = self.model.transcribe(
            audio16k, language=self.language,
            beam_size=config.STT_BEAM_SIZE,
            patience=1.2, temperature=0.0,
            word_timestamps=True, condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350,
                            "speech_pad_ms": 250},
            hotwords=config.STT_HOTWORDS[self.language])
        segs = list(segs)
        text = " ".join(s.text for s in segs).strip()
        words = [w for s in segs for w in (s.words or [])]
        gaps, prev = [], None
        for w in words:
            if prev is not None and w.start - prev > 0.35:
                gaps.append(w.start - prev)
            prev = w.end
        return {"text": text,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "input_gain": round(input_gain, 3),
                "language_probability": round(float(getattr(info, "language_probability", 0) or 0), 3),
                "words": [{"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3),
                           "p": round(float(getattr(w, "probability", 0) or 0), 3)}
                          for w in words][:80],
                "avg_logprob": round(float(np.mean([s.avg_logprob for s in segs])), 3) if segs else None,
                "no_speech_prob": round(float(np.mean([s.no_speech_prob for s in segs])), 3) if segs else None,
                "n_pauses": len(gaps), "total_pause_s": round(sum(gaps), 2)}

    @staticmethod
    def _prepare_audio(audio16k: np.ndarray) -> tuple[np.ndarray, float]:
        """Remove DC and gently lift only a genuinely quiet microphone.

        Real learner recordings around 0.04--0.05 RMS got worse when normalized,
        so normal speech is left untouched. Gain starts only below 0.02 RMS, is
        capped at 1.5x (+3.5 dB), and always remains clipping-safe.
        """
        x = np.asarray(audio16k, dtype=np.float32).reshape(-1).copy()
        if not x.size:
            return x, 1.0
        x = np.nan_to_num(x, copy=False)
        x -= float(np.mean(x, dtype=np.float64))
        frame = 320
        usable = x[:x.size // frame * frame]
        if not usable.size:
            return x, 1.0
        rms = np.sqrt(np.mean(usable.reshape(-1, frame).astype(np.float64) ** 2,
                              axis=1))
        voiced = rms[rms > config.SPEECH_RMS_FLOOR]
        if not voiced.size:
            return x, 1.0
        level = float(np.percentile(voiced, 65))
        gain = 1.0 if level >= 0.02 else min(1.5, 0.025 / max(level, 1e-5))
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            gain = min(gain, 0.97 / peak)
        gain = max(0.1, gain)
        x *= gain
        return x, gain


# ------------------------------------------------------------------------ TTS
class TTS:
    def __init__(self, voice_path: Path):
        from piper import PiperVoice
        self.voice_path = Path(voice_path)
        self.profile = config.TTS_PROFILES.get(self.voice_path.name, {})
        self.voice = PiperVoice.load(str(voice_path))
        self.sr = self.voice.config.sample_rate
        self.warm()

    def warm(self):
        # Cold first synthesis measured at 4438 ms vs ~180 ms warm.
        for _ in self.voice.synthesize("Willkommen.",
                                       syn_config=self._synthesis_config(1.0)):
            pass

    def _synthesis_config(self, length_scale: float):
        from piper import SynthesisConfig
        return SynthesisConfig(length_scale=length_scale,
                               speaker_id=self.profile.get("speaker_id"),
                               noise_scale=self.profile.get("noise_scale"),
                               noise_w_scale=self.profile.get("noise_w_scale"),
                               volume=self.profile.get("volume", 1.0))

    def synth(self, text: str, length_scale: float = 1.0) -> tuple[np.ndarray, float]:
        """Returns (int16 pcm, first_audio_ms)."""
        t0 = time.perf_counter()
        buf, first = [], None
        cfg = self._synthesis_config(length_scale)
        try:
            chunks = self.voice.synthesize(text, syn_config=cfg)
        except TypeError:                      # older piper signature
            chunks = self.voice.synthesize(text)
        for c in chunks:
            buf.append(np.frombuffer(c.audio_int16_bytes, dtype=np.int16))
            if first is None:
                first = round((time.perf_counter() - t0) * 1000, 1)
        if not buf:
            return np.zeros(0, np.int16), 0.0
        return np.concatenate(buf), (first or 0.0)

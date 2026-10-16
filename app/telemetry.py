"""Durable session logging + telemetry.

Append-only JSONL, fsync'd per record. A crash or a runtime hang can never
destroy a completed turn -- that lesson cost 12 hours during E1 Stage A.

This is EXPERIMENT telemetry. Almost none of it is shown in the UI.

⚑ NEVER STORE A FAKE ZERO. An unmeasured quantity is `null`, not 0. The old
version let the smoke harness inject `stt_latency_ms: 0`, which propagated into
`median_stt_ms: 0` in every session file and was mistaken for a real
measurement for two days. Every summary statistic below therefore reports
alongside itself how many turns actually contributed to it (`_n`), and returns
null rather than a number when nothing did.

⚑ SYNTHETIC RUNS ARE MARKED. Sessions driven by the smoke harness carry
`synthetic: true` and their turns carry `input_source: "synthetic"`. Product
conclusions must never be drawn from them -- the existing `learner_word_ratio`
values in the pre-alpha session files came from hardcoded strings, not speech.

⚑ HARNESS RUNS ARE MARKED TOO, and `synthetic` is NOT enough to find them.
Some verification harnesses must construct `Engine(synthetic=False)` on purpose,
because the code path under test is the real one: e05 verify_cycle replays the
user's OWN recordings through the real microphone callback, so its turns come
out with `input_source: "microphone"` AND a real `audio_file` -- indistinguishable
from a learner session by inspection. Three such files already exist on disk
(s20260808-183249, s20260808-184158, s20260810-171503) and the previously
documented rule for spotting harness runs ("synthetic false but audio_file null")
does not catch them. Automating those harnesses would have produced a new
counterfeit "real session" on every regression run.

So: any process that sets LENA_HARNESS in its environment stamps every session
it writes with `harness: "<name>"`. The real launcher (start.cmd -> app.server)
never sets it, so genuine learner sessions carry `harness: null`. The rule for
real-learner evidence is now: `synthetic == false` AND `harness == null` AND
turns carry a non-null `audio_file`.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import time
from pathlib import Path

from . import config

SCHEMA_VERSION = 2


def _probe() -> dict:
    """Best-effort resource snapshot. Missing keys stay missing -- not zero."""
    out = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_used_mb"] = round(vm.used / 1048576)
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        out["vram_used_mb"] = round(m.used / 1048576)
        out["vram_free_mb"] = round(m.free / 1048576)
        out["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(h, 0)
    except Exception:
        pass
    return out


class SessionLog:
    def __init__(self, level: str, llm_info: dict, synthetic: bool = False,
                 input_device=None, language: str = "de", tts_voice=None,
                 session_kind: str = "conversation"):
        self.session_id = f"s{dt.datetime.now():%Y%m%d-%H%M%S}"
        self.path = config.SESSIONS_DIR / f"{self.session_id}.jsonl"
        self.t0 = time.time()
        self.turns: list[dict] = []
        self.flags: list[dict] = []
        self.vram_peak = 0
        self.synthetic = synthetic
        self.language = language
        self.tts_voice = Path(tts_voice or config.TTS_VOICE).name
        # ⚑ set by verification harnesses (see module docstring). None for the
        # real launcher, so a harness run can never be mistaken for evidence.
        self.harness = os.environ.get("LENA_HARNESS") or None
        self.closed = False
        self.n_restarts = 0
        self._append({
            "kind": "session_start", "schema": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "language": language,
            "session_kind": session_kind,
            # ⚑ a synthetic session is NOT evidence about a real learner
            "synthetic": synthetic,
            # ⚑ nor is a harness-driven one, even when synthetic is False
            "harness": self.harness,
            "input_device": input_device,
            "model_decision": config.MODEL_DECISION,
            "model_status": config.MODEL_STATUS,
            "known_risks": config.KNOWN_RISKS,
            "llm": llm_info,
            "stt": {"model": config.STT_MODEL, "compute": config.STT_COMPUTE,
                    "language": language},
            "tts": {"voice": self.tts_voice,
                    "sr": config.TTS_SAMPLE_RATE},
            "containment": {"req_timeout_s": config.REQ_TIMEOUT_S,
                            "restart_every_turns": config.RESTART_EVERY_TURNS,
                            "max_turn_retries": config.MAX_TURN_RETRIES,
                            "max_consec_stalls": config.MAX_CONSEC_STALLS},
            "resources": _probe(),
        })

    def _append(self, rec: dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------ turns
    def turn(self, rec: dict):
        res = _probe()
        if res.get("vram_used_mb"):
            self.vram_peak = max(self.vram_peak, res["vram_used_mb"])
        rec = {"kind": "turn", "schema": SCHEMA_VERSION,
               "session_id": self.session_id,
               "t_offset_s": round(time.time() - self.t0, 1), **rec,
               "resources": res}
        self.turns.append(rec)
        self._append(rec)

    def event(self, kind: str, **payload):
        """Durable non-turn event: server restart, stall, recovery, abort."""
        if kind == "server_restart":
            self.n_restarts += 1
        self._append({"kind": kind, "session_id": self.session_id,
                      "t_offset_s": round(time.time() - self.t0, 1),
                      "at": dt.datetime.now().isoformat(timespec="seconds"),
                      **payload})

    def flag(self, turn: int, code: str, note: str = ""):
        rec = {"kind": "flag", "session_id": self.session_id, "turn": turn,
               "code": code, "note": note,
               "t_offset_s": round(time.time() - self.t0, 1),
               "at": dt.datetime.now().isoformat(timespec="seconds")}
        self.flags.append(rec)
        self._append(rec)

    # ---------------------------------------------------------------- summary
    def _stat(self, key: str):
        """Median over turns that ACTUALLY measured `key`.

        Returns (median_or_None, n_contributing). A metric nobody measured is
        null with n=0 -- it is never silently rendered as 0.
        """
        vals = [t[key] for t in self.turns
                if isinstance(t.get(key), (int, float))]
        if not vals:
            return None, 0
        return round(statistics.median(vals), 1), len(vals)

    def _drift(self) -> dict:
        """Second-half vs first-half comparison of Lena's own output."""
        t = [x for x in self.turns if x.get("ai_words")]
        if len(t) < 6:
            return {"drift_measured": False}
        h = len(t) // 2
        def med(v):
            v = sorted(v)
            return round(statistics.median(v), 1) if v else None
        def qrate(rows):
            txt = [(r.get("ai_text_spoken") or "").strip() for r in rows]
            txt = [x for x in txt if x]
            return round(sum(1 for x in txt if x.endswith("?")) / len(txt), 2) if txt else None
        return {
            "drift_measured": True,
            "ai_words_median_first_half": med([r["ai_words"] for r in t[:h]]),
            "ai_words_median_second_half": med([r["ai_words"] for r in t[h:]]),
            "question_end_rate_first_half": qrate(t[:h]),
            "question_end_rate_second_half": qrate(t[h:]),
        }

    def close(self, stalls: list, reason: str = "user_ended"):
        """Write the session summary.

        `reason` distinguishes a clean end from an abnormal one. If this is
        never called the file simply has no session_end record, which is itself
        the signal that the process died -- see `was_abnormal()` consumers.
        """
        if self.closed:
            return {}
        self.closed = True
        dur = time.time() - self.t0

        real = [t for t in self.turns if t.get("input_source") != "synthetic"]
        lw = sum(t.get("learner_words") or 0 for t in self.turns)
        aw = sum(t.get("ai_words") or 0 for t in self.turns)
        ls = sum(t.get("learner_speech_s") or 0 for t in self.turns)
        as_ = sum(t.get("ai_speech_s") or 0 for t in self.turns)

        san = {}
        for t in self.turns:
            for k, v in (t.get("sanitizer_removed") or {}).items():
                san[k] = san.get(k, 0) + v

        # every latency below carries its own sample count
        metrics = {}
        for name, key in (
            ("stt_ms", "stt_latency_ms"),
            ("llm_first_token_ms", "llm_first_token_ms"),
            ("tok_per_s", "llm_tok_per_s"),
            ("tts_first_chunk_ms", "tts_first_chunk_ms"),
            ("tts_synth_total_ms", "tts_synth_total_ms"),
            ("e2e_first_audio_ms", "e2e_speech_end_to_first_audio_ms"),
            ("response_total_ms", "response_total_ms"),
            ("record_s", "learner_speech_s"),
            ("turn_wall_ms", "turn_wall_ms"),
        ):
            med, n = self._stat(key)
            metrics[f"median_{name}"] = med
            metrics[f"n_{name}"] = n

        summary = {
            "kind": "session_end", "schema": SCHEMA_VERSION,
            "session_id": self.session_id,
            "ended_at": dt.datetime.now().isoformat(timespec="seconds"),
            "end_reason": reason,                 # user_ended | crash | shutdown
            "clean_end": reason == "user_ended",
            "synthetic": self.synthetic,
            "harness": self.harness,
            "duration_min": round(dur / 60, 2),
            "turns": len(self.turns),
            "turns_real": len(real),
            "turns_interrupted": sum(1 for t in self.turns if t.get("interrupted")),

            # --- the product metric -------------------------------------------
            # ⚑ meaningless unless turns_real > 0 and the input was real speech
            "learner_word_ratio": round(lw / (lw + aw), 3) if (lw + aw) else None,
            "learner_words": lw, "ai_words": aw,
            "learner_speech_s": round(ls, 1) if ls else None,
            "ai_speech_s": round(as_, 1) if as_ else None,
            "learner_words_per_turn": round(lw / len(self.turns), 1) if self.turns else None,
            "ai_words_per_turn": round(aw / len(self.turns), 1) if self.turns else None,
            # ⚑ a harness run is not a product measurement either, even though
            # its turns are real microphone-path turns with real audio files.
            "product_metrics_valid": ((not self.synthetic) and self.harness is None
                                      and len(real) > 0),

            # --- latency (median, each with its sample count) ------------------
            **metrics,

            # --- reliability ---------------------------------------------------
            # ⚑ DRIFT WATCH -- observation only, nothing acts on these.
            # Both real 17-turn sessions showed Lena's replies growing and her
            # question rate rising in the second half (26->36 words, question
            # rate 0.12->0.33 in one; 31->35 and 0.50->0.67 in the other). The
            # user has explicitly APPROVED the current response length and
            # style, so no guardrail is applied -- but the trend is now
            # measured every session instead of being rediscovered by hand.
            **self._drift(),
            "sanitizer_removed_total": san,
            "n_stalls": len(stalls), "stalls": stalls,
            "n_server_restarts": self.n_restarts,
            "server_generations": max((t.get("server_generation") or 1)
                                      for t in self.turns) if self.turns else 1,
            "vram_peak_mb": self.vram_peak or None,
            "flags": self.flags,
            "flag_counts": {c: sum(1 for f in self.flags if f["code"] == c)
                            for c in {f["code"] for f in self.flags}},
            "model_status": config.MODEL_STATUS,
            "tts_voice": self.tts_voice,
            "language": self.language,
        }
        self._append(summary)
        return summary

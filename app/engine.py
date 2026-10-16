"""Conversation engine: audio in -> STT -> LLM (streamed) -> TTS -> audio out.

Deliberately small. The pipeline below is UNCHANGED from the version that was
measured and approved; V2 added a teacher on top of it, not a new engine.

⚑ V2 (teacher release). Three things were added, all of them OFF the audio path:
  * the system prompt now comes from app/teaching.py (teacher persona + policy)
    instead of the friend persona in app/persona.json;
  * a per-turn directive from the Conversation Director is inserted as a system
    message immediately BEFORE the learner's last message -- never appended to
    the stored history, never by rewriting messages[0] mid-conversation, because
    both of those cost either correctness (Gemma's template drops a trailing
    system message) or first-token latency (a changed prefix invalidates
    llama-server's prompt cache);
  * app/learner.py counts what the learner did, so difficulty can move
    GRADUALLY and topics do not repeat.
None of this adds an LLM call to the conversation path, and none of it touches
STT, TTS, streaming, half-duplex or containment.

⚑ THE STREAMING IS PRODUCER/CONSUMER AND MUST STAY THAT WAY. Consuming tokens
inside the playback loop makes the pipeline sequential, holds the HTTP
connection open for the whole utterance (risking the 45 s watchdog) and drops
measured throughput from ~77 to ~13 tok/s. That was the original bug.

⚑ RUNTIME CONTAINMENT IS LOAD-BEARING. llama-server can go generation-dead
while /health still returns 200. Do not remove the watchdog, the retry on a
FRESH server, the periodic lifecycle restart, or the in-character filler.
"""
from __future__ import annotations

import json
import queue
import random
import re
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from . import config, interview as interview_mode, teaching, visemes
from .audio_out import SpeechPlayer
from .learner import LearnerModel
from .mistakes import MistakeAnalyzer
from .providers import LlamaCppProvider, StallError, STT, TTS
from .sanitize import merge_counts, sanitize
from .telemetry import SessionLog

# Emit on sentence end, or on a clause boundary once the buffer is long enough.
# Getting the FIRST chunk out early matters more than its prosody; later chunks
# prefer full sentences.
SENTENCE_END = re.compile(r"(?<=[.!?…])\s")
CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s|\s(?=(?:und|aber|weil|oder|denn|also)\s)")
FIRST_CHUNK_MIN = 28
LATER_CHUNK_MIN = 60
# ⚑ A COMPLETE short sentence is a better first chunk than a comma fragment.
# Measured on real streamed replies: the 28-char minimum was skipping PAST
# legitimate sentence ends and landing on the next comma --
#     "Oh, cool." (9 chars)          -> emitted "Oh, cool. Ich hab ... gelaufen,"
#     "Ach, die Drogerie?" (18)      -> emitted "Ach, ... was gekauft,"
# so the first thing the user heard ended on a comma, with Piper's prosody
# resetting mid-thought. Short reactions ARE Lena's register (persona
# voice_notes: "ach so", "echt?", "na ja"), and giving one its own chunk gets
# audio out sooner AND lets it carry its own complete intonation.
# Clause breaks still require FIRST_CHUNK_MIN; only true sentence ends qualify
# this early.
FIRST_SENTENCE_MIN = 8
# ⚑ After the first chunk, WAIT for a sentence end rather than splitting at a
# comma. Piper restarts its prosody contour at every chunk, so a mid-sentence
# split leaves an unnatural falling intonation even though the audio itself is
# now gapless. Measured across the user's two 17-turn real sessions: 35 of 108
# boundaries (32%) fell somewhere other than a sentence end -- 21 at a comma,
# 14 mid-phrase. Those are the remaining "segmented" moments.
#
# A comma split is still allowed once the buffer gets long, otherwise a single
# long German sentence would delay audio indefinitely. The FIRST chunk keeps
# its 28-char rule untouched: first-audio latency is user-approved and must not
# regress.
LATER_CLAUSE_FALLBACK_MIN = 150

MIN_UTTERANCE_S = 0.35


class State:
    """What the user is told is happening. Every one of these is a real,
    distinguishable phase of the loop -- none is cosmetic."""
    INITIALIZING = "initializing"
    READY = "ready"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    RECOVERING = "recovering"
    ERROR = "error"


class Engine:
    def __init__(self, level: str = config.DEFAULT_LEVEL, device=None, on_event=None,
                 synthetic: bool = False, handsfree: bool = False):
        self.level = level if level in config.LEVELS else config.DEFAULT_LEVEL
        self.device = device
        self.on_event = on_event or (lambda *_a, **_k: None)
        self.state = State.INITIALIZING
        self.messages: list[dict] = []
        self.turn = 0
        self.turns_since_restart = 0
        self.synthetic = synthetic
        self._rec_stop = threading.Event()
        self._rec_thread = None
        self._audio_q: queue.Queue = queue.Queue()
        self._level_meter = 0.0
        self._speaking_cancel = threading.Event()
        self._interrupted = False
        # active continuous output stream, so an interrupt can abort it.
        # sd.stop() only affects sd.play(); it does NOT stop an OutputStream.
        self._player: SpeechPlayer | None = None
        # ⚑ one turn at a time. Without this, a fast double release (keyup +
        # mouseup, or an impatient second press) starts two _handle_turn threads
        # against one llama-server slot (-np 1) and one output device.
        self._turn_busy = threading.Lock()
        self._stopping = False
        self._finishing = False
        # hands-free (half-duplex only -- see config HANDSFREE notes)
        self.handsfree = bool(handsfree)
        self._auto_active = False
        self._endpoint_reason = None
        self._meter_lock = threading.Lock()
        self._recent_rms: list[float] = []
        self.log: SessionLog | None = None
        self.mistakes: MistakeAnalyzer | None = None
        # ⚑ V2: the teaching layer. `learner` is counted state (app/learner.py);
        # the director reads it and never writes to the conversation history.
        self.learner: LearnerModel | None = None
        self.interview: interview_mode.InterviewController | None = None
        self.interview_language: str | None = None
        self._rng = random.Random()
        self._prompt_built_at_turn = 0
        self._last_report_path = None
        self.llm = self.stt = self.tts = None
        self.sr = config.RECORD_SR
        self.boot_error: str | None = None

    # ------------------------------------------------------------------ setup
    def _emit(self, kind, **payload):
        self.on_event({"kind": kind, "t": time.time(), **payload})

    def _set_state(self, s):
        self.state = s
        self._emit("state", state=s)

    def _ready(self, rearm: str = ""):
        """Return to READY and, in hands-free mode, reopen the mic.

        Every path that ends a turn goes through here, so there is exactly one
        place where the microphone is allowed to reopen -- and it is never
        reachable while Lena is speaking.
        """
        if self.state == State.LISTENING:
            return                       # user already started the next turn
        self._set_state(State.READY)
        if self.handsfree and not self._stopping:
            self._arm_auto_listen(reason=rearm)

    def start(self):
        """Bring the stack up. Every failure produces a USEFUL message and
        raises -- never an indefinite hang and never a raw traceback in the UI."""
        self._set_state(State.INITIALIZING)
        try:
            self._emit("boot", step="llm", pct=None,
                       detail="Gesprächsmodell wird geladen…")
            self.llm = LlamaCppProvider(config.LLM_MODEL_PATH, config.LLAMA_BUILD_DIR,
                                        config.LLM_PORT, config.LLM_CTX, config.LLM_NGL,
                                        config.LLM_SERVER_ARGS, config.REQ_TIMEOUT_S)
            self.llm.start()

            self._emit("boot", step="stt", detail="Spracherkennung wird geladen…")
            self.stt = STT()

        except FileNotFoundError as e:
            return self._boot_failed(
                "Eine benötigte Datei fehlt.",
                f"{e}", "missing_file")
        except TimeoutError as e:
            return self._boot_failed(
                "Das Gesprächsmodell hat zu lange zum Starten gebraucht.",
                f"{e}", "llm_timeout")
        except RuntimeError as e:
            msg = str(e)
            if "port" in msg.lower():
                return self._boot_failed(
                    "Es läuft bereits eine Instanz (Port belegt).",
                    msg, "port_busy")
            return self._boot_failed(
                "Das Gesprächsmodell konnte nicht starten.", msg, "llm_failed")
        except Exception as e:
            return self._boot_failed(
                "Beim Start ist etwas schiefgegangen.",
                f"{type(e).__name__}: {e}", "unknown")

        # The interview cannot start before the candidate chooses a language.
        # Loading the voice, opening the log and speaking the first question are
        # therefore deferred to start_interview().
        self._emit("ready", level=self.level, model_status=config.MODEL_STATUS,
                   session_id=None, awaiting_language=True,
                   teacher="Sophie", sessions_before=0)
        self._set_state(State.READY)
        return self

    def start_interview(self, language: str,
                        selected_ids: list[str] | None = None,
                        count: int | None = None) -> dict:
        """Configure the chosen language and start exactly one interview."""
        language = interview_mode.valid_language(language)
        if language is None:
            raise ValueError("language must be de or en")
        if self.interview is not None or self.turn:
            return self.interview.snapshot() if self.interview else {"active": False}

        self._set_state(State.INITIALIZING)
        self._emit("interview_starting", language=language)
        self.stt.set_language(language)
        voice = config.TTS_VOICES[language]
        self.tts = TTS(voice)
        self.interview_language = language
        self.interview = interview_mode.InterviewController(
            language, selected_ids=selected_ids, count=count)
        self.log = SessionLog(level=self.level, llm_info=self.llm.info,
                              synthetic=self.synthetic, input_device=self.device,
                              language=language, tts_voice=voice,
                              session_kind="call_center_interview")
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        snap = self.interview.snapshot()
        self._emit("interview_started", **snap, session_id=self.log.session_id)
        self._set_state(State.READY)
        self.speak_opener()
        return snap

    def begin_finishing(self):
        """Mark the interview as finishing: no more recording or turns.

        Called at the start of finalization so that concurrent hands-free
        capture or a late PTT stop cannot start a new turn while the log is
        being closed and the evaluator is running.
        """
        self._finishing = True
        self._speaking_cancel.set()
        try:
            sd.stop()
        except Exception:
            pass
        if self._player is not None:
            self._player.abort()
        # Stop any active recording so it does not try to dispatch a turn
        if self.state == State.LISTENING:
            self._rec_stop.set()

    def reset_interview(self):
        """Reset interview state so a new one can be configured.

        Closes the current session log and clears interview state, but keeps
        the engine (LLM, STT) alive so the user does not wait for a full reboot.
        """
        # Stop any in-progress audio
        self._speaking_cancel.set()
        try:
            sd.stop()
        except Exception:
            pass
        if self._player is not None:
            self._player.abort()

        # Close log for the finished interview
        if self.log and not self.log.closed:
            try:
                self.log.close(self.llm.stalls if self.llm else [], reason="home")
            except Exception:
                pass

        # Clear interview state
        self.interview = None
        self.interview_language = None
        self.messages = []
        self.turn = 0
        self.turns_since_restart = 0
        self._stopping = False
        self._finishing = False
        self._auto_active = False
        self.handsfree = True
        self._last_report_path = None
        self.log = None

        self._set_state(State.READY)
        self._emit("interview_reset")

    def _boot_failed(self, user_msg: str, detail: str, code: str):
        self.boot_error = user_msg
        self._set_state(State.ERROR)
        self._emit("boot_failed", message=user_msg, detail=detail, code=code)
        # tear down anything already up, so a retry finds a clean machine
        try:
            if self.llm:
                self.llm.stop()
        except Exception:
            pass
        raise RuntimeError(f"{code}: {detail}")

    def stop(self, reason: str = "user_ended"):
        """Idempotent shutdown. `reason` separates a clean end from a crash."""
        if self._stopping:
            return
        self._stopping = True
        try:
            # ⚑ best-effort and non-load-bearing: a failed write costs the next
            # lesson some continuity, never this one's data.
            if self.learner:
                self.learner.persist(self.log.session_id if self.log else None)
            if self.mistakes:
                self.mistakes.stop()
            self._speaking_cancel.set()
            try:
                sd.stop()
            except Exception:
                pass
            if self.log and not self.log.closed:
                self.log.close(self.llm.stalls if self.llm else [], reason=reason)
        finally:
            if self.llm:
                self.llm.stop()

    # ----------------------------------------------------------------- prompt
    def _system_prompt(self) -> str:
        """The teacher persona + teaching policy + what she knows about him.

        ⚑ Composed in app/teaching.py, not here. The engine deliberately owns no
        teaching content: swapping the persona, the correction policy or the
        topic catalogue must never mean touching the pipeline.
        """
        if getattr(self, "interview", None) is not None:
            self._prompt_built_at_turn = self.turn
            return interview_mode.system_prompt(self.interview.language)
        state = self.learner.prompt_state() if self.learner else {"level": self.level}
        self._prompt_built_at_turn = self.turn
        return teaching.build_system_prompt(self.level, state)

    def _maybe_refresh_prompt(self):
        """Fold newly learned facts about the learner back into the prompt.

        ⚑ NOT EVERY TURN, ON PURPOSE. `messages[0]` is the prompt PREFIX that
        llama-server caches; changing it forces a full re-evaluation of the
        conversation and shows up directly in first-token latency. Every
        TEACHER_PROMPT_REFRESH_TURNS turns is enough for the state that actually
        belongs in the prompt (recurring weaknesses, structures he now uses,
        topics already practised).
        """
        if getattr(self, "interview", None) is not None:
            return
        if not (self.learner and self.messages):
            return
        if self.turn - self._prompt_built_at_turn < config.TEACHER_PROMPT_REFRESH_TURNS:
            return
        try:
            self.messages[0] = {"role": "system", "content": self._system_prompt()}
            if self.log:
                self.log.event("prompt_refresh", turn=self.turn,
                               difficulty=round(self.learner.difficulty, 2))
        except Exception:
            pass

    def _request_messages(self, directive: str | None) -> list[dict]:
        """Messages for ONE request, with the director's instruction in place.

        ⚑ POSITION IS LOAD-BEARING. The directive goes immediately BEFORE the
        last user message:
          * llama.cpp's Gemma template folds a system message into the FOLLOWING
            user turn, so a trailing system message is silently dropped;
          * putting it in messages[0] instead would change the cached prefix
            every turn.
        It is never stored in `self.messages`, so it cannot accumulate, cannot
        be repeated on later turns and costs nothing in context.
        """
        if not directive or not self.messages:
            return self.messages
        head, last = self.messages[:-1], self.messages[-1]
        if last.get("role") != "user":
            return self.messages
        return head + [{"role": "system", "content": directive}, last]

    def _on_mistake(self, rec: dict):
        """Mistakes Panel -> learner model -> UI. The model is what makes a
        recurring error influence LATER teaching instead of just being listed."""
        try:
            if self.learner:
                self.learner.observe_mistake(rec)
        except Exception:
            pass
        self._emit("mistake", **rec)

    def _on_vocab(self, card: dict):
        try:
            if self.learner:
                self.learner.observe_vocab(card)
        except Exception:
            pass
        self._emit("vocab", **card)

    def set_level(self, level: str) -> bool:
        """Live level change. Rewrites the system prompt in place so the rest of
        the conversation is preserved -- changing level must not reset context."""
        if level not in config.LEVELS or level == self.level:
            return False
        self.level = level
        if self.learner:
            self.learner.level = level
        if self.messages:
            self.messages[0] = {"role": "system", "content": self._system_prompt()}
        if self.log:
            self.log.event("level_change", level=level, turn=self.turn)
        return True

    # ------------------------------------------------------------------ audio
    def start_recording(self, auto: bool = False, reason: str = ""):
        """Begin capture. Returns False if the app is not in a state to listen.

        Interrupting Lena mid-sentence IS allowed for a MANUAL press -- it is a
        deliberate user-initiated stop, not barge-in (no talking over audio).
        The interrupted turn is MARKED so its truncated text is never mistaken
        for a complete reply.

        ⚑ An AUTOMATIC open never interrupts: hands-free is strictly
        half-duplex, so the mic must not open while Lena is speaking.
        """
        if self.interview is None or self._finishing:
            return False
        if self.state == State.LISTENING:
            return False                      # already recording; ignore repeat
        if auto and self.state != State.READY:
            return False                      # never auto-open over playback
        if self.state not in (State.READY, State.SPEAKING):
            return False                      # thinking/transcribing/booting

        if self.state == State.SPEAKING:
            self._interrupted = True
            self._speaking_cancel.set()
            # abort the continuous stream, then sd.stop() for the opener/filler
            # paths that still use sd.play()
            if self._player is not None:
                self._player.abort()
            try:
                sd.stop()
            except Exception:
                pass

        self._rec_stop.clear()
        self._audio_q = queue.Queue()
        self._auto_active = bool(auto)
        self._endpoint_reason = None
        with self._meter_lock:
            self._recent_rms = []
        self._set_state(State.LISTENING)
        self._emit("listening", auto=bool(auto), reason=reason)
        self._t_rec_start = time.perf_counter()

        def cb(indata, frames, t, status):
            self._audio_q.put(indata.copy())
            self._level_meter = float(np.abs(indata).max())
            # feed the endpointer the same frame-energy measure the speech gate
            # uses, so listening and gating can never disagree
            if self._auto_active:
                block = indata[:, 0]
                n = int(0.02 * self.sr)
                if block.size >= n:
                    f = block[:block.size // n * n].reshape(-1, n)
                    rms = np.sqrt((f.astype(np.float64) ** 2).mean(axis=1))
                    with self._meter_lock:
                        self._recent_rms.extend(rms.tolist())

        def run():
            try:
                with sd.InputStream(samplerate=self.sr, channels=1, device=self.device,
                                    dtype="float32", callback=cb, blocksize=1024):
                    while not self._rec_stop.is_set():
                        time.sleep(0.03)
                        self._emit("level_meter", value=round(self._level_meter, 4))
            except Exception as e:
                self._emit("mic_error",
                           message=("The microphone could not be opened."
                                    if self.interview_language == "en"
                                    else "Das Mikrofon konnte nicht geöffnet werden."),
                           detail=str(e))
                self._set_state(State.READY)

        self._rec_thread = threading.Thread(target=run, daemon=True)
        self._rec_thread.start()
        if auto:
            threading.Thread(target=self._auto_listen_loop, daemon=True).start()
        return True

    # ---------------------------------------------------------- hands-free
    def set_handsfree(self, on: bool):
        """Switch between automatic turn-taking and push-to-talk."""
        on = bool(on)
        if on == self.handsfree:
            return self.handsfree
        self.handsfree = on
        if self.log:
            self.log.event("handsfree_mode", enabled=on, turn=self.turn)
        self._emit("handsfree", enabled=on)
        if on and self.state == State.READY:
            self._arm_auto_listen(reason="mode_on")
        elif not on and self.state == State.LISTENING and self._auto_active:
            self.stop_recording()
        return self.handsfree

    def _arm_auto_listen(self, reason: str = ""):
        """Open the mic after a guard delay. Never while Lena is speaking."""
        if not self.handsfree or self._stopping:
            return
        if self.state not in (State.READY,):
            return

        def go():
            # ⚑ guard: let the tail of Lena's audio and the room decay before
            # the mic opens. Measured echo is only one frame below the speech
            # gate, so recording during or immediately after playback risks
            # transcribing Lena as the learner.
            time.sleep(config.LISTEN_START_GUARD_S)
            if self.handsfree and self.state == State.READY and not self._stopping:
                self.start_recording(auto=True, reason=reason)

        threading.Thread(target=go, daemon=True).start()

    def _auto_listen_loop(self):
        """Energy-based endpointing. Runs only while the mic is open.

        This is NOT a VAD model: it is the same frame-energy measure the speech
        gate already uses, which was calibrated against this machine's real
        noise floor and this learner's real speech. A neural VAD would add a
        dependency and a failure mode without solving the thing that actually
        matters here -- tolerating very long word-search pauses.
        """
        sr = self.sr
        frame_s = 0.02
        started_at = time.perf_counter()
        last_voiced = None
        voiced_total = 0.0
        speech_started = False
        endpoint_reason = None

        while not self._rec_stop.is_set():
            time.sleep(0.03)
            now = time.perf_counter()
            with self._meter_lock:
                recent = self._recent_rms
                self._recent_rms = []
            for rms in recent:
                if rms > config.SPEECH_RMS_FLOOR:
                    voiced_total += frame_s
                    last_voiced = now
                    if not speech_started and voiced_total >= config.ENDPOINT_MIN_SPEECH_S:
                        speech_started = True
                        self._emit("speech_detected")

            if speech_started:
                if last_voiced is not None and (now - last_voiced) >= config.ENDPOINT_SILENCE_S:
                    endpoint_reason = "silence"
                    break
                if (now - started_at) >= config.ENDPOINT_MAX_UTTERANCE_S:
                    endpoint_reason = "max_utterance"
                    break
            else:
                if (now - started_at) >= config.ENDPOINT_NO_SPEECH_TIMEOUT_S:
                    endpoint_reason = "no_speech_timeout"
                    break

        if endpoint_reason is None:
            return                                  # stopped manually
        self._endpoint_reason = endpoint_reason
        if endpoint_reason == "no_speech_timeout":
            # nothing was said: close the mic, do not invent a turn
            self._rec_stop.set()
            if self._rec_thread:
                self._rec_thread.join(timeout=2)
            self._drain_audio()
            self._auto_active = False
            if self.log:
                self.log.event("auto_listen_timeout",
                               waited_s=round(time.perf_counter() - started_at, 1))
            self._emit("notice", text=("I cannot hear anything. Speak when you are ready."
                                       if self.interview_language == "en"
                                       else "Ich höre nichts — sprechen Sie, wenn Sie bereit sind."))
            self._set_state(State.READY)
            return
        self.stop_recording(endpoint=endpoint_reason,
                            voiced_s_est=round(voiced_total, 2))

    def _trim_trailing_silence(self, audio: np.ndarray) -> np.ndarray:
        """Drop only silence that lies well beyond the last detected speech.

        Never cuts before `last voiced frame + SPEECH_POSTROLL_S`, so word
        endings survive even when the final syllable is quieter than the gate
        floor -- which German endings (-n, -en, -em, -er, ch/sch/s/f/h) very
        often are.
        """
        n = int(0.02 * self.sr)
        if audio.size < n * 2:
            return audio
        f = audio[:audio.size // n * n].reshape(-1, n)
        rms = np.sqrt((f.astype(np.float64) ** 2).mean(axis=1))
        idx = np.flatnonzero(rms > config.SPEECH_RMS_FLOOR)
        if idx.size == 0:
            return audio
        keep = int((idx[-1] + 1) * n + config.SPEECH_POSTROLL_S * self.sr)
        keep = min(audio.size, keep)
        # only bother if it saves real time; otherwise leave it alone
        if audio.size - keep < int(config.SPEECH_TRIM_MIN_SAVING_S * self.sr):
            return audio
        return audio[:keep]

    def _drain_audio(self):
        chunks = []
        while not self._audio_q.empty():
            chunks.append(self._audio_q.get())
        return (np.concatenate(chunks, axis=0)[:, 0] if chunks
                else np.zeros(0, np.float32))

    def stop_recording(self, endpoint: str = "manual", voiced_s_est=None):
        """End capture and dispatch the turn. Idempotent."""
        if self.state != State.LISTENING:
            return False
        self._rec_stop.set()
        if self._rec_thread:
            self._rec_thread.join(timeout=2)
        record_s = time.perf_counter() - getattr(self, "_t_rec_start", time.perf_counter())
        was_auto = self._auto_active
        self._auto_active = False
        audio = self._drain_audio()

        # ⚑ TRIMMING IS ANCHORED TO DETECTED SPEECH, NEVER TO A FIXED DURATION.
        #
        # The previous version removed a flat ENDPOINT_SILENCE_S * 0.8 = 2.0 s
        # from the end. Measured against the user's 17 real recordings (E8),
        # that CHANGED 14 OF 17 TRANSCRIPTS, and not subtly:
        #     "Ich habe jetzt ins Bett gehen." -> "Ich habe jetzt..."
        #     "Trinken Sie viel Kaffee?"       -> "Drinken Sie"
        #     "Wie heute sind du?"             -> "Vielen Dank."
        # Destroying word endings is the worst possible failure here, because
        # the Mistakes Panel judges exactly that morphology and would "correct"
        # a mistake the learner never made.
        #
        # Now: find the last frame above the speech floor and keep a generous
        # post-roll after it. Measured quiet tail beyond the detected end on
        # real learner audio is median 0.04 s (max 2.86 s, and that longest case
        # is room tone, not speech), so a 1.2 s post-roll sits far above the
        # realistic worst case. If that would not save meaningful silence, we
        # keep the audio untouched -- Whisper coping with a little extra room
        # tone is strictly safer than losing a final -n.
        if (config.SPEECH_TRIM_ENABLED and was_auto
                and endpoint == "silence" and audio.size):
            audio = self._trim_trailing_silence(audio)

        self._set_state(State.TRANSCRIBING)
        threading.Thread(target=self._handle_turn,
                         args=(audio, record_s),
                         kwargs={"endpoint": endpoint if was_auto else "manual",
                                 "auto": was_auto, "voiced_s_est": voiced_s_est},
                         daemon=True).start()
        return True

    # ------------------------------------------------------------------- turn
    @staticmethod
    def voiced_seconds(audio: np.ndarray, sr: int) -> float:
        """How much of this recording is plausibly speech, by frame energy.

        Cheap, deterministic, and model-free -- this runs BEFORE Whisper both to
        stop silence hallucinations and to avoid paying ~460 ms to transcribe
        nothing. It is not endpointing: it never decides when to stop recording,
        only whether what was recorded contains anything at all.
        """
        n = int(0.02 * sr)
        if audio.size < n:
            return 0.0
        frames = audio[:audio.size // n * n].reshape(-1, n)
        rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
        return float((rms > config.SPEECH_RMS_FLOOR).sum() * 0.02)

    def _handle_turn(self, audio: np.ndarray, record_s: float,
                     endpoint: str = "manual", auto: bool = False,
                     voiced_s_est=None):
        # ⚑ single-flight: a second turn must never race the first
        if self._finishing:
            return
        if not self._turn_busy.acquire(blocking=False):
            self._emit("notice", text=("One moment…" if self.interview_language == "en"
                                       else "Einen Moment noch…"))
            return
        t_turn = time.perf_counter()
        try:
            if audio.size < self.sr * MIN_UTTERANCE_S:
                self._emit("notice", text=("That was too short. Please try again."
                                           if self.interview_language == "en"
                                           else "Zu kurz — bitte nochmal."))
                self._ready("too_short")
                return

            # ⚑ energy gate BEFORE Whisper: silence in, nothing out. Without
            # this, a quiet room transcribes as "Vielen Dank." and Lena answers
            # a sentence that was never spoken.
            voiced_s = self.voiced_seconds(audio, self.sr)
            if voiced_s < config.SPEECH_MIN_VOICED_S:
                self._emit("notice", text=("I could not hear an answer. Please try again."
                                           if self.interview_language == "en"
                                           else "Ich habe keine Antwort gehört — bitte nochmal."))
                if self.log:
                    self.log.event("empty_input", turn=self.turn,
                                   voiced_s=round(voiced_s, 3),
                                   record_s=round(record_s, 2))
                self._ready("empty_input")
                return

            learner_s = len(audio) / self.sr
            r = self.stt.transcribe(audio)
            text = r["text"]
            lp = r.get("avg_logprob")
            unreliable = lp is not None and lp < config.STT_MIN_AVG_LOGPROB
            if not text or len(text) < 2 or (r["no_speech_prob"] or 0) > 0.6 or unreliable:
                # Do NOT call the LLM on nothing, and never fabricate an AI turn.
                self._emit("notice", text=("I could not understand that. Please try again."
                                           if self.interview_language == "en"
                                           else "Ich habe Sie nicht verstanden — bitte nochmal."))
                if self.log:
                    self.log.event("rejected_input", turn=self.turn,
                                   text=text, avg_logprob=lp,
                                   no_speech_prob=r.get("no_speech_prob"),
                                   voiced_s=round(voiced_s, 3),
                                   reason="low_confidence" if unreliable else "no_speech")
                self._ready("rejected")
                return

            self.turn += 1
            wav_name = None
            try:
                wav = config.AUDIO_DIR / f"{self.log.session_id}-t{self.turn:03d}.wav"
                sf.write(wav, audio, self.sr)
                wav_name = wav.name
            except Exception:
                pass                          # audio keeping is best-effort
            self._emit("transcript", role="user", text=text, turn=self.turn)

            # Low confidence no longer blocks the turn (that regression made the
            # user repeat themselves). It is recorded so misheard-diagnostics
            # and E2 can find these turns later.
            low_conf = lp is not None and lp < config.STT_LOW_CONFIDENCE_LOGPROB
            # ⚑ V2: counted learner state, updated BEFORE the director runs, so
            # a short answer is expanded on the very turn it happens.
            if self.learner:
                try:
                    self.learner.observe_learner_turn(
                        text, voiced_s=voiced_s, pauses=r.get("n_pauses"),
                        low_conf=low_conf)
                except Exception:
                    pass
            interview_answer = None
            if self.interview is not None:
                interview_answer = self.interview.record_answer(text)
            self.messages.append({"role": "user", "content": text})
            self._speak_streamed(t_turn, text, learner_s, r, wav_name, record_s,
                                 low_conf=low_conf, voiced_s=voiced_s,
                                 endpoint=endpoint, auto=auto,
                                 voiced_s_est=voiced_s_est,
                                 interview_answer=interview_answer)
        except Exception as e:
            self._emit("turn_error",
                       message=("Something went wrong — please try again."
                                if self.interview_language == "en"
                                else "Da ist etwas schiefgelaufen — sag es nochmal?"),
                       detail=f"{type(e).__name__}: {e}")
            if self.log:
                self.log.event("turn_error", turn=self.turn,
                               detail=f"{type(e).__name__}: {e}")
            self._ready("turn_error")
        finally:
            self._turn_busy.release()

    def _chunks(self, buf: str, first: bool) -> tuple[list[str], str]:
        """Split off speakable chunks, return (chunks, remainder)."""
        out = []
        minlen = FIRST_CHUNK_MIN if first else LATER_CHUNK_MIN
        is_first = first
        while True:
            m = None
            # A sentence end is always preferred. A clause break is accepted
            # immediately for the FIRST chunk (latency), but for later chunks
            # only once the buffer is long enough that waiting would stall.
            for pat in (SENTENCE_END, CLAUSE_BREAK):
                if pat is CLAUSE_BREAK and not is_first \
                        and len(buf) < LATER_CLAUSE_FALLBACK_MIN:
                    continue
                # a complete short sentence is an acceptable FIRST chunk
                need = (FIRST_SENTENCE_MIN
                        if (pat is SENTENCE_END and is_first) else minlen)
                for cand in pat.finditer(buf):
                    if cand.end() >= need:
                        m = cand
                        break
                if m:
                    break
            if not m:
                break
            is_first = False
            out.append(buf[:m.end()].strip())
            buf = buf[m.end():]
            minlen = LATER_CHUNK_MIN
        return out, buf

    def _speak_streamed(self, t_turn, learner_text, learner_s, stt_r, wav_name,
                        record_s, low_conf: bool = False, voiced_s=None,
                        endpoint: str = "manual", auto: bool = False,
                        voiced_s_est=None, interview_answer=None):
        lv = config.LEVELS[self.level]
        # ⚑ within-level speaking rate. `length_scale` is Piper's DURATION
        # multiplier (higher = slower); the per-level table stays frozen and this
        # only nudges +-6% around it. See learner.speech_scale_factor.
        speech_scale = (1.0 if self.interview is not None else round(
            lv["length_scale"] * (self.learner.speech_scale_factor()
                                  if self.learner else 1.0), 3))
        # ⚑ ONE decision, from counted state, no extra model call. See
        # teaching.directive and _request_messages for why it is placed where it
        # is placed.
        self._maybe_refresh_prompt()
        directive_text, directive_meta = (None, {"kind": None})
        fixed_response = None
        if self.interview is not None:
            fixed_response, directive_meta = self.interview.response_after_answer()
        elif self.learner:
            try:
                directive_text, directive_meta = teaching.directive(
                    self.learner,
                    self.mistakes.items if self.mistakes else [],
                    self._rng)
            except Exception as _de:
                if self.log:
                    self.log.event("director_error",
                                   detail=f"{type(_de).__name__}: {_de}")
        req_messages = (self._request_messages(directive_text)
                        if fixed_response is None else [])
        raw_parts, removed_total = [], {}
        spoken_parts = []
        # ⚑ Cumulative audio ALREADY handed to the speaker, in ms. This is the
        # clock the UI's mouth animation runs on: the SSE event for a chunk
        # arrives while earlier audio is still playing, so the UI schedules from
        # (first_audio + audio_at_ms) rather than from arrival time.
        audio_at_ms = 0.0
        # ⚑ every timing starts as None. An unmeasured stage stays null.
        first_audio_ms = None
        first_token_ms = None
        tts_first_chunk_ms = None
        tts_synth_total_ms = 0.0
        n_tts_calls = 0
        gen_meta = {}
        n_retries = 0
        play_stats: dict = {}
        self._speaking_cancel.clear()
        self._interrupted = False

        # THINKING, not SPEAKING: nothing has been said yet. Showing "speaking"
        # here (the old behaviour) told the user Lena was talking during the
        # entire LLM latency, which is the most visible part of the wait.
        self._set_state(State.THINKING)

        # ── PRODUCER / CONSUMER ────────────────────────────────────────────
        # Producer drains the LLM stream at full speed into a queue; consumer
        # synthesises and plays. By the time chunk 1 finishes playing, the rest
        # of the reply has usually already been generated.
        max_attempts = 1 if fixed_response is not None else config.MAX_TURN_RETRIES
        for attempt in range(1, max_attempts + 1):
            chunk_q: queue.Queue = queue.Queue()
            producer_err: list[Exception] = []

            def produce():
                nonlocal first_token_ms, gen_meta
                buf, first = "", True
                try:
                    # Interview questions are curated application content. Speak
                    # the exact full question as one utterance: no model drift,
                    # no generation stall and no comma split that restarts TTS
                    # prosody in the middle of a question.
                    if fixed_response is not None:
                        gen_meta = {"source": "deterministic_interview"}
                        chunk_q.put(fixed_response)
                        return
                    for delta, meta in self.llm.stream_chat(
                            req_messages, max_tokens=lv["max_words"] * 3):
                        if meta.get("first_token_ms") and first_token_ms is None:
                            first_token_ms = meta["first_token_ms"]
                        if meta.get("tokens"):
                            gen_meta = meta
                        if not delta:
                            continue
                        buf += delta
                        chunks, buf = self._chunks(buf, first)
                        for c in chunks:
                            first = False
                            chunk_q.put(c)
                        if self._speaking_cancel.is_set():
                            break
                    if buf.strip():
                        chunk_q.put(buf.strip())
                except Exception as ex:
                    producer_err.append(ex)
                finally:
                    chunk_q.put(None)          # sentinel

            # ── SYNTH-AHEAD WORKER ─────────────────────────────────────────
            # Synthesis must NOT sit between two playback calls. E4 measured the
            # old inline arrangement: every inter-chunk silence equalled the next
            # chunk's synthesis time (median 149.9 ms). This thread turns text
            # into PCM while the previous chunk is still playing.
            pcm_q: queue.Queue = queue.Queue(maxsize=8)
            synth_err: list[Exception] = []

            def synthesize():
                nonlocal tts_first_chunk_ms, tts_synth_total_ms, n_tts_calls
                try:
                    while True:
                        c = chunk_q.get()
                        if c is None:
                            break
                        if self._speaking_cancel.is_set():
                            continue          # drain, do not synthesise
                        raw_parts.append(c)
                        clean, rm = sanitize(c)
                        merged = merge_counts(removed_total, rm)
                        removed_total.clear()
                        removed_total.update(merged)
                        if not clean:
                            continue
                        t_syn = time.perf_counter()
                        pcm, _fa = self.tts.synth(clean, speech_scale)
                        syn_ms = (time.perf_counter() - t_syn) * 1000
                        tts_synth_total_ms += syn_ms
                        n_tts_calls += 1
                        if tts_first_chunk_ms is None:
                            tts_first_chunk_ms = round(syn_ms, 1)
                        pcm_q.put((pcm, clean, syn_ms))
                except Exception as ex:
                    synth_err.append(ex)
                finally:
                    pcm_q.put(None)

            try:
                prod = threading.Thread(target=produce, daemon=True)
                prod.start()
                syn = threading.Thread(target=synthesize, daemon=True)
                syn.start()

                player = SpeechPlayer(self.tts.sr)
                self._player = player
                try:
                    player.open()
                    while True:
                        item = pcm_q.get()
                        if item is None:
                            break
                        pcm, clean, syn_ms = item
                        if self._speaking_cancel.is_set():
                            break
                        if first_audio_ms is None:
                            first_audio_ms = round((time.perf_counter() - t_turn) * 1000, 1)
                            self._set_state(State.SPEAKING)
                            self._emit("first_audio", ms=first_audio_ms)
                        spoken_parts.append(clean)
                        self._emit("transcript", role=("interviewer" if self.interview else "lena"), text=clean,
                                   turn=self.turn, partial=True)
                        # ⚑ Mouth shapes for exactly this chunk, from its text
                        # and its REAL duration. No audio leaves Python, no GPU
                        # is touched, and a failure here can never stop speech.
                        if config.VISEMES_ENABLED:
                            try:
                                dur_ms = len(pcm) / self.tts.sr * 1000.0
                                self._emit("speak_chunk",
                                           at_ms=round(audio_at_ms, 1),
                                           dur_ms=round(dur_ms, 1),
                                           visemes=visemes.timeline(clean, dur_ms))
                                audio_at_ms += dur_ms
                            except Exception:
                                pass
                        if not player.write(pcm, text=clean, synth_ms=syn_ms):
                            break
                    if self._speaking_cancel.is_set():
                        player.abort()
                finally:
                    play_stats = player.stats
                    player.close()
                    self._player = None

                prod.join(timeout=2)
                syn.join(timeout=2)
                if producer_err:
                    raise producer_err[0]
                if synth_err:
                    raise synth_err[0]
                break
            except StallError as e:
                # Never a silent frozen conversation: tell the user in product
                # language, restart on a FRESH server, retry.
                n_retries += 1
                self._set_state(State.RECOVERING)
                self._emit("recovering", message="Kurzer Moment…", attempt=attempt)
                if self.log:
                    self.log.event("stall", turn=self.turn, attempt=attempt,
                                   detail=str(e))
                if attempt < max_attempts:
                    try:
                        self.llm.restart()
                        self.turns_since_restart = 0
                        if self.log:
                            self.log.event("server_restart",
                                           generation=self.llm.generation,
                                           cause="stall")
                    except Exception as re_:
                        self._emit("turn_error",
                                   message=("The connection was lost."
                                            if self.interview_language == "en"
                                            else "Die Verbindung zu Lena ist abgebrochen."),
                                   detail=str(re_))
                        break
                else:
                    filler = ("Sorry, I lost my place for a moment. Please repeat that."
                              if self.interview_language == "en"
                              else "Entschuldigung, ich war kurz abgelenkt. Bitte wiederholen Sie das.")
                    try:
                        pcm, _ = self.tts.synth(filler, speech_scale)
                        spoken_parts.append(filler)
                        self._set_state(State.SPEAKING)
                        self._emit("transcript", role=("interviewer" if self.interview else "lena"), text=filler,
                                   turn=self.turn)
                        sd.play(pcm, self.tts.sr)
                        sd.wait()
                    except Exception:
                        pass

        raw = " ".join(raw_parts).strip()
        spoken = " ".join(spoken_parts).strip()
        if raw:
            self.messages.append({"role": "assistant", "content": raw})

        interrupted = self._interrupted
        ai_s = sum(len(p.split()) for p in spoken_parts) / 2.5  # derived, rough
        self.log.turn({
            "turn": self.turn,
            "input_source": "synthetic" if self.synthetic else "microphone",
            "interrupted": interrupted,
            # ── hands-free (E5) ───────────────────────────────────────────
            "handsfree": bool(auto),
            "endpoint_reason": endpoint,
            "endpoint_voiced_s_est": voiced_s_est,
            "learner_text": learner_text,
            "learner_speech_s": round(learner_s, 2) if learner_s else None,
            "record_wall_s": round(record_s, 2) if record_s else None,
            "learner_words": len(learner_text.split()),
            # --- measured -------------------------------------------------
            "stt_latency_ms": stt_r.get("latency_ms"),
            "stt_avg_logprob": stt_r.get("avg_logprob"),
            "stt_low_confidence": low_conf,
            "stt_language_probability": stt_r.get("language_probability"),
            "stt_words": stt_r.get("words"),
            "stt_beam_size": config.STT_BEAM_SIZE,
            "stt_pauses": stt_r.get("n_pauses"),
            "stt_pause_s": stt_r.get("total_pause_s"),
            "voiced_s": round(voiced_s, 3) if voiced_s is not None else None,
            "llm_first_token_ms": first_token_ms,
            "llm_tok_per_s": gen_meta.get("tok_per_s"),
            "llm_tokens": gen_meta.get("tokens"),
            "response_source": gen_meta.get("source", "language_model"),
            "tts_first_chunk_ms": tts_first_chunk_ms,
            "tts_synth_total_ms": round(tts_synth_total_ms, 1) if n_tts_calls else None,
            "tts_calls": n_tts_calls,
            # ── playback continuity (E4) ──────────────────────────────────
            # n_underflows is the device reporting real audible holes. It must
            # stay 0; anything above 0 means the chopping has regressed.
            "playback_underflows": play_stats.get("n_underflows"),
            "playback_gap_ms_max": play_stats.get("gap_ms_max"),
            "playback_gap_ms_median": play_stats.get("gap_ms_median"),
            "playback_audio_ms_total": play_stats.get("audio_ms_total"),
            "playback_chunks": play_stats.get("chunks"),
            "e2e_speech_end_to_first_audio_ms": first_audio_ms,
            "response_total_ms": round((time.perf_counter() - t_turn) * 1000, 1),
            "turn_wall_ms": round((time.perf_counter() - t_turn) * 1000, 1),
            # --- derived / bookkeeping ------------------------------------
            "ai_text_raw": raw, "ai_text_spoken": spoken,
            "ai_words": len(spoken.split()),
            "ai_speech_s_estimated": round(ai_s, 2),
            "sanitizer_removed": removed_total,
            # ── teaching (V2) ─────────────────────────────────────────────
            # What the director asked for on THIS turn, and the counted state it
            # asked from. Recorded so a teaching regression is diagnosable from
            # the session file alone, exactly like a latency regression.
            "teacher_directive": directive_meta.get("kind"),
            "teacher_directive_detail": {k: v for k, v in directive_meta.items()
                                         if k != "kind"} or None,
            "teacher_phase": self.learner.phase if self.learner else None,
            "teacher_difficulty": (round(self.learner.difficulty, 3)
                                   if self.learner else None),
            "teacher_topic": self.learner.topic_id if self.learner else None,
            # Exact deterministic mapping used by the interview report.
            "interview_language": self.interview_language,
            "interview_question_id": (interview_answer or {}).get("id"),
            "interview_question_number": (interview_answer or {}).get("number"),
            "interview_question_category": (interview_answer or {}).get("category"),
            "interview_question": (interview_answer or {}).get("text"),
            "interview_competency": (interview_answer or {}).get("competency"),
            "tts_length_scale": speech_scale,
            "n_retries": n_retries,
            "server_generation": self.llm.generation,
            "audio_file": wav_name,
        })
        # ── V2 teaching follow-up. All of it AFTER the audio, none of it on
        # the critical path, and every piece of it failure-tolerant.
        if self.learner and spoken:
            try:
                self.learner.observe_teacher_turn(spoken)
                offered = teaching.detect_vocab_offer(spoken)
                if (self.mistakes and config.VOCAB_ENABLED and not interrupted
                        and (offered or self.learner.turns_since_vocab
                             >= config.VOCAB_PROBE_EVERY_TURNS)):
                    self.mistakes.submit_vocab(
                        turn=self.turn, learner_text=learner_text,
                        teacher_text=spoken, offered=offered,
                        level=self.level,
                        known=self.learner.profile.known_words)
                    self.learner.mark_vocab_done()
            except Exception:
                pass
            self._emit("progress", **self.learner.snapshot())
            # ⚑ A CEFR change is never automatic. The dial has been at the
            # ceiling long enough that the next level is worth OFFERING; the
            # learner decides, and the offer is made at most once per session.
            try:
                if self.learner.wants_level_up():
                    nxt = {"A2": "B1", "B1": "B2"}.get(self.level)
                    if nxt:
                        self._emit("level_suggestion", level=nxt,
                                   current=self.level)
                        if self.log:
                            self.log.event("level_suggested", suggested=nxt,
                                           current=self.level, turn=self.turn)
            except Exception:
                pass

        if self.interview is not None:
            self._emit("interview_progress", **self.interview.snapshot())

        self._emit("turn_done", turn=self.turn, e2e_ms=first_audio_ms,
                   interrupted=interrupted)

        # ⚑ AFTER the reply has been spoken, never before. The analyzer also
        # waits for the LLM to go idle, so it cannot sit in front of the
        # learner's next turn on the single -np 1 slot.
        if self.mistakes and not interrupted:
            self.mistakes.submit(turn=self.turn, text=learner_text,
                                 avg_logprob=stt_r.get("avg_logprob"),
                                 low_conf=low_conf, audio_file=wav_name,
                                 level=self.level)

        # periodic lifecycle restart -- containment, not a fix
        self.turns_since_restart += 1
        if self.turns_since_restart >= config.RESTART_EVERY_TURNS:
            self._set_state(State.RECOVERING)
            try:
                self.llm.restart()
                self.turns_since_restart = 0
                if self.log:
                    self.log.event("server_restart", generation=self.llm.generation,
                                   cause="periodic")
                self._emit("server_restart", generation=self.llm.generation)
            except Exception as e:
                self._emit("turn_error",
                           message="Lena musste neu starten und kam nicht zurück.",
                           detail=str(e))
                self._set_state(State.ERROR)
                return

        if self.interview is not None and self.interview.complete:
            self.handsfree = False
            self._set_state(State.READY)
            self._emit("interview_complete", **self.interview.snapshot())
            return

        # If the user interrupted, they are already holding PTT -- do not stomp
        # the LISTENING state they just entered. Otherwise hand back to READY,
        # which reopens the mic when hands-free is on.
        self._ready("after_lena")

    # --------------------------------------------------------------- opener
    def speak_opener(self):
        """First thing the learner hears.

        ⚑ Deterministic, assembled from teacher.json -- no LLM call at startup.
        A generated opener would put a full generation in front of the very
        first "ready", which is the one latency the learner always notices.
        """
        lv = config.LEVELS[self.level]
        try:
            if self.interview is not None:
                opener = self.interview.opener()
            else:
                opener = teaching.opener(
                    self.learner.prompt_state() if self.learner else {}, self._rng)
        except Exception:
            opener = ("Good to meet you. Please introduce yourself briefly."
                      if self.interview_language == "en" else
                      "Guten Tag. Stellen Sie sich bitte kurz vor.")
        self.messages.append({"role": "assistant", "content": opener})
        self._emit("transcript", role=("interviewer" if self.interview else "lena"),
                   text=opener, turn=0)
        self._speaking_cancel.clear()
        self._set_state(State.SPEAKING)
        try:
            scale = (1.0 if self.interview is not None else round(
                lv["length_scale"] * (self.learner.speech_scale_factor()
                                      if self.learner else 1.0), 3))
            pcm, _ = self.tts.synth(opener, scale)
            if not self._speaking_cancel.is_set():
                sd.play(pcm, self.tts.sr)
                sd.wait()
        except Exception as e:
            self._emit("turn_error",
                       message=("The voice could not play."
                                if self.interview_language == "en"
                                else "Die Stimme konnte nicht abgespielt werden."),
                       detail=str(e))
        self._ready("after_opener")

    # -------------------------------------------------------------- feedback
    def dispute_transcript(self, turn: int | None = None) -> dict | None:
        """"Das habe ich nicht gesagt." One keypress, no typing.

        Marks a turn ASR_DISPUTED_BY_USER and pins every piece of evidence
        needed to adjudicate it later: the captured audio file, exactly what
        Whisper returned, its confidence, the word timings, and any Mistakes
        Panel cards that were generated FROM that transcript -- because a
        disputed transcript means those corrections are suspect too.

        These turns are the highest-value dataset we have for E2 and for any
        future STT decision, and they cost the user one keystroke mid-conversation.
        """
        if not self.log:
            return None
        t = turn or self.turn
        rec = next((x for x in reversed(self.log.turns) if x.get("turn") == t), None)
        if rec is None:
            return None
        cards = []
        if self.mistakes:
            cards = [dict(m) for m in self.mistakes.items
                     if t in (m.get("turns") or [])]
        payload = {
            "turn": t,
            "status": "ASR_DISPUTED_BY_USER",
            "heard_text": rec.get("learner_text"),
            "audio_file": rec.get("audio_file"),
            "stt_avg_logprob": rec.get("stt_avg_logprob"),
            "stt_low_confidence": rec.get("stt_low_confidence"),
            "stt_language_probability": rec.get("stt_language_probability"),
            "stt_words": rec.get("stt_words"),
            "voiced_s": rec.get("voiced_s"),
            "record_wall_s": rec.get("record_wall_s"),
            "endpoint_reason": rec.get("endpoint_reason"),
            "handsfree": rec.get("handsfree"),
            "beam_size": config.STT_BEAM_SIZE,
            "compute_type": config.STT_COMPUTE,
            "trim_enabled": config.SPEECH_TRIM_ENABLED,
            "mistakes_from_this_turn": cards,
        }
        self.log.event("asr_disputed", **payload)
        rec["asr_disputed"] = True
        self._emit("disputed", turn=t, heard=rec.get("learner_text"))
        return payload

    def flag(self, code: str, note: str = ""):
        if not self.log:
            return
        self.log.flag(turn=self.turn, code=code, note=note)
        self._emit("flagged", code=code, turn=self.turn)

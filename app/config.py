"""MVP configuration. Everything the pipeline depends on, in one place.

⚑ MODEL STATUS IS PROVISIONAL. Gemma 4 E4B was selected from E0 + E1 Stage A
evidence only. B1 (blind transcript rating) and B2 (blind live conversation)
were deferred by user decision, and no native speaker has reviewed its German.
An observed counter-example is recorded in KNOWN_RISKS below.

Do not treat this as a final model choice and do not optimise the app around
Gemma-specific behaviour. The provider interface exists so the LLM can be
swapped without touching the engine.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
# ⚑ WHERE THE BIG ASSETS LIVE. In the development tree this stays `lab/`, so
# nothing about the developer flow changes. The INSTALLED application sets
# LENA_ASSET_ROOT to %LOCALAPPDATA%\Lena, because Program Files is read-only
# after installation and first-run provisioning must not require admin. The
# layout underneath is identical either way: <root>/models and <root>/bin.
LAB = Path(os.environ["LENA_ASSET_ROOT"]) if os.environ.get("LENA_ASSET_ROOT") \
    else ROOT / "lab"
# ⚑ USER DATA IS NOT MOVED AND NOT RENAMED. This directory has held the real
# sessions, the 118 recordings, the settings and the evaluation history since
# the first session; it is the evidence base for E2 and every longitudinal
# claim in the handoff. Renaming it to match the product name would orphan all
# of it for a cosmetic gain. Packaging changes where the PROGRAM lives, never
# where the learner's data lives.
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", ROOT)) / "GermanPartner"
SESSIONS_DIR = DATA_DIR / "sessions"
AUDIO_DIR = DATA_DIR / "audio"
for d in (DATA_DIR, SESSIONS_DIR, AUDIO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- model status
MODEL_STATUS = "provisional"
MODEL_DECISION = {
    "llm": "gemma-4-E4B-it-qat-q4_0",
    "status": MODEL_STATUS,
    "confidence": "MEDIUM",
    "selected_from": "E0 + E1 Stage A (48 conversations, 576 turns)",
    "NOT_validated_by": ["E1 B1 blind transcript rating",
                         "E1 B2 blind live conversation",
                         "native-speaker German review"],
    "unresolved_risk": "German lexical/grammatical accuracy not fully validated",
    "runner_up": "qwen3-8b",
    "reopen_if": "systematic German errors observed during real use",
}
KNOWN_RISKS = [
    "WRONG-VOCABULARY RATE, MEASURED (app/_vocab_probe.py, 40 samples across 5 "
    "word-finding prompts at production sampling): correct 34/40 (85%), "
    "WRONG WORD 2/40 (5%), no word supplied 4/40 (10%). Both wrong instances "
    "were 'Drogerie' for 'Apotheke' -- a narrow, specific confusion rather than "
    "general vocabulary breakdown. Stochastic: the same prompt produced the "
    "correct word on other samples. Wrong vocabulary is the most damaging error "
    "class here because the learner cannot detect it.",
    "10% of word-finding attempts supplied NO German word at all -- the help "
    "silently fails. Arguably a bigger product problem than the 5% wrong rate, "
    "and it was not anticipated before measurement.",
    "Produced 'ein paar Zeug' (ungrammatical) in E1 Stage A S7.",
    "NOT re-opened on this evidence: 5% on one narrow confusion does not "
    "outweigh Gemma's structural advantages (0 artifacts vs 99 for Qwen, "
    "3038 MB vs 5258 MB VRAM, no prompt recitation, no template repetition). "
    "Real-usage flags are the instrument that should settle it.",
]

# ----------------------------------------------------------------------- LLM
LLM_PROVIDER = "llamacpp"
LLM_MODEL_PATH = LAB / "models" / "gemma-4-E4B_q4_0-it.gguf"
LLAMA_BUILD_DIR = LAB / "bin" / "llama124"        # CUDA 12.4 + PTX JIT -> sm_120
LLM_PORT = 8790
LLM_CTX = 4096
LLM_NGL = 999
LLM_SERVER_ARGS = ("--reasoning", "off", "--reasoning-budget", "0")
LLM_SAMPLING = dict(temperature=0.85, top_p=0.9, top_k=40, repeat_penalty=1.1)

# ------------------------------------------------------- runtime containment
# E0/E1 established a llama-server hang: /health keeps returning 200 while every
# completion times out, GPU idle, no recovery. Cause unknown; this is
# containment, not a fix. Stage A ran 576 turns with ZERO stalls under it.
REQ_TIMEOUT_S = 45
MAX_CONSEC_STALLS = 2
RESTART_EVERY_TURNS = 30        # conversations are longer here than Stage A's 12 turns
MAX_TURN_RETRIES = 2

# ----------------------------------------------------------------------- STT
STT_MODEL = "large-v3-turbo"
STT_DEVICE = "cuda"
STT_COMPUTE = "int8_float16"    # E0: 800 MB vs 1670 MB fp16, and marginally faster
STT_LANGUAGE = "de"
# A long interview vocabulary list caused severe repetitions on the learner's
# paused speech. A measured two-term hint fixed "Gonda/Konda" -> "Kunde" on the
# disputed recording without injecting or repeating words across the adjacent
# six real recordings. Keep this intentionally tiny.
STT_HOTWORDS = {
    "de": "Kunde Kundenservice",
    "en": "customer customer service",
}
# ⚑ 5, not 1. See providers.STT.transcribe for the measured justification:
# better readings on 8/17 real learner turns, fully deterministic output, +28 ms.
STT_BEAM_SIZE = 5

# ----------------------------------------------------------------------- TTS
# E3 tested six local German Piper voices on full sentences and the production
# chunked path. de_DE-thorsten-medium won on intelligibility (0.86% real WER)
# and latency. The later MLS speaker 14 substitution regressed real dialogue to
# 36% WER in the same evidence set and produced unintelligible short turns, so
# it must not be used for live interviews.
TTS_VOICE = LAB / "models" / "piper" / "de_DE-thorsten-medium.onnx"
TTS_VOICES = {
    "de": TTS_VOICE,
    "en": LAB / "models" / "piper" / "en_US-lessac-medium.onnx",
}
TTS_PROFILES = {
    "en_US-lessac-medium.onnx": {
        "speaker_id": None, "noise_scale": 0.45,
        "noise_w_scale": 0.55, "volume": 0.92,
    },
}
TTS_SAMPLE_RATE = 22050

# --------------------------------------------------------------------- level
# Level controls only high-value behaviour. No automatic CEFR assessment.
#
# ⚑ `length_scale` is Piper's DURATION multiplier, not a rate: HIGHER = SLOWER.
# Measured on de_DE-thorsten-high with stochastic duration disabled, one fixed
# sentence: 0.80 -> 6.34 s, 0.92 -> 6.78 s, 1.00 -> 6.90 s, 1.25 -> 7.52 s.
#
# This field was previously named `speech_rate` and held 0.85 / 0.92 / 1.00 for
# A2 / B1 / B2, which was passed straight into length_scale. That INVERTED the
# intent: A2 beginners got the fastest speech (15% faster than natural) and B2
# the slowest. Corrected below so A2 is slowest. Renamed to length_scale so the
# Piper semantics are visible at the call site and the inversion cannot recur.
LEVELS = {
    "A2": {"length_scale": 1.12, "target_words": "15 bis 25", "max_words": 40,
           "complexity": "Sehr einfache Sätze. Höchstens ein Nebensatz. "
                         "Erklär schwierige Wörter beiläufig im Satz.",
           "help_when_stuck": "Wenn er nicht weiterkommt, hilf schnell und direkt."},
    "B1": {"length_scale": 1.05, "target_words": "20 bis 35", "max_words": 55,
           "complexity": "Normale Sätze, auch mal ein Nebensatz. "
                         "Ungewöhnliche Wörter erklärst du kurz im Satz.",
           "help_when_stuck": "Wenn er nicht weiterkommt, hilf kurz und mach weiter."},
    "B2": {"length_scale": 1.0, "target_words": "25 bis 45", "max_words": 70,
           "complexity": "Ganz normales gesprochenes Deutsch, auch idiomatisch.",
           "help_when_stuck": "Wenn er nicht weiterkommt, gib nur einen kleinen Hinweis."},
}
DEFAULT_LEVEL = "B1"

# ------------------------------------------------------------------- audio io
AUDIO_INPUT_DEVICE = None        # None = system default; set from the UI
RECORD_SR = 16000
MAX_UTTERANCE_S = 120

# ------------------------------------------------------------- speech gate
# ⚑ Whisper HALLUCINATES ON SILENCE. Measured on a real 2.18 s capture of a
# quiet room (rms 0.00086 = -61 dBFS, peak 0.0091, zero frames above 0.01):
# large-v3-turbo returned "Vielen Dank." with avg_logprob -0.901 -- confident
# enough to pass the no_speech_prob guard, so Lena answered a sentence the user
# never said. In a real conversation that is worse than doing nothing.
#
# So: gate on ENERGY before spending ~460 ms on transcription. A recording must
# contain at least SPEECH_MIN_VOICED_S of audio above SPEECH_RMS_FLOOR to count
# as an utterance. The floor sits ~21 dB above the measured room noise, well
# below any speech that actually reaches a microphone.
# CALIBRATED, not guessed. Measured on this machine's real microphone:
#   silent room  -> max frame RMS 0.00293, mean 0.00046
#   floor 0.004  -> silence scores 0.00 s voiced, and real German speech still
#                   passes down to a peak of 0.015 (-36.5 dBFS), i.e. a very
#                   quiet talker on a low-gain mic.
# Raising this to 0.01 would reject anyone speaking below -26 dBFS, which is a
# realistic quiet voice. Rejecting real speech is far worse than letting a
# little noise through -- the confidence gate below is the second net.
SPEECH_RMS_FLOOR = 0.004         # frame RMS
# Kept deliberately short. Lena's whole register is built on one-word reactions
# ("Ja.", "Echt?", "Ach so.") and a gate that eats them would break exactly the
# behaviour the persona depends on. Measured: a QUIETLY spoken "Echt?" yields
# 0.10 s voiced, so 0.15 rejected it. The real silent room yields 0.00 s, so
# lowering this costs no noise robustness at all.
SPEECH_MIN_VOICED_S = 0.09
# Second net, after transcription.
#
# ⚑ RE-TUNED FROM REAL LEARNER SPEECH (session s20260808-142438 + -144143).
# The first value, -1.15, was a guess made before any human had spoken to the
# app. It then REJECTED FOUR REAL UTTERANCES the user actually said, including
# "Was bedeutet denn gar nicht 14 Uhr?" (-1.272) and one the user had to repeat
# (-1.160). Measured distribution:
#     18 accepted turns : -0.967 .. -0.264  (median -0.550)
#      4 rejected turns : -1.404 .. -1.160  -- all of them real speech
# Learner German scores systematically lower than native speech, so a threshold
# calibrated on intuition silently punishes exactly the user this app is for.
#
# Rejecting real speech is worse than passing imperfect text through: Gemma
# handles garbled input gracefully (it correctly recovered "Fingerspitzengefühl"
# from "Fingals betsell gifur"), whereas a rejection forces the learner to
# repeat with no explanation. The energy gate above already stops the silence
# hallucinations this was originally added for.
STT_MIN_AVG_LOGPROB = -1.60
# Turns below this are still REAL and still spoken, but flagged in telemetry so
# the misheard-diagnostic and E2 can find them later.
STT_LOW_CONFIDENCE_LOGPROB = -0.95

# ------------------------------------------------------- hands-free (E5)
# Half-duplex ONLY. The microphone stays CLOSED while Lena speaks.
# Measured justification (lab/results/e05_handsfree.jsonl, echo_probe): with the
# mic open during playback the recording sits +23.6 dB above silence and scores
# 0.08 s voiced against a 0.09 s gate -- a ONE FRAME margin, and that margin is a
# function of speaker volume. Whisper also transcribed the echo as "Vielen Dank."
# with avg_logprob -0.238, i.e. confidently wrong. So echo can never be filtered
# downstream; the only safe answer is to not record it.
# ⚑ DEFAULT ON. It was False, and that is exactly why the user's real 17-turn
# session logged handsfree=False and endpoint_reason="manual" on every single
# turn: the normal launcher started in push-to-talk and hands-free had to be
# opted into. The isolated test that "passed" had been launched with
# --handsfree, so it proved the machinery while missing the actual UX failure.
# Hands-free IS the intended normal experience; PTT is the fallback.
HANDSFREE_ENABLED_DEFAULT = True
# Silence that ends the learner's turn.
# ⚑ CALIBRATED ON THE USER'S OWN SPEECH (36 real recordings, 482 internal
# pauses): p50 0.10s · p90 0.64s · p95 0.78s · p98 1.07s · p99 1.44s · max 2.16s.
# 11 of 36 real turns contain a pause >= 1.0 s. A 1.0-1.5 s threshold -- the
# common default in voice assistants -- would cut this learner off mid-sentence
# on roughly a third of turns while they hunt for a German word. That is the
# failure the user explicitly said must not happen, so the threshold sits above
# the LONGEST pause ever observed, not above the median.
ENDPOINT_SILENCE_S = 2.5
# A cough, a click or a single loud breath must not end a turn.
ENDPOINT_MIN_SPEECH_S = 0.30
# Safety cap. Longest real turn so far was 25.5 s.
ENDPOINT_MAX_UTTERANCE_S = 60.0
# If nothing is said at all, stop listening rather than recording the room
# forever. The mic reopens on the next manual action.
ENDPOINT_NO_SPEECH_TIMEOUT_S = 45.0
# ⚑ POST-ROLL. The speech detector decides WHETHER there is speech; it must not
# decide the last sample Whisper is allowed to hear. Measured on the user's real
# recordings, quiet content beyond the gate-detected end is median 0.04 s, and
# German word endings routinely fall below the energy floor. 1.2 s sits far
# above the realistic worst case, so trimming can never eat a final -n or -en.
SPEECH_POSTROLL_S = 1.2
SPEECH_TRIM_MIN_SAVING_S = 0.8
# ⚑ TRIMMING IS OFF. Even the safe post-roll version saved a mean of only
# 0.10 s of silence while still perturbing 2/17 transcripts (against a 1/17
# baseline from Whisper's own run-to-run variance at greedy decoding). A
# tenth of a second of dead air is worth nothing; a clipped final -n corrupts
# the Mistakes Panel. So the recording goes to Whisper exactly as captured.
# The energy gate still decides WHETHER there is speech -- it just no longer
# decides which samples Whisper may hear.
SPEECH_TRIM_ENABLED = False

# Guard after Lena's audio ends before the mic opens, so the tail of her speech
# and any room reverberation cannot be captured as the learner's turn.
LISTEN_START_GUARD_S = 0.45

# ------------------------------------------------- mistakes panel (PROVISIONAL)
# ⚑ PROVISIONAL CORRECTION ASSISTANCE, pending E2. E2 (learner-error
# preservation through STT) has never been run, so we do not yet know whether
# Whisper preserves learner errors or quietly repairs them. Everything about
# this feature is built to be narrowed or disabled on E2's result, and nothing
# else in the product depends on it.
MISTAKES_ENABLED = True
# Precision over recall: a false correction teaches something wrong, which is
# worse than saying nothing. Deliberately high.
MISTAKES_MIN_CONFIDENCE = 0.75
MISTAKES_MIN_WORDS = 4           # one-word turns carry no analysable grammar
MISTAKES_MAX_PER_TURN = 2        # a wall of corrections is a failed panel
MISTAKES_MAX_TOKENS = 220        # keep the analysis request short: -np 1

# ------------------------------------------------- post-session evaluation
# Runs ONCE, after the conversation ends, on the already-loaded Gemma. Never
# during the conversation, and never a second GPU model.
EVALUATION_ENABLED = True
# Below this there is nothing meaningful to evaluate, and a thin evaluation
# invites over-reading. Real sessions run 11-17 turns.
EVALUATION_MIN_TURNS = 5
# Let in-flight mistake analysis finish before the evaluation asks for the
# single llama-server slot.
EVALUATION_DRAIN_S = 20.0
# Bounded, never infinite. One retry only, and only for failures that a fresh
# provider could plausibly fix.
EVALUATION_MAX_ATTEMPTS = 2
# Judgment needs turns Whisper was reasonably sure about. Below this we say so
# rather than guessing from unreliable transcripts.
EVALUATION_MIN_USABLE_TURNS = 3
# ⚑ A correction whose rewritten span contains a word Whisper scored below this
# is not used as EVIDENCE in the evaluation. Audited on s20260810-165738:
# "ich bin der Mode" rested on der=0.275 / Mode=0.178 while the turn as a whole
# scored a confident -0.407. The Mistakes Panel still shows such cards -- this
# only stops them supporting a claim about the learner's grammar.
EVALUATION_MIN_SPAN_WORD_P = 0.35

# ------------------------------------------------------------------ feedback
# Single-keypress flags. Deliberately not a questionnaire.
FEEDBACK_FLAGS = {
    "1": ("wrong_word", "falsches Wort"),
    "2": ("grammar_error", "Grammatikfehler"),
    "3": ("unnatural", "unnatürlich"),
    "4": ("strange", "seltsame Antwort"),
    "5": ("repetitive", "wiederholt sich"),
    "6": ("too_many_questions", "zu viele Fragen"),
    "7": ("too_long", "zu lang"),
    "8": ("too_short", "zu kurz"),
    "9": ("good", "gute Antwort"),
}

# ------------------------------------------------------- coach (V2)
# ⚑ PRODUCT DIRECTION CHANGE, on explicit user decision (2026-09-03): the app is
# no longer a conversation partner who never corrects. Every turn is meant to
# produce more learner speech, and mistakes are worked on.
#
# ⚑ SECOND DECISION, SAME DAY: the coaching must not SHOW. The character is
# Jonas, 29, from Leipzig -- someone you talk to, not a teacher standing in
# front of you. The intelligence is unchanged; the surface is a conversation.
# The prompt therefore forbids level names, lesson announcements and any
# commentary on the learner's German (app/coach.json voice_notes.avoid).
#
# What did NOT change, and must not: the model, the sampling, the containment,
# half-duplex hands-free, the endpointing calibration, the response-length
# table, the streaming pipeline, and the rule that a correction the learner
# disputes is not a learner mistake. The teacher is a PROMPT + POLICY layer over
# the same engine (app/teaching.py, app/learner.py), not a new pipeline.
#
# Two earlier personas are kept unchanged on disk as the record of what was
# frozen before each decision, and neither is loaded any more:
#   app/persona.json  Lena, the conversation partner who never corrected
#   app/teacher.json  Frau Berger, the explicit teacher
PERSONA_FILE = "coach.json"
TEACHER_PERSONA_FILE = PERSONA_FILE          # kept: older references still read it

# The per-turn Conversation Director. Costs NO extra LLM call: every decision is
# made from counted session state. Turn it off and the teacher still teaches --
# she just stops being steered turn by turn.
TEACHER_DIRECTOR_ENABLED = True
# A directive is inserted as a system message immediately BEFORE the learner's
# last message, never appended to the stored history. See app/teaching.py.
TEACHER_DEEPEN_CHANCE = 0.75      # how often a plain turn still gets steered
TEACHER_SOCRATIC_ENABLED = True
# Spoken review of collected corrections. The user's rule: 2-4, never 15.
TEACHER_REVIEW_EVERY_TURNS = 8
TEACHER_REVIEW_MAX_ITEMS = 2
# A topic that has run this long is exhausted; the director bridges to a new one.
TEACHER_TOPIC_MAX_TURNS = 6
TEACHER_CHALLENGE_EVERY_TURNS = 5
TEACHER_ROLEPLAY_CHANCE = 0.5
# ⚑ The system prompt is the cached prompt PREFIX. Rebuilding it every turn
# throws away llama-server's prefix cache and re-processes the whole
# conversation, which is a first-token-latency regression. It is rebuilt on a
# level change and at most every N turns.
TEACHER_PROMPT_REFRESH_TURNS = 6
# Within-level speaking-rate adaptation. Bounded to +-6% around the frozen
# per-level length_scale (see LEVELS above and learner.speech_scale_factor).
TEACHER_ADAPT_SPEECH_RATE = True

# ------------------------------------------------------------- vocabulary
# Cards run on the SAME background worker as the mistake analysis, because
# llama-server has one slot. Never on the conversation path.
VOCAB_ENABLED = True
VOCAB_MAX_TOKENS = 160
# The director nudges the teacher to offer a better word at most this often.
VOCAB_SUGGEST_EVERY_TURNS = 4
# If she offered nothing, probe anyway this often so the panel is not empty.
VOCAB_PROBE_EVERY_TURNS = 5
VOCAB_MAX_PER_SESSION = 12

# --------------------------------------------------------------- visemes
# ⚑ TEXT-DERIVED LIP SYNC, NOT AUDIO ANALYSIS. The browser never sees the audio
# -- all playback is in Python (that is what lets the desktop shell run without
# media permissions), so there is nothing for the UI to analyse. Instead the
# engine derives a coarse mouth-shape timeline from the chunk TEXT and the EXACT
# PCM duration it is about to play, and streams it with the chunk. Cost is a few
# microseconds of string work per chunk and no GPU at all.
#
# It is an approximation and is meant to be: the goal is "this person is
# talking", not phoneme-accurate animation. If it is ever wrong it is wrong by a
# few tens of milliseconds, which reads as natural speech anyway.
VISEMES_ENABLED = True
VISEME_MIN_MS = 55               # no shape shorter than this; faster reads as jitter

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8791

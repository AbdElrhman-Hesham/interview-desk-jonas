"""Frozen invariants — the cheap check that runs first and needs no hardware.

Nothing here is a new experiment. Every assertion below is a decision that was
already MEASURED and then written down, plus the hand-checklist the project
handoff tells the next person to re-verify by hand before touching anything.
This file exists so nobody has to.

Each entry names the evidence that fixed the value. If one of these fails, the
correct response is NOT to update the number here — it is to find out who
changed a frozen behaviour and why.

⚑ Two of these are checked on the BUILT artefact, not on the source text:

  * the conversation prompt is checked on `Engine._system_prompt()`'s output,
    because the forbidden word still appears in a source comment documenting
    its removal — grep alone gives a false alarm (and a false pass, if someone
    only edits the comment);
  * the shutdown ordering is checked by reading which statement sets
    `_shutdown_done` relative to the evaluation call, because the bug it guards
    against was an ordering bug that every unit-level test missed.

Runs in ~1 s. No GPU, no model, no audio device.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.engine import Engine  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((bool(ok), name, detail))


# ─────────────────────────────────────────────────── runtime / model identity
def check_models():
    check("llm is gemma-4-E4B_q4_0-it.gguf",
          config.LLM_MODEL_PATH.name == "gemma-4-E4B_q4_0-it.gguf",
          config.LLM_MODEL_PATH.name)
    check("llm sampling frozen (0.85/0.9/40/1.1)",
          config.LLM_SAMPLING == dict(temperature=0.85, top_p=0.9, top_k=40,
                                      repeat_penalty=1.1),
          str(config.LLM_SAMPLING))
    check("llama.cpp runs with -np 1 reasoning off",
          config.LLM_SERVER_ARGS == ("--reasoning", "off", "--reasoning-budget", "0"),
          str(config.LLM_SERVER_ARGS))
    provider_src = (ROOT / "app" / "providers.py").read_text(encoding="utf-8")
    check("llama.cpp stays headless on Windows",
          "subprocess.CREATE_NO_WINDOW" in provider_src
          and "creationflags=creationflags" in provider_src,
          "Windows console suppression missing")
    # E8: beam 5 gave the better reading on 8/17 real learner turns and is
    # deterministic (0/17 run-to-run variance vs 1/17 greedy).
    check("STT beam_size == 5 (E8)", config.STT_BEAM_SIZE == 5,
          str(config.STT_BEAM_SIZE))
    check("STT model large-v3-turbo / int8_float16 / de",
          (config.STT_MODEL, config.STT_COMPUTE, config.STT_LANGUAGE)
          == ("large-v3-turbo", "int8_float16", "de"),
          f"{config.STT_MODEL} {config.STT_COMPUTE} {config.STT_LANGUAGE}")
    check("TTS voice is the measured E3 intelligibility winner",
          Path(config.TTS_VOICE).name == "de_DE-thorsten-medium.onnx",
          Path(config.TTS_VOICE).name)


# ────────────────────────────────────────────── learner-speech safety (E5/E6/E8)
def check_learner_speech_safety():
    # E5: calibrated on 482 internal pauses in the user's own 36 recordings;
    # max pause observed 2.16 s. Tightening this cuts the learner off.
    check("endpoint silence >= 2.5 s (E5 calibration)",
          config.ENDPOINT_SILENCE_S >= 2.5, f"{config.ENDPOINT_SILENCE_S}s")
    # E8b: a flat trim changed 14/17 real transcripts; even the safe post-roll
    # version perturbed 1-2/17 to save a mean of 0.10 s.
    check("audio trimming DISABLED (E8b)",
          config.SPEECH_TRIM_ENABLED is False, str(config.SPEECH_TRIM_ENABLED))
    # Four real utterances were silently rejected by a guessed -1.15 threshold.
    check("STT reject threshold <= -1.60 (re-tuned on real speech)",
          config.STT_MIN_AVG_LOGPROB <= -1.60, str(config.STT_MIN_AVG_LOGPROB))
    # Measured: a quietly spoken "Echt?" yields 0.10 s voiced.
    check("speech gate floor 0.004 / min voiced <= 0.09 s",
          config.SPEECH_RMS_FLOOR == 0.004 and config.SPEECH_MIN_VOICED_S <= 0.09,
          f"{config.SPEECH_RMS_FLOOR} / {config.SPEECH_MIN_VOICED_S}")
    # The "hands-free is not hands-free" report: shipped but defaulted off.
    check("hands-free ON by default", config.HANDSFREE_ENABLED_DEFAULT is True,
          str(config.HANDSFREE_ENABLED_DEFAULT))
    # E5 echo probe: mic open during playback sits +23.6 dB over silence and
    # Whisper reads it as confident German. Half-duplex needs this guard.
    check("listen-start guard >= 0.45 s (half-duplex)",
          config.LISTEN_START_GUARD_S >= 0.45, f"{config.LISTEN_START_GUARD_S}s")


# ──────────────────────────────────────────────────── conversation behaviour
def check_conversation_prompt():
    """Checked on the BUILT prompt, not the file. See module docstring.

    ⚑ V2. The product became a coach on explicit user decision (2026-09-03) --
    first an explicit teacher, then, the same day, a coach whose teaching must
    NOT show. The persona changed twice; the two findings below did NOT, because
    neither of them was about who is talking:
      * "ungefragt" licensed non-sequiturs in 55 of 92 real turns;
      * "anknüpfend" is what fixed hook pickup (0.31 -> 0.56).
    Someone who ignores what the learner just said is a bad conversation partner
    whatever you call him. Both assertions therefore stay, and are now checked
    against the prompt built by app/teaching.py.
    """
    e = Engine.__new__(Engine)
    e.level = "B1"
    e.learner = None
    e.turn = 0
    e._prompt_built_at_turn = 0
    p = Engine._system_prompt(e)
    # The literal instruction "Du erzählst von dir selbst, ungefragt." is what
    # licensed the non-sequiturs (Lena's sister in Hamburg, her fourth coffee)
    # in 55 of 92 real turns. Removing it moved hook pickup 0.31 -> 0.56.
    check("prompt no longer licenses unprompted self-disclosure",
          "ungefragt" not in p, "found 'ungefragt' in the BUILT prompt")
    check("prompt requires connecting to what the learner said",
          "anknüpfend" in p, "'anknüpfend' missing from the BUILT prompt")
    # Response length is user-approved. The instruction that carries it must
    # still be there, and the level table must still hold the approved numbers.
    check("B1 target length unchanged (20 bis 35 / max 55)",
          config.LEVELS["B1"]["target_words"] == "20 bis 35"
          and config.LEVELS["B1"]["max_words"] == 55,
          str(config.LEVELS["B1"]))
    check("A2 speaks slowest, B2 fastest (length_scale not inverted)",
          config.LEVELS["A2"]["length_scale"] > config.LEVELS["B1"]["length_scale"]
          > config.LEVELS["B2"]["length_scale"],
          str({k: v["length_scale"] for k, v in config.LEVELS.items()}))
    check("default level B1", config.DEFAULT_LEVEL == "B1", config.DEFAULT_LEVEL)


# ───────────────────────────────────────────────────────── coach policy (V2)
def check_teacher():
    """The coaching layer's own load-bearing decisions.

    Each of these is a rule the product would silently lose without noticing:
    a prompt rebuilt every turn costs first-token latency; a directive appended
    to the history repeats itself forever; a correction review that grows past
    a handful of items becomes the wall of corrections the user explicitly
    rejected; an automatic CEFR jump is exactly what he asked not to happen;
    and a prompt that names the level is how a model ends up saying "gut für B1"
    out loud, which is the one thing the coach must never do.
    """
    import json as _json
    import inspect as _inspect
    from app import teaching

    t = _json.loads((ROOT / "app" / config.PERSONA_FILE).read_text(encoding="utf-8"))
    check("persona file is the coach, not an earlier persona",
          config.PERSONA_FILE == "coach.json" and t.get("id") == "jonas",
          f"{config.PERSONA_FILE} / {t.get('id')}")
    check("coach persona has identity, principles and a correction policy",
          bool(t.get("identity") and t.get("principles") and t.get("correction_policy")),
          "the persona file is missing a required section")
    check("topic catalogue is large enough to avoid repetition",
          len(t.get("topics") or []) >= 24, f"{len(t.get('topics') or [])} topics")
    kinds = {x.get("kind") for x in t.get("topics") or []}
    check("topic catalogue covers role-play, opinion and comparison",
          {"rollenspiel", "meinung", "vergleich"} <= kinds, str(sorted(kinds)))

    check("topic chains exist, so transitions can be associative",
          len(t.get("chains") or []) >= 3
          and all(len(c) >= 3 for c in (t.get("chains") or [])),
          "chains missing -- topic changes fall back to random jumps")
    chain_ids = {x for c in (t.get("chains") or []) for x in c}
    topic_ids = {x["id"] for x in t.get("topics") or []}
    check("every chain station is a real topic", chain_ids <= topic_ids,
          str(sorted(chain_ids - topic_ids)))

    p = teaching.build_system_prompt("B1", {"level": "B1"})
    check("built prompt states the speaking purpose",
          "besser sprechen" in p and "mehr" in p,
          "the coaching purpose disappeared from the built prompt")
    # ⚑ THE COACHING MUST NOT SHOW. These four are the whole second decision.
    # ⚑ The level names DO appear once, in the list of things he must never say
    # ("Niveau-Angaben wie A2, B1, B2"). That line is the rule, so it is removed
    # before scanning -- otherwise the check fails on its own enforcement.
    p_body = "\n".join(ln for ln in p.splitlines() if not ln.startswith("- Vermeide:"))
    check("built prompt never names a CEFR level outside the ban list",
          not any(x in p_body for x in ("A2", "B1", "B2", "Niveau")),
          "a level name reached the prompt -- it will reach the learner")
    check("built prompt forbids teacher language",
          "Lehrersprache" in p and "Metakommentare" in p,
          "the anti-lesson-language rule is gone")
    check("built prompt forbids announcing a topic change",
          "kündigst einen Themenwechsel nie an" in p,
          "topic changes may now be announced")
    check("every director instruction carries the keep-quiet clause",
          teaching.QUIET.strip().startswith("Sag nicht")
          and "korrigierst" in teaching.QUIET,
          "the directive suffix that hides the coaching changed")
    check("built prompt carries the recast-first correction policy",
          "beiläufig" in p, "recast policy missing from the built prompt")
    check("built prompt still forbids markdown/emoji/stage directions",
          "Keine Emojis" in p and "Regieanweisungen" in p, "output hygiene lost")

    # ⚑ The response-length table is user-approved and is NOT a teaching dial.
    check("prompt still carries the approved B1 length",
          "20 bis 35" in p and "55" in p, "approved response length lost")

    # A wall of corrections is a failed review. The user's number was 2-4.
    check("spoken review stays small (<= 4 items)",
          1 <= config.TEACHER_REVIEW_MAX_ITEMS <= 4,
          str(config.TEACHER_REVIEW_MAX_ITEMS))
    # Rebuilding messages[0] every turn invalidates llama-server's prefix cache.
    check("system prompt is NOT rebuilt every turn",
          config.TEACHER_PROMPT_REFRESH_TURNS >= 3,
          str(config.TEACHER_PROMPT_REFRESH_TURNS))

    src = _inspect.getsource(Engine._request_messages)
    check("directive is placed BEFORE the last user message",
          "head + [{" in src and "self.messages[:-1]" in src,
          "directive placement changed -- a trailing system message is dropped "
          "by the Gemma chat template")
    check("directive is never stored in the conversation history",
          "self.messages.append" not in src, "the directive leaks into history")

    # Difficulty may crawl, never jump, and never change CEFR by itself.
    from app import learner as learner_mod
    check("difficulty moves by <= 0.05 per turn",
          learner_mod.DIFFICULTY_STEP <= 0.05, str(learner_mod.DIFFICULTY_STEP))
    lm = learner_mod.LearnerModel.__new__(learner_mod.LearnerModel)
    lm.level, lm.difficulty = "B1", 1.0
    lm.high_streak = learner_mod.LEVEL_SUGGEST_STREAK
    lm.turns, lm.level_suggested = 50, False
    check("a CEFR change is only ever SUGGESTED, never applied automatically",
          lm.wants_level_up() is True and lm.level == "B1",
          "wants_level_up changed the level itself")

    # The vocabulary probe must stay on the one background worker.
    # Lip sync must stay text-derived: the browser has no audio to analyse.
    vsrc = (ROOT / "app" / "visemes.py").read_text(encoding="utf-8")
    check("viseme timeline is derived from text + real PCM duration",
          "def timeline" in vsrc and "duration_ms" in vsrc,
          "app/visemes.py changed shape")
    from app import visemes as _vz
    _tl = _vz.timeline("Ach so, und warum?", 1200)
    check("viseme timeline covers the chunk and ends closed",
          bool(_tl) and _tl[-1][1] == "X" and _tl[-1][0] <= 1200,
          str(_tl[-3:]))
    ui_raw = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    # ⚑ Strip HTML comments first. The page's own header comment DOCUMENTS this
    # rule by naming the forbidden APIs, and a naive grep fails on the
    # documentation -- the same false-alarm class this file's docstring warns
    # about for the conversation prompt.
    import re as _re
    ui = _re.sub(r"<!--.*?-->", "", ui_raw, flags=_re.S)
    check("the UI still performs no audio",
          not any(x in ui for x in ("getUserMedia", "AudioContext",
                                    "MediaRecorder", "new Audio(")),
          "the page started touching audio -- the desktop shell depends on it "
          "not doing that")
    check("the UI carries no WebGL/3D engine",
          not any(x in ui for x in ("three.min", "THREE.", "getContext('webgl'",
                                    'getContext("webgl"')),
          "a 3D engine appeared in the UI -- it competes with the models for GPU")

    msrc = (ROOT / "app" / "mistakes.py").read_text(encoding="utf-8")
    check("vocabulary probes share the mistake worker (-np 1 discipline)",
          "def submit_vocab" in msrc and "threading.Thread" not in
          msrc.split("def submit_vocab")[1].split("def ")[0],
          "the vocabulary coach started its own thread")


# ───────────────────────────────────────────── interview product (V3)
def check_interview():
    """The bilingual interview flow must stay deterministic and reportable."""
    from app import interview

    de = interview.InterviewController("de")
    en = interview.InterviewController("en")
    check("interview has 12 German and 12 English questions",
          de.total == 12 and en.total == 12, f"{de.total} / {en.total}")
    ids = [q.id for q in de.questions + en.questions]
    check("interview question ids are unique", len(ids) == len(set(ids)), str(ids))
    check("both interview banks include a customer role-play",
          any(q.competency == "roleplay" for q in de.questions)
          and any(q.competency == "roleplay" for q in en.questions),
          "role-play missing from one language")

    q = de.current_question
    saved = de.record_answer("Eine konkrete Antwort.")
    check("an answer is saved with its exact question",
          saved["id"] == q.id and saved["text"] == q.text
          and saved["answer"] == "Eine konkrete Antwort.", str(saved))
    response, meta = de.response_after_answer()
    check("the next deterministic question is spoken exactly",
          response == de.current_question.text
          and meta.get("next_question_id") == de.current_question.id,
          response)
    engine_src = (ROOT / "app" / "engine.py").read_text(encoding="utf-8")
    check("live interview questions bypass language-model generation",
          "self.interview.response_after_answer()" in engine_src
          and '"source": "deterministic_interview"' in engine_src,
          "the live question path can drift or stall on model generation")

    de_prompt, en_prompt = interview.system_prompt("de"), interview.system_prompt("en")
    check("interviewer does not score or coach during the interview",
          "bewertest du nicht" in de_prompt and "Do not coach" in en_prompt,
          "live-assessment boundary missing")
    check("interview prompts enforce the selected language",
          "ausschließlich Deutsch" in de_prompt and "only English" in en_prompt,
          "language containment missing")

    check("native German and English voices are configured",
          set(config.TTS_VOICES) == {"de", "en"}
          and config.TTS_VOICES["de"].name.startswith("de_DE-")
          and config.TTS_VOICES["en"].name.startswith("en_US-"),
          str(config.TTS_VOICES))
    check("German interview voice is the measured intelligibility winner",
          config.TTS_VOICES["de"].name == "de_DE-thorsten-medium.onnx",
          config.TTS_VOICES["de"].name)
    from app.providers import STT
    stt_src = inspect.getsource(STT.transcribe)
    check("Whisper reads the selected interview language",
          "language=self.language" in stt_src, "STT language is still fixed")
    check("Whisper uses neural VAD and only the measured minimal hotwords",
          config.STT_HOTWORDS == {"de": "Kunde Kundenservice",
                                 "en": "customer customer service"}
          and "vad_filter=True" in stt_src
          and "hotwords=config.STT_HOTWORDS[self.language]" in stt_src,
          "safe interview recognition settings missing")
    prepare_src = inspect.getsource(STT._prepare_audio)
    check("quiet microphone enhancement is bounded and clipping-safe",
          "level >= 0.02" in prepare_src and "min(1.5" in prepare_src
          and "0.97 / peak" in prepare_src,
          "bounded input conditioning missing")

    server_src = (ROOT / "app" / "server.py").read_text(encoding="utf-8")
    report_src = (ROOT / "app" / "interview_report.py").read_text(encoding="utf-8")
    ui = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    check("language choice is an explicit start-screen action",
          'data-lang="de"' in ui and 'data-lang="en"' in ui
          and "/api/interview/start" in ui, "bilingual start screen missing")
    check("ordered shutdown creates the interview PDF before teardown",
          "interview_report.finalize" in server_src
          and "create_pdf" in report_src and "report_path" in report_src,
          "PDF finalization is not on the real end path")
    check("PDF report includes every saved question and answer",
          "controller.answers" in report_src and "answer_feedback" in report_src,
          "report lost the auditable answer mapping")


# ──────────────────────────────────────────────── mistakes panel is provisional
def check_mistakes_provisional():
    check("mistakes precision-first (conf >= 0.75, <= 2/turn)",
          config.MISTAKES_MIN_CONFIDENCE >= 0.75
          and config.MISTAKES_MAX_PER_TURN <= 2,
          f"{config.MISTAKES_MIN_CONFIDENCE} / {config.MISTAKES_MAX_PER_TURN}")
    src = (ROOT / "app" / "evaluate.py").read_text(encoding="utf-8")
    check("evaluation still labels itself PROVISIONAL_pending_E2",
          "PROVISIONAL_pending_E2" in src,
          "the E2 provisionality label disappeared from app/evaluate.py")


# ────────────────────────────────────────────────── shutdown ordering (E10)
def check_shutdown_ordering():
    """The StallError was an ORDERING bug, so assert the order structurally.

    Parse app/server.py and require, inside _shutdown(): the evaluator call
    appears BEFORE the statement that sets _shutdown_done. main() waits on
    _shutdown_done and then kills llama-server in its finally block, so setting
    it first is what killed the evaluation mid-request (eval_wall_ms = 44.9).
    """
    src = (ROOT / "app" / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_shutdown"), None)
    if fn is None:
        check("server._shutdown exists", False, "function not found")
        return

    eval_lines, done_lines, started_lines = [], [], []
    for n in ast.walk(fn):
        # Cover both paths: standard evaluation (evaluator.build) and
        # interview evaluation (interview_report.finalize).
        if isinstance(n, ast.Attribute) and n.attr in ("build", "finalize"):
            eval_lines.append(n.lineno)
        if isinstance(n, ast.Attribute) and n.attr == "set":
            tgt = getattr(n.value, "id", "")
            if tgt == "_shutdown_done":
                done_lines.append(n.lineno)
            elif tgt == "_shutdown_started":
                started_lines.append(n.lineno)

    check("_shutdown calls the evaluator", bool(eval_lines),
          "no evaluator.build() or interview_report.finalize()")
    check("_shutdown_started gives idempotency", bool(started_lines),
          "no _shutdown_started.set()")
    check("_shutdown_done set AFTER the evaluation (E10 race)",
          bool(eval_lines and done_lines and min(done_lines) > max(eval_lines)),
          f"evaluator.build() at line {eval_lines}, _shutdown_done.set() at {done_lines}")
    check("_shutdown_started set BEFORE the evaluation",
          bool(started_lines and eval_lines and min(started_lines) < min(eval_lines)),
          f"_shutdown_started at {started_lines}, evaluator at {eval_lines}")

    # main() must wait on _shutdown_done, or the ordering above buys nothing.
    mainfn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    waits = mainfn is not None and any(
        isinstance(n, ast.Attribute) and n.attr == "wait"
        and getattr(n.value, "id", "") == "_shutdown_done" for n in ast.walk(mainfn))
    check("main() waits on _shutdown_done before tearing down", waits,
          "main() does not wait on _shutdown_done")


# ─────────────────────────────────────── telemetry honesty + harness marking
def check_telemetry_honesty():
    from app.telemetry import SessionLog
    src = inspect.getsource(SessionLog.close)
    # An unmeasured stage is null, never 0. `median_stt_ms: 0` was read as a
    # real measurement for two days.
    check("every median carries its sample count",
          'metrics[f"n_{name}"]' in src or "n_{name}" in src,
          "sample counts dropped from the summary")
    check("harness runs are stamped and excluded from product metrics",
          "harness" in src and "self.harness is None" in src,
          "harness marking missing from SessionLog.close")
    # Guard the identification rule itself: sessions written under LENA_HARNESS
    # must be distinguishable from learner sessions.
    import os
    check("LENA_HARNESS is honoured by SessionLog",
          "LENA_HARNESS" in inspect.getsource(SessionLog.__init__),
          "SessionLog ignores LENA_HARNESS")
    del os


def main() -> int:
    print("=" * 78)
    print("  FROZEN INVARIANTS  —  config, built prompt, teaching policy,")
    print("                        shutdown ordering")
    print("  (no GPU, no model, no audio device)")
    print("=" * 78)
    for fn in (check_models, check_learner_speech_safety,
               check_conversation_prompt, check_teacher, check_interview,
               check_mistakes_provisional,
               check_shutdown_ordering, check_telemetry_honesty):
        try:
            fn()
        except Exception as ex:
            check(f"{fn.__name__} raised", False, f"{type(ex).__name__}: {ex}")

    print()
    for ok, name, detail in RESULTS:
        line = f"  {'PASS' if ok else 'FAIL'}  {name}"
        if not ok and detail:
            line += f"\n           -> {detail}"
        print(line)

    failed = [n for ok, n, _d in RESULTS if not ok]
    print(f"\n  {len(RESULTS) - len(failed)}/{len(RESULTS)} invariants hold")
    print(f"\n  INVARIANTS: {'PASS' if not failed else 'FAIL'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

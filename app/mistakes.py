"""Passive mistakes panel -- PROVISIONAL CORRECTION ASSISTANCE.

⚑ STATUS: PROVISIONAL, AND IT MUST STAY LABELLED THAT WAY UNTIL E2 PASSES.
E2 (learner-error preservation through STT) has never been run. Until it has, we
do NOT know whether Whisper preserves the errors a learner actually made or
quietly repairs them. Everything here is therefore built to be narrowed or
switched off, and nothing else in the product depends on it.

DESIGN CONSTRAINTS, all from the user:
  - Lena must NEVER correct out loud. The conversation stays a conversation;
    this panel is the only educational surface.
  - Analysis must NOT delay Lena. It is queued and runs AFTER her reply is
    already streaming, in the long idle window while the learner formulates
    their next turn (measured: real learner turns take 4-25 s).
  - No second GPU model. This reuses the ONE llama-server already loaded, with
    a different system prompt, so conversation and correction stay logically
    separate without costing VRAM or contending for the GPU.
  - PRECISION OVER RECALL. A false correction teaches the learner something
    wrong and is worse than a missed one.

⚑ V2: THIS WORKER NOW CARRIES TWO JOB KINDS, NOT ONE.
The teacher release added vocabulary cards. They deliberately do NOT get their
own thread: llama-server runs `-np 1`, so a second background analyser would sit
in front of the learner's next turn on the single slot. A vocabulary probe is
queued as `kind: "vocab"` on THIS queue, handled by THIS worker, behind the same
wait-for-idle gate. The grammar prompt, its thresholds and its accept filter are
untouched -- precision over recall still governs corrections.

THE ASR PROBLEM, HANDLED EXPLICITLY:
A Whisper transcript is NOT ground truth about what the learner said. If the
transcript is low-confidence, we cannot tell "the learner made a case error"
from "Whisper misheard the ending". Those turns are SUPPRESSED rather than
guessed at, and the suppression is recorded so E2 can quantify what we lost.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
import unicodedata

from . import config, vocab as vocab_mod

# Only these are worth a learner's attention. Anything outside this list is
# noise in a panel that must stay short.
USEFUL_TYPES = {
    "case", "gender", "article", "verb", "tense", "word_order",
    "preposition", "vocabulary", "expression", "auxiliary",
}

# ⚑ NOT every card is an error. Some are "a German would phrase it this way",
# which is useful but must NOT be presented as wrong -- the user asked for this
# distinction explicitly, and mislabelling acceptable German as a mistake is
# discouraging for exactly the learner this app exists for.
# This is a LABEL derived from the type we already have. Detection, density,
# confidence and the prompt are untouched.
STYLE_TYPES = {"expression"}


def family_of(mtype: str) -> str:
    return "style" if mtype in STYLE_TYPES else "grammar"

SYSTEM = (
    "Du bist ein präziser Analysator für Deutsch-Lernerfehler. "
    "Du antwortest AUSSCHLIESSLICH mit gültigem JSON, ohne Erklärungen davor "
    "oder danach, ohne Markdown, ohne Codeblöcke."
)

PROMPT = """Analysiere diesen Satz eines Deutschlerners (Niveau {level}).

SATZ: "{text}"

Finde NUR klare, lernwerte Fehler. Melde KEINE:
- Füllwörter, Zögern, Wiederholungen
- Zeichensetzung oder Groß-/Kleinschreibung
- umgangssprachliche, aber korrekte Formen
- stilistische Vorlieben
- Dinge, die nur wegen fehlender Satzzeichen falsch wirken

Melde NUR: falscher Kasus, falsches Genus, falscher Artikel, falsche
Verbform, falsche Zeitform, falsche Wortstellung, falsche Präposition,
falsches Hilfsverb, klar falsches Wort, oder ein Ausdruck, für den es eine
eindeutig natürlichere deutsche Form gibt.

Wenn der Satz in Ordnung ist, gib eine leere Liste zurück.
Wenn der Satz unverständlich oder abgeschnitten wirkt, gib eine leere Liste
zurück - dann ist wahrscheinlich die Spracherkennung schuld, nicht der Lerner.

Antworte exakt in diesem JSON-Format:
{{"mistakes":[{{"original":"<genau der falsche Teil>","better":"<korrigiert>",\
"type":"<case|gender|article|verb|tense|word_order|preposition|vocabulary|\
expression|auxiliary>","explanation":"<max 12 Wörter, auf Deutsch>",\
"confidence":<0.0-1.0>}}]}}"""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", (s or "").lower().strip())
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_json(raw: str) -> dict | None:
    """Gemma is a chat model, not a JSON API. Recover the object if it wraps it."""
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(raw[i:j + 1])
    except Exception:
        return None


class MistakeAnalyzer:
    """Queued, off-critical-path learner-error analysis."""

    def __init__(self, llm, engine, on_result, min_confidence: float = 0.75,
                 on_vocab=None):
        self.llm = llm
        self.engine = engine
        self.on_result = on_result
        # ⚑ V2: vocabulary cards ride this same worker. See module docstring.
        self.on_vocab = on_vocab or (lambda _rec: None)
        self.min_confidence = min_confidence
        self.q: queue.Queue = queue.Queue()
        self.items: list[dict] = []          # what the panel shows
        self.vocab: list[dict] = []          # V2: vocabulary cards
        self._vocab_keys: set[str] = set()
        self.n_vocab_probes = 0
        self._by_key: dict[str, dict] = {}   # dedup: recurring patterns
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.n_analyzed = 0
        self.n_suppressed_asr = 0
        self.n_skipped_short = 0
        self.analysis_ms: list[float] = []
        self._busy = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- intake
    def submit(self, turn: int, text: str, avg_logprob, low_conf: bool,
               audio_file: str | None, level: str):
        """Queue a learner turn. Never blocks the conversation."""
        if not config.MISTAKES_ENABLED:
            return
        words = len((text or "").split())
        if words < config.MISTAKES_MIN_WORDS:
            self.n_skipped_short += 1
            return
        # ⚑ ASR uncertainty gate. If we cannot trust the transcript we cannot
        # attribute the error to the learner, so we do not guess.
        if low_conf:
            self.n_suppressed_asr += 1
            self._log_event("mistake_suppressed_asr", turn=turn, text=text,
                            avg_logprob=avg_logprob)
            return
        self.q.put({"kind": "grammar", "turn": turn, "text": text,
                    "avg_logprob": avg_logprob,
                    "audio_file": audio_file, "level": level})

    def submit_vocab(self, turn: int, learner_text: str, teacher_text: str,
                     offered: list, level: str, known: set):
        """Queue a vocabulary probe. Same queue, same slot discipline.

        Dropped silently once the session has enough cards -- a panel that keeps
        growing all lesson is a vocabulary list, not a lesson.
        """
        if not config.VOCAB_ENABLED or self._stop.is_set():
            return
        if len(self.vocab) >= config.VOCAB_MAX_PER_SESSION:
            return
        if not (teacher_text or "").strip():
            return
        self.q.put({"kind": "vocab", "turn": turn, "text": learner_text,
                    "teacher_text": teacher_text, "offered": list(offered or []),
                    "level": level, "known": set(known or ())})

    def drain(self, timeout: float = 20.0) -> bool:
        """Wait for queued analyses to finish, then stop accepting new ones.

        ⚑ Called at session end BEFORE the evaluation runs. llama-server has one
        slot (-np 1), so an analysis still in flight would sit in front of the
        evaluation request. Returns True if the queue emptied in time.
        """
        import time as _t
        t0 = _t.perf_counter()
        while _t.perf_counter() - t0 < timeout:
            if self.q.empty() and not self._busy:
                break
            _t.sleep(0.1)
        drained = self.q.empty() and not self._busy
        self.stop()
        return drained

    def stop(self):
        self._stop.set()
        self.q.put(None)

    def _log_event(self, kind, **kw):
        try:
            if self.engine and self.engine.log:
                self.engine.log.event(kind, **kw)
        except Exception:
            pass

    # ------------------------------------------------------------- worker
    def _worker(self):
        while not self._stop.is_set():
            item = self.q.get()
            if item is None:
                return
            self._busy = True
            try:
                if item.get("kind") == "vocab":
                    self._analyze_vocab(item)
                else:
                    self._analyze(item)
            except Exception as e:
                self._log_event("mistake_analysis_error",
                                turn=item.get("turn"),
                                detail=f"{type(e).__name__}: {e}")
            finally:
                self._busy = False

    def _wait_for_idle(self, timeout=25.0):
        """Do not compete with Lena for the single llama-server slot.

        llama-server runs with -np 1, so an analysis request issued while a
        reply is generating would sit in front of the learner's next turn. The
        natural idle window is exactly while the learner is formulating -- long
        in practice (measured 4-25 s per real turn).
        """
        from .engine import State
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self._stop.is_set():
                return False
            if self.engine.state in (State.READY, State.LISTENING):
                return True
            time.sleep(0.15)
        return False

    def _analyze(self, item):
        if not self._wait_for_idle():
            self._log_event("mistake_analysis_skipped_busy", turn=item["turn"])
            return
        lv = item["level"] if item["level"] in config.LEVELS else config.DEFAULT_LEVEL
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(level=lv, text=item["text"])}]
        t0 = time.perf_counter()
        raw = ""
        try:
            for delta, _meta in self.llm.stream_chat(
                    msgs, max_tokens=config.MISTAKES_MAX_TOKENS,
                    temperature=0.1, top_p=0.9):
                raw += delta or ""
        except Exception as e:
            self._log_event("mistake_analysis_error", turn=item["turn"],
                            detail=f"{type(e).__name__}: {e}")
            return
        ms = (time.perf_counter() - t0) * 1000
        self.analysis_ms.append(ms)
        self.n_analyzed += 1

        obj = _extract_json(raw)
        if not obj or not isinstance(obj.get("mistakes"), list):
            self._log_event("mistake_parse_failed", turn=item["turn"],
                            raw=raw[:300], analysis_ms=round(ms, 1))
            return

        accepted = []
        for m in obj["mistakes"][:config.MISTAKES_MAX_PER_TURN]:
            rec = self._accept(m, item)
            if rec:
                accepted.append(rec)

        self._log_event("mistake_analysis", turn=item["turn"],
                        analysis_ms=round(ms, 1),
                        n_raw=len(obj["mistakes"]), n_accepted=len(accepted),
                        text=item["text"], audio_file=item["audio_file"],
                        avg_logprob=item["avg_logprob"],
                        mistakes=accepted,
                        status="PROVISIONAL_pending_E2")
        for rec in accepted:
            self.on_result(rec)

    # -------------------------------------------------------- vocabulary
    def _analyze_vocab(self, item):
        """One card, or nothing. Never raises into the worker loop."""
        if not self._wait_for_idle():
            self._log_event("vocab_skipped_busy", turn=item["turn"])
            return
        t0 = time.time()
        self.n_vocab_probes += 1
        lv = item["level"] if item["level"] in config.LEVELS else config.DEFAULT_LEVEL
        card = vocab_mod.analyze(self.llm, lv, item.get("text") or "",
                                 item.get("teacher_text") or "",
                                 item.get("offered") or [],
                                 item.get("known") or set())
        if not card:
            self._log_event("vocab_none", turn=item["turn"],
                            offered=item.get("offered"),
                            analysis_ms=round((time.time() - t0) * 1000, 1))
            return
        key = card["word"].strip().lower()
        with self._lock:
            if key in self._vocab_keys:
                return
            self._vocab_keys.add(key)
            card = {**card, "id": f"v{len(self.vocab) + 1}",
                    "turn": item["turn"], "created_at": time.time(),
                    "saved": False}
            self.vocab.append(card)
        self._log_event("vocab_card", turn=item["turn"], card=card,
                        analysis_ms=round((time.time() - t0) * 1000, 1))
        try:
            self.on_vocab(dict(card))
        except Exception:
            pass

    def save_vocab(self, vocab_id: str, saved: bool = True) -> dict | None:
        """The learner keeping a word IS the useful signal -- it is the only
        place where he tells us that a word mattered to him."""
        with self._lock:
            for c in self.vocab:
                if c["id"] == vocab_id:
                    c["saved"] = bool(saved)
                    self._log_event("vocab_saved", word=c["word"], saved=saved)
                    return dict(c)
        return None

    def _accept(self, m, item) -> dict | None:
        """Filter hard. A wrong correction is worse than a missing one."""
        if not isinstance(m, dict):
            return None
        orig = (m.get("original") or "").strip()
        better = (m.get("better") or "").strip()
        mtype = (m.get("type") or "").strip().lower()
        expl = (m.get("explanation") or "").strip()
        try:
            conf = float(m.get("confidence", 0))
        except Exception:
            conf = 0.0

        if not orig or not better:
            return None
        if conf < self.min_confidence:
            return None
        if mtype not in USEFUL_TYPES:
            return None
        # a "correction" that changes nothing, or only punctuation/case
        if _norm(orig) == _norm(better):
            return None
        # the model must be quoting the learner, not inventing the sentence
        if _norm(orig) not in _norm(item["text"]):
            return None
        # keep explanations to a glance
        if len(expl.split()) > 16:
            expl = " ".join(expl.split()[:16])

        key = f"{mtype}:{_norm(orig)}"
        with self._lock:
            if key in self._by_key:
                rec = self._by_key[key]
                rec["count"] += 1
                rec["turns"].append(item["turn"])
                rec["updated_at"] = time.time()
                return dict(rec)
            rec = {
                "id": f"m{len(self.items)+1}",
                "key": key,
                "original": orig, "better": better,
                "type": mtype, "family": family_of(mtype), "explanation": expl,
                "confidence": round(conf, 2),
                "count": 1, "turns": [item["turn"]],
                "session_turn": item["turn"],
                "audio_file": item["audio_file"],
                "learner_text": item["text"],
                "avg_logprob": item["avg_logprob"],
                "status": "PROVISIONAL_pending_E2",
                "created_at": time.time(), "updated_at": time.time(),
                "user_flag": None,
            }
            self._by_key[key] = rec
            self.items.append(rec)
            return dict(rec)

    # ------------------------------------------------------------ feedback
    def flag(self, mistake_id: str, flag: str):
        """User feedback. 'not_said' is the important one -- it means Whisper
        misheard, and it is exactly the evidence E2 needs."""
        with self._lock:
            for rec in self.items:
                if rec["id"] == mistake_id:
                    rec["user_flag"] = flag
                    self._log_event(
                        "mistake_user_flag", mistake_id=mistake_id, flag=flag,
                        original=rec["original"], better=rec["better"],
                        learner_text=rec["learner_text"],
                        audio_file=rec["audio_file"],
                        avg_logprob=rec["avg_logprob"],
                        asr_disputed=(flag == "not_said"))
                    return dict(rec)
        return None

    @property
    def stats(self) -> dict:
        import statistics
        return {
            "n_analyzed": self.n_analyzed,
            "n_shown": len(self.items),
            "n_suppressed_asr_uncertain": self.n_suppressed_asr,
            "n_skipped_too_short": self.n_skipped_short,
            "median_analysis_ms": round(statistics.median(self.analysis_ms), 1)
            if self.analysis_ms else None,
            "n_asr_disputed": sum(1 for i in self.items
                                  if i.get("user_flag") == "not_said"),
            "n_vocab_cards": len(self.vocab),
            "n_vocab_probes": self.n_vocab_probes,
            "n_vocab_saved": sum(1 for c in self.vocab if c.get("saved")),
            "n_grammar": sum(1 for i in self.items if i.get("family") == "grammar"),
            "n_style": sum(1 for i in self.items if i.get("family") == "style"),
            "status": "PROVISIONAL_pending_E2",
        }

"""The learner model: what the teacher knows about the student.

⚑ WHY THIS EXISTS. The conversation-partner version of this app deliberately had
no memory and no level inference -- "all of that is gated on evidence we do not
have yet" (engine.py docstring). A TEACHER cannot work that way: adapting
difficulty, avoiding repeated topics and returning to a recurring weakness are
the difference between a teacher and a chat partner. So the model exists, but it
is kept deliberately small and honest:

  * Everything in here is COUNTED from the session, never invented. The same
    rule the post-session evaluation follows (measured vs judgment) applies:
    this file only measures.
  * `difficulty` is a within-level dial, not a CEFR assessment. It moves by at
    most DIFFICULTY_STEP per turn, so it cannot jump, and it NEVER changes the
    CEFR level by itself -- it only raises a suggestion the learner may accept.
  * Nothing here is load-bearing. If the profile file is missing or corrupt the
    session runs exactly as before, with an empty profile.

The profile lives next to the session logs in %LOCALAPPDATA%\\GermanPartner, not
in the repository and not in the OneDrive-synced project folder -- same rule as
settings.py.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import tempfile
import time
import unicodedata
from collections import Counter

from . import config

SCHEMA = 1

# Subordinate-clause markers. Used ONLY as a coarse complexity signal for the
# difficulty dial and as a hint to the teacher about which structures the
# learner already dares to use. Not grammar analysis, and never shown as one.
SUBORDINATORS = ("weil", "obwohl", "wenn", "dass", "damit", "während",
                 "bevor", "nachdem", "falls", "sobald", "seitdem", "obgleich")
CONNECTORS = ("deshalb", "trotzdem", "außerdem", "allerdings", "einerseits",
              "andererseits", "zum beispiel", "meiner meinung nach")

# German function words carry no information about vocabulary range, so they are
# excluded from the "words the learner actually commands" count.
STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "eines", "und", "oder", "aber", "ich", "du", "er", "sie", "es",
    "wir", "ihr", "mich", "mir", "dich", "dir", "sich", "ist", "bin", "bist",
    "sind", "war", "waren", "habe", "hast", "hat", "haben", "hatte", "nicht",
    "kein", "keine", "auch", "so", "wie", "was", "wer", "wo", "wann", "warum",
    "in", "im", "an", "am", "auf", "aus", "bei", "mit", "nach", "von", "vom",
    "zu", "zum", "zur", "für", "über", "unter", "vor", "ja", "nein", "doch",
    "noch", "schon", "sehr", "mehr", "man", "mal", "dann", "denn", "hier",
    "da", "dieser", "diese", "dieses", "als", "wenn", "weil", "dass", "sich",
}

# ⚑ A short answer is the single most important teaching signal in this product:
# it is the moment the Speaking Expander has to fire. Measured against the real
# sessions in %LOCALAPPDATA%\GermanPartner: the learner's median turn was 8-12
# words, so "short" has to sit below that or it fires on every turn.
SHORT_TURN_WORDS = 6
LONG_TURN_WORDS = 18

DIFFICULTY_STEP = 0.04          # per turn, maximum. Gradual by construction.
DIFFICULTY_START = 0.35
# Consecutive turns at the ceiling before the learner is ASKED about the next
# CEFR level. Never automatic -- see module docstring.
LEVEL_SUGGEST_STREAK = 8


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", (s or "").lower())
    return re.sub(r"[^\wäöüß\s]", " ", s)


def words_of(s: str) -> list[str]:
    return [w for w in _norm(s).split() if w]


def content_words(s: str) -> list[str]:
    return [w for w in words_of(s) if len(w) > 3 and w not in STOPWORDS]


# ───────────────────────────────────────────────────────── persistent profile
class LearnerProfile:
    """What survives between sessions. Small, readable JSON, atomic writes."""

    PATH = config.DATA_DIR / "learner_profile.json"

    DEFAULTS = {
        "schema": SCHEMA,
        "created_at": None,
        "updated_at": None,
        "sessions": 0,
        "total_turns": 0,
        "level": config.DEFAULT_LEVEL,
        "difficulty": DIFFICULTY_START,
        # {type: count} over confirmed corrections, all sessions
        "error_counts": {},
        # the few concrete corrections worth bringing back
        "error_examples": [],
        # words the teacher introduced, with the card the UI showed
        "vocabulary": [],
        # topic ids already practised, newest last
        "topics_done": [],
        "topic_domains": {},
        "strengths": [],
        "goals": [],
        "speaking": {"words_per_turn_median": None, "long_turn_share": None},
        "history": [],           # one small record per session
    }

    def __init__(self, data: dict | None = None):
        self.data = dict(self.DEFAULTS)
        if data:
            for k in self.DEFAULTS:
                if k in data:
                    self.data[k] = data[k]

    # ------------------------------------------------------------- load/save
    @classmethod
    def load(cls) -> "LearnerProfile":
        try:
            raw = json.loads(cls.PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return cls(raw)
        except FileNotFoundError:
            pass
        except Exception:
            # A corrupt profile must never stop a lesson. It is rebuilt from
            # this session onwards; the broken file is kept for inspection.
            try:
                cls.PATH.replace(cls.PATH.with_suffix(
                    f".broken-{time.strftime('%Y%m%d-%H%M%S')}.json"))
            except Exception:
                pass
        return cls()

    def save(self) -> bool:
        self.data["updated_at"] = time.time()
        if not self.data.get("created_at"):
            self.data["created_at"] = self.data["updated_at"]
        try:
            self.PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.PATH.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.PATH)
            return True
        except Exception:
            return False

    # --------------------------------------------------------------- getters
    @property
    def recurring_errors(self) -> list[tuple[str, int]]:
        c = Counter(self.data.get("error_counts") or {})
        return [(t, n) for t, n in c.most_common(4) if n >= 2]

    @property
    def recent_topics(self) -> list[str]:
        return list(self.data.get("topics_done") or [])[-12:]

    @property
    def known_words(self) -> set[str]:
        return {(v.get("word") or "").lower()
                for v in (self.data.get("vocabulary") or [])}

    def last_topic_domain(self) -> str | None:
        h = self.data.get("history") or []
        return (h[-1] or {}).get("main_domain") if h else None


# ────────────────────────────────────────────────────────────── session model
class LearnerModel:
    """Session state + the persistent profile behind it.

    The engine owns one of these. It is fed from three places, all of which
    already existed: the transcript of each learner turn, the Mistakes Panel's
    accepted corrections, and the vocabulary coach's cards.
    """

    def __init__(self, level: str, synthetic: bool = False):
        self.profile = LearnerProfile.load()
        # ⚑ A HARNESS RUN IS NOT A LESSON. telemetry.py already refuses to let a
        # synthetic or LENA_HARNESS session count as product evidence; the same
        # rule has to hold here, or every `regress.cmd` run would add sessions,
        # topics and vocabulary to the real learner's profile and quietly
        # corrupt the thing that decides what he is taught next.
        # The profile is still READ, so harnesses exercise the real code path.
        self.record = (not synthetic) and not os.environ.get("LENA_HARNESS")
        self.level = level
        self.difficulty = float(self.profile.data.get("difficulty") or DIFFICULTY_START)
        self.turns = 0
        self.words: list[int] = []
        self.voiced_s = 0.0
        self.short_streak = 0
        self.long_streak = 0
        self.high_streak = 0                 # consecutive turns at the ceiling
        self.last_text = ""
        self.last_words = 0
        self.session_errors: list[dict] = []
        self.session_vocab: list[dict] = []
        self.distinct: set[str] = set()
        self.used_subordinators: set[str] = set()
        self.low_conf_turns = 0
        # conversation direction
        self.topic_id: str | None = None
        self.topic_domain: str | None = None
        self.turns_on_topic = 0
        self.topics_this_session: list[str] = []
        self.phase = "warmup"
        self.turns_since_review = 0
        self.turns_since_vocab = 0
        self.last_review_turn = 0
        self.teacher_openings: list[str] = []
        self.level_suggested = False
        self.started_at = time.time()

    # ------------------------------------------------------------- observers
    def observe_learner_turn(self, text: str, voiced_s: float | None = None,
                             pauses: int | None = None, low_conf: bool = False):
        self.turns += 1
        self.turns_on_topic += 1
        self.turns_since_review += 1
        self.turns_since_vocab += 1
        w = words_of(text)
        n = len(w)
        self.words.append(n)
        self.last_text = text or ""
        self.last_words = n
        self.distinct.update(content_words(text))
        for s in SUBORDINATORS:
            if s in w:
                self.used_subordinators.add(s)
        if voiced_s:
            self.voiced_s += float(voiced_s)
        if low_conf:
            self.low_conf_turns += 1

        if n <= SHORT_TURN_WORDS:
            self.short_streak += 1
            self.long_streak = 0
        elif n >= LONG_TURN_WORDS:
            self.long_streak += 1
            self.short_streak = 0
        else:
            self.short_streak = 0
            self.long_streak = 0

        self._adapt(n, w, low_conf, pauses)
        self._advance_phase()

    def observe_teacher_turn(self, text: str):
        """Repetition avoidance: remember how the teacher opened her turns."""
        w = words_of(text)
        if w:
            self.teacher_openings.append(" ".join(w[:4]))
            del self.teacher_openings[:-8]

    def observe_mistake(self, rec: dict):
        """One accepted correction from the Mistakes Panel."""
        if not rec:
            return
        key = (rec.get("type"), (rec.get("original") or "").lower())
        if any((m.get("type"), (m.get("original") or "").lower()) == key
               for m in self.session_errors):
            return
        self.session_errors.append(dict(rec))
        # a fresh error nudges difficulty down slightly -- but only slightly,
        # because errors are the normal state of a learner who is trying.
        self.difficulty = max(0.0, self.difficulty - DIFFICULTY_STEP / 3)

    def observe_vocab(self, card: dict):
        if not card or not card.get("word"):
            return
        if any((v.get("word") or "").lower() == card["word"].lower()
               for v in self.session_vocab):
            return
        self.session_vocab.append(dict(card))

    def start_topic(self, topic_id: str | None, domain: str | None):
        self.topic_id = topic_id
        self.topic_domain = domain
        self.turns_on_topic = 0
        if topic_id:
            self.topics_this_session.append(topic_id)

    def mark_review_done(self):
        self.turns_since_review = 0
        self.last_review_turn = self.turns

    def mark_vocab_done(self):
        self.turns_since_vocab = 0

    # ------------------------------------------------------- difficulty dial
    def _adapt(self, n_words: int, w: list[str], low_conf: bool, pauses):
        """Move the dial by at most DIFFICULTY_STEP. Gradual by construction."""
        signal = 0.0
        if n_words >= LONG_TURN_WORDS:
            signal += 1.0
        elif n_words >= 12:
            signal += 0.5
        elif n_words <= SHORT_TURN_WORDS:
            signal -= 0.8
        if any(s in w for s in SUBORDINATORS):
            signal += 0.5
        if any(c in " ".join(w) for c in CONNECTORS):
            signal += 0.3
        if low_conf:
            signal -= 0.3
        if pauses is not None and pauses >= 4 and n_words < 12:
            signal -= 0.3
        # recent error density in this session
        if self.turns >= 3:
            density = len(self.session_errors) / max(1, self.turns)
            if density > 0.8:
                signal -= 0.5
            elif density < 0.25:
                signal += 0.2

        delta = max(-1.0, min(1.0, signal)) * DIFFICULTY_STEP
        self.difficulty = max(0.0, min(1.0, self.difficulty + delta))
        self.high_streak = self.high_streak + 1 if self.difficulty >= 0.9 else 0

    @property
    def difficulty_label(self) -> str:
        d = self.difficulty
        return "leicht" if d < 0.33 else ("mittel" if d < 0.7 else "fordernd")

    def wants_level_up(self) -> bool:
        """True at most once per session, and only as a QUESTION to the user."""
        if self.level_suggested or self.level == "B2":
            return False
        if self.high_streak >= LEVEL_SUGGEST_STREAK and self.turns >= 10:
            self.level_suggested = True
            return True
        return False

    def speech_scale_factor(self) -> float:
        """Small within-level speaking-rate adjustment.

        ⚑ Bounded on purpose. `length_scale` is Piper's DURATION multiplier
        (higher = slower, see config.LEVELS), and the per-level values are
        frozen. This only nudges within +-6% around the level's value, so a
        struggling learner hears slightly slower German and a confident one
        slightly quicker, without touching the frozen table.
        """
        if not config.TEACHER_ADAPT_SPEECH_RATE:
            return 1.0
        return round(1.06 - 0.12 * self.difficulty, 3)

    # ------------------------------------------------------------- lifecycle
    def _advance_phase(self):
        """Phases are internal. The learner never sees them and the teacher
        never announces them -- they only bias what the director asks for."""
        t = self.turns
        if t <= 2:
            self.phase = "warmup"
        elif t <= 6:
            self.phase = "main"
        elif self.turns_since_review >= config.TEACHER_REVIEW_EVERY_TURNS \
                and self.session_errors:
            self.phase = "review"
        elif t % config.TEACHER_CHALLENGE_EVERY_TURNS == 0:
            self.phase = "challenge"
        else:
            self.phase = "deepen"

    # -------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """What the UI progress panel shows. Counted, never judged."""
        wpt = round(statistics.median(self.words), 1) if self.words else None
        long_share = (round(sum(1 for x in self.words if x >= LONG_TURN_WORDS)
                            / len(self.words), 2) if self.words else None)
        return {
            "turns": self.turns,
            "level": self.level,
            "difficulty": round(self.difficulty, 2),
            "difficulty_label": self.difficulty_label,
            "phase": self.phase,
            "words_per_turn_median": wpt,
            "long_turn_share": long_share,
            "short_turn_streak": self.short_streak,
            "distinct_content_words": len(self.distinct),
            "speaking_s": round(self.voiced_s, 1),
            "speaking_min": round(self.voiced_s / 60, 1),
            "new_words": len(self.session_vocab),
            "corrections": len(self.session_errors),
            "structures_used": sorted(self.used_subordinators),
            "topic": self.topic_id,
            "topic_domain": self.topic_domain,
            "sessions_before": self.profile.data.get("sessions", 0),
        }

    def prompt_state(self) -> dict:
        """Everything teaching.build_system_prompt is allowed to see."""
        p = self.profile
        return {
            "level": self.level,
            "difficulty": self.difficulty,
            "difficulty_label": self.difficulty_label,
            "sessions_before": p.data.get("sessions", 0),
            "recurring_errors": p.recurring_errors,
            "session_errors": self.session_errors[-6:],
            "structures_used": sorted(self.used_subordinators),
            "recent_topics": p.recent_topics,
            "topics_this_session": self.topics_this_session,
            "known_words": sorted(p.known_words)[:40],
            "session_vocab": [v.get("word") for v in self.session_vocab],
            "words_per_turn_median": (round(statistics.median(self.words), 1)
                                      if self.words else None),
            "goals": p.data.get("goals") or [],
        }

    # ---------------------------------------------------------------- persist
    def persist(self, session_id: str | None = None):
        """Fold this session into the durable profile. Best-effort.

        Returns False without writing anything for synthetic and harness runs --
        see the note in __init__.
        """
        if not self.record:
            return False
        try:
            d = self.profile.data
            d["schema"] = SCHEMA
            d["sessions"] = int(d.get("sessions") or 0) + 1
            d["total_turns"] = int(d.get("total_turns") or 0) + self.turns
            d["level"] = self.level
            d["difficulty"] = round(self.difficulty, 3)

            counts = Counter(d.get("error_counts") or {})
            for m in self.session_errors:
                if m.get("user_flag") == "not_said":
                    continue          # a mishearing is not a learner error
                counts[m.get("type") or "?"] += 1
            d["error_counts"] = dict(counts)

            ex = list(d.get("error_examples") or [])
            seen = {(e.get("original"), e.get("better")) for e in ex}
            for m in self.session_errors:
                if m.get("user_flag") == "not_said":
                    continue
                k = (m.get("original"), m.get("better"))
                if k in seen:
                    for e in ex:
                        if (e.get("original"), e.get("better")) == k:
                            e["count"] = int(e.get("count") or 1) + 1
                            e["last_seen"] = time.time()
                    continue
                seen.add(k)
                ex.append({"original": m.get("original"), "better": m.get("better"),
                           "type": m.get("type"), "count": 1,
                           "last_seen": time.time()})
            d["error_examples"] = ex[-40:]

            vocab = list(d.get("vocabulary") or [])
            have = {(v.get("word") or "").lower() for v in vocab}
            for c in self.session_vocab:
                if (c.get("word") or "").lower() in have:
                    for v in vocab:
                        if (v.get("word") or "").lower() == c["word"].lower():
                            v["times_seen"] = int(v.get("times_seen") or 1) + 1
                    continue
                vocab.append({**c, "first_seen": time.time(), "times_seen": 1})
            d["vocabulary"] = vocab[-300:]

            topics = list(d.get("topics_done") or []) + self.topics_this_session
            d["topics_done"] = topics[-60:]
            dom = Counter(d.get("topic_domains") or {})
            if self.topic_domain:
                dom[self.topic_domain] += 1
            d["topic_domains"] = dict(dom)

            wpt = round(statistics.median(self.words), 1) if self.words else None
            d["speaking"] = {
                "words_per_turn_median": wpt,
                "long_turn_share": (round(sum(1 for x in self.words
                                              if x >= LONG_TURN_WORDS) / len(self.words), 2)
                                    if self.words else None),
            }
            hist = list(d.get("history") or [])
            hist.append({
                "session_id": session_id,
                "at": time.time(),
                "level": self.level,
                "difficulty": round(self.difficulty, 3),
                "turns": self.turns,
                "speaking_s": round(self.voiced_s, 1),
                "new_words": len(self.session_vocab),
                "corrections": len(self.session_errors),
                "words_per_turn_median": wpt,
                "main_domain": self.topic_domain,
                "topics": self.topics_this_session,
            })
            d["history"] = hist[-40:]
            return self.profile.save()
        except Exception:
            return False

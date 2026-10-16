"""Post-session evaluation. Runs AFTER the conversation, never during it.

⚑ THE CENTRAL RULE: measured data and model judgment are kept in separate
branches of the output and are never mixed. Everything under `measured` is
counted from telemetry and can be recomputed from the session file. Everything
under `judgment` is Gemma's opinion, is labelled as such, and uses broad bands
rather than invented precision. There is no overall score, because a single
conversation cannot support one.

⚑ SECTION F (the teacher / conversation quality) IS MEASURED ONLY, DELIBERATELY.
Asking the model that produced the conversation to grade that conversation is
not evidence. Question rate, length drift, repetition and ASR-damaged turns are
all countable, so they are counted; nothing is asked of the model here.

⚑ ASR-DISPUTED TURNS ARE NOT LEARNER MISTAKES. A card the user flagged
"nicht das, was ich gesagt habe" is excluded from confirmed mistakes and
reported separately. Turns Whisper was unsure about are excluded from the
grammar picture too. The Mistakes Panel remains PROVISIONAL pending E2, and the
evaluation says so.

⚑ NO CEFR CERTIFICATION. The wording is always "this session showed B1-like
ability in X", never "your level is B1".
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
import unicodedata
from collections import Counter

from . import config

SCHEMA = 1

# ⚑ ONLY tokens that are unambiguously English AND are not German words or
# established German loanwords. Telling the learner they switched to English
# when they did not is worse than missing a switch, so this errs heavily toward
# missing, and the matched terms are always reported verbatim so the user can
# check them.
#
# Deliberately EXCLUDED after the first run flagged 7 turns on the word "was":
#   German homographs : was, will, so, war, man, bad, in, an, wer, die, hat
#   German loanwords  : Job, Interview, Meeting, Service, Shop, Problem,
#                       Student, Team, Baby, Computer, Handy
# Every one of those would produce a false accusation of code-switching.
EN_HINTS = {
    "the", "and", "but", "because", "really", "actually", "maybe",
    "always", "never", "something", "everything", "nothing", "beautiful",
    "people", "friend", "friends", "money", "week", "water", "house", "home",
    "school", "teacher", "doctor", "customer", "company", "question",
    "answer", "example", "pharmacy", "charger", "nice", "good",
    "i", "you", "he", "she", "they", "we", "are", "were", "have", "has",
    "does", "did", "would", "my", "your", "there", "their", "about",
    "with", "from", "what", "when", "where", "which", "who", "how",
}

# ⚑ The user must never see "StallError". These are the only strings the UI
# gets for a failed judgment; the exception detail goes to telemetry.
FAILURE_TEXT = {
    "llm_unavailable": "Die Einschätzung war diesmal nicht möglich — "
                       "die Auswertung konnte das Sprachmodell nicht erreichen.",
    "llm_timeout": "Die Einschätzung hat zu lange gebraucht und wurde abgebrochen.",
    "llm_error": "Die Einschätzung war diesmal nicht möglich.",
    # ⚑ Four distinct engineering causes, ONE sentence for the learner. The
    # classes are kept apart in telemetry because they have different fixes;
    # the user does not care which flavour of unreadable it was.
    "parse_failed": "Die Einschätzung war diesmal nicht lesbar.",
    "bad_syntax": "Die Einschätzung war diesmal nicht lesbar.",
    "truncated": "Die Einschätzung war diesmal nicht lesbar.",
    "empty_content": "Die Einschätzung war diesmal nicht lesbar.",
    "schema_invalid": "Die Einschätzung war diesmal nicht lesbar.",
    "insufficient_evidence": "Zu wenige sicher verstandene Redebeiträge für eine "
                             "sinnvolle Einschätzung.",
}

SYSTEM = ("Du bist ein erfahrener, wohlwollender Deutschlehrer. Du antwortest "
          "AUSSCHLIESSLICH mit gültigem JSON, ohne Text davor oder danach, "
          "ohne Markdown, ohne Codeblöcke.")

# ─────────────────────────────────────────────── constrained output (E13)
# ⚑ THE JUDGMENT USED TO FAIL TO PARSE ABOUT ONE SESSION IN EIGHT.
#
# Measured on 30 requests against three real frozen sessions, through the real
# streaming path: 25/30 first-attempt successes. EVERY failure was the same
# defect -- Gemma emitting a bare "" where a key/value pair belongs:
#
#     {"speaking":{"band":"noch im Aufbau","evidence":"...",""},
#      "sustaining":{"band":"heute noch nicht gezeigt","evidence":"",""}, ...
#
# which is not JSON. It clustered on thin-evidence sessions (4 of 5 failures on
# the 5-turn one), i.e. exactly where a field had nothing to say and the model
# padded the object. Two attempts made it ~1/8 sessions with no judgment at all.
#
# ⚑ Truncation was NOT a cause, contrary to first impression: median 171 output
# tokens against a 460 budget, maximum 199. The one reply that looked cut off
# was cut off by the debug print that captured it. Budgets below are still
# raised to a measured multiple of the observed maximum, but that is headroom,
# not the fix.
#
# THE FIX IS PREVENTION, NOT REPAIR. llama.cpp compiles a JSON schema into a
# GBNF grammar and constrains sampling to it, so a token that would break the
# structure cannot be emitted at all. There is no post-hoc regex repair here
# and there must not be one: rewriting invalid JSON risks changing what the
# evaluation says about the learner.
#
# Verified against the pinned b10298 runtime, not assumed:
#   response_format {"type":"json_schema", ...}   works, finish_reason "stop"
#   top-level "json_schema"                        also works (same result)
#   "grammar" (raw GBNF)                           HTTP 400 -- not accepted here
# ⚑ With a loose schema and a tight budget the server was observed to route the
# whole generation into `reasoning_content`, leaving `content` EMPTY despite
# --reasoning off. Bounded fields plus the budgets below avoid it, and the
# empty-content case is still detected and classified rather than assumed away.
_BANDS = ["sicher", "gut", "noch im Aufbau", "heute noch nicht gezeigt"]
_DEPTHS = ["nur geantwortet", "eigene Beiträge", "ausführlich"]


def _section(evidence_max: int) -> dict:
    return {"type": "object",
            "properties": {"band": {"type": "string", "enum": _BANDS},
                           "evidence": {"type": "string", "maxLength": evidence_max}},
            "required": ["band", "evidence"], "additionalProperties": False}


SPEAKING_SCHEMA = {
    "type": "object",
    "properties": {
        "speaking": _section(220),
        "sustaining": _section(180),
        "complexity": _section(180),
        "topics": {"type": "array", "minItems": 1, "maxItems": 4,
                   "items": {"type": "object",
                             "properties": {
                                 "topic": {"type": "string", "maxLength": 60},
                                 "depth": {"type": "string", "enum": _DEPTHS},
                                 "evidence": {"type": "string", "maxLength": 160}},
                             "required": ["topic", "depth", "evidence"],
                             "additionalProperties": False}},
        "strengths": {"type": "array", "minItems": 1, "maxItems": 3,
                      "items": {"type": "string", "maxLength": 120}},
    },
    "required": ["speaking", "sustaining", "complexity", "topics", "strengths"],
    "additionalProperties": False,
}

FOCUS_SCHEMA = {
    "type": "object",
    "properties": {
        "next_focus": {"type": "array", "minItems": 1, "maxItems": 3,
                       "items": {"type": "object",
                                 "properties": {
                                     "focus": {"type": "string", "maxLength": 160},
                                     "why": {"type": "string", "maxLength": 110}},
                                 "required": ["focus", "why"],
                                 "additionalProperties": False}},
        "expressions": {"type": "array", "minItems": 2, "maxItems": 4,
                        "items": {"type": "object",
                                  "properties": {
                                      "phrase": {"type": "string", "maxLength": 90},
                                      "use": {"type": "string", "maxLength": 90}},
                                  "required": ["phrase", "use"],
                                  "additionalProperties": False}},
    },
    "required": ["next_focus", "expressions"], "additionalProperties": False,
}

# Measured maximum was 199 output tokens for `speaking`. These are deliberate
# headroom over that, not a guess, and the evaluation runs after the session so
# a slower worst case costs the learner nothing.
SPEAKING_MAX_TOKENS = 700
FOCUS_MAX_TOKENS = 480

PROMPT_SPEAKING = """Hier sind die Redebeiträge eines Deutschlerners aus einem
Gespräch (Niveau ungefähr {level}). Der Text stammt aus automatischer
Spracherkennung und kann Hörfehler enthalten.

{turns}

Beurteile NUR den Lerner, nicht den Gesprächspartner.

Wichtig:
- Bewerte nicht negativ, nur weil der Lerner langsamer als ein Muttersprachler ist.
- Sei konkret und beziehe dich auf das, was er wirklich gesagt hat.
- Keine erfundene Präzision, keine Prozentzahlen, kein CEFR-Niveau.
- Erkenne an, was schon funktioniert.
- Sprich über DIESES Gespräch ("heute", "in diesem Gespräch"), nicht über sein
  Deutsch allgemein.
{evidence_note}

Erlaubte Werte für "band": "sicher", "gut", "noch im Aufbau",
"heute noch nicht gezeigt".

Du bist Coach, nicht Prüfer. Formuliere Schwächen als das, was als Nächstes
dran ist, nicht als Urteil. Wenn etwas schwach war, sag es klar -- aber
beschreibend ("die Sätze blieben kurz") statt abwertend.

Antworte exakt so:
{{"speaking":{{"band":"<...>","evidence":"<max 25 Wörter, konkret>"}},
"sustaining":{{"band":"<...>","evidence":"<max 20 Wörter>"}},
"complexity":{{"band":"<...>","evidence":"<max 20 Wörter>"}},
"topics":[{{"topic":"<kurz>","depth":"<nur geantwortet|eigene Beiträge|ausführlich>",
"evidence":"<max 18 Wörter>"}}],
"strengths":["<max 12 Wörter>","<max 12 Wörter>"]}}"""

PROMPT_FOCUS = """Ein Deutschlerner hatte ein Gespräch. Daten:

Wiederkehrende Fehlerarten: {patterns}
Beispiel-Korrekturen: {examples}
Themen: {topics}
Wörter pro Redebeitrag (Median): {wpt}

Gib SEHR WENIG zurück: höchstens 3 Punkte, am besten 1 Hauptfokus.
Kein Hausaufgabenkatalog. Formuliere es als etwas, das er beim nächsten
Sprechen ausprobieren kann, nicht als Grammatikregel.

Nenne außerdem 2 bis 4 natürliche deutsche Wendungen, die ihm bei genau diesen
Themen beim nächsten Mal helfen würden.

Antworte exakt so:
{{"next_focus":[{{"focus":"<max 18 Wörter, auf Deutsch>","why":"<max 12 Wörter>"}}],
"expressions":[{{"phrase":"<deutsche Wendung>","use":"<max 10 Wörter>"}}]}}"""


def _norm(s):
    s = unicodedata.normalize("NFC", (s or "").lower())
    return re.sub(r"[^\wäöüß\s]", " ", s)


def _words(s):
    return [w for w in _norm(s).split() if w]


def prompt_version() -> str:
    """Hash of the exact prompts AND schemas, so a stored evaluation can be tied
    to what produced it. Change a prompt or a schema -> this changes.

    ⚑ The schemas are part of this on purpose: with constrained decoding the
    grammar shapes the output as much as the wording does, so an evaluation
    stored under an unchanged prompt hash but a changed schema would be
    mislabelled."""
    h = hashlib.sha256()
    for t in (SYSTEM, PROMPT_SPEAKING, PROMPT_FOCUS):
        h.update(t.encode("utf-8"))
    for s in (SPEAKING_SCHEMA, FOCUS_SCHEMA):
        h.update(json.dumps(s, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    h.update(f"{SPEAKING_MAX_TOKENS}:{FOCUS_MAX_TOKENS}".encode())
    return h.hexdigest()[:16]


def _extract_json(raw: str):
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(raw[i:j + 1])
    except Exception:
        return None


def validate(obj, schema: dict) -> list[str]:
    """Structural validation of a parsed reply. Returns the problems found.

    ⚑ `json.loads` succeeding is not the same as the reply being usable. The
    grammar makes malformed syntax impossible, but a schema is compiled from a
    grammar, not enforced by one, and nothing stops a future runtime change from
    letting an empty object or a wrong band through. This is the second net, and
    it never repairs -- it only reports, so a bad reply is retried or degraded
    rather than shown to the learner as if it meant something.

    Deliberately small: required keys, types, enums, array bounds, non-empty
    evidence. Not a general JSON-Schema implementation, and not a dependency.
    """
    problems: list[str] = []

    def walk(node, sch, path):
        t = sch.get("type")
        if t == "object":
            if not isinstance(node, dict):
                problems.append(f"{path}: expected object, got {type(node).__name__}")
                return
            for key in sch.get("required", []):
                if key not in node:
                    problems.append(f"{path}.{key}: missing")
            if sch.get("additionalProperties") is False:
                for key in node:
                    if key not in sch.get("properties", {}):
                        problems.append(f"{path}.{key}: unexpected key")
            for key, sub in sch.get("properties", {}).items():
                if key in node:
                    walk(node[key], sub, f"{path}.{key}")
        elif t == "array":
            if not isinstance(node, list):
                problems.append(f"{path}: expected array, got {type(node).__name__}")
                return
            lo, hi = sch.get("minItems"), sch.get("maxItems")
            if lo is not None and len(node) < lo:
                problems.append(f"{path}: {len(node)} items, minimum {lo}")
            if hi is not None and len(node) > hi:
                problems.append(f"{path}: {len(node)} items, maximum {hi}")
            for i, item in enumerate(node):
                walk(item, sch.get("items", {}), f"{path}[{i}]")
        elif t == "string":
            if not isinstance(node, str):
                problems.append(f"{path}: expected string, got {type(node).__name__}")
                return
            if sch.get("enum") and node not in sch["enum"]:
                problems.append(f"{path}: {node!r} is not one of {sch['enum']}")

    walk(obj, schema, "")

    # ⚑ An evidence string is the whole point of a band: "gut" with nothing
    # behind it is exactly the invented authority this evaluation avoids.
    # `heute noch nicht gezeigt` is the one band that legitimately has nothing
    # to cite, so it is the one exemption.
    if isinstance(obj, dict):
        for key in ("speaking", "sustaining", "complexity"):
            sec = obj.get(key)
            if isinstance(sec, dict) and sec.get("band") != "heute noch nicht gezeigt" \
                    and not (sec.get("evidence") or "").strip():
                problems.append(f".{key}.evidence: empty for band {sec.get('band')!r}")
    return problems


def span_confidence(turn_rec: dict, span: str):
    """Lowest Whisper word-probability inside the text a correction rewrites.

    ⚑ WHY THIS EXISTS -- audited on the real session s20260810-165738. Turn 2
    had avg_logprob -0.407, i.e. comfortably confident at SEGMENT level, and the
    panel showed:
         "ich bin der Mode" -> "ich bin in der Mode"
    But the words being corrected scored p=0.275 ("der") and p=0.178 ("Mode"),
    the two least-confident words in the entire turn (turn mean 0.784). The
    "mistake" sits exactly where Whisper was guessing.
    The second card from the same turn -- "ich brauche arbeiten" -> "Arbeit" --
    rests on "brauche" p=0.990, and is well supported.

    Segment confidence cannot see this (E9 showed it does not discriminate at
    all), but SPAN confidence localises it. This does NOT change the Mistakes
    Panel: the card is still shown. It only stops a correction built on guessed
    words from becoming evidence for a judgment about the learner's grammar.
    """
    words = turn_rec.get("stt_words") or []
    if not words or not span:
        return None
    want = set(_words(span))
    if not want:
        return None
    probs = [w.get("p") for w in words
             if w.get("w", "").strip().lower().strip(".,?!;:") in want
             and w.get("p") is not None]
    return round(min(probs), 3) if probs else None


# ─────────────────────────────────────────────────────────── measured half
def measure(turns, mistakes, disputed_turns, duration_min, level) -> dict:
    """Everything here is COUNTED. No model involved, fully reproducible."""
    real = [t for t in turns if t.get("input_source") != "synthetic"]
    lw = [t.get("learner_words") or 0 for t in real]
    voiced = [t.get("voiced_s") for t in real if t.get("voiced_s") is not None]
    pauses = [t.get("stt_pauses") or 0 for t in real]
    pause_s = [t.get("stt_pause_s") or 0.0 for t in real]
    ai_w = sum(t.get("ai_words") or 0 for t in real)
    lw_total = sum(lw)

    # vocabulary spread over what the learner actually said
    all_words = []
    for t in real:
        all_words += _words(t.get("learner_text"))
    distinct = len(set(all_words))

    # possible code-switching, reported verbatim so the user can check
    cs_terms = Counter(w for w in all_words if w in EN_HINTS)
    cs_turns = sum(1 for t in real
                   if any(w in EN_HINTS for w in _words(t.get("learner_text"))))

    # mistakes: disputed ones are NOT learner mistakes
    confirmed = [m for m in mistakes if m.get("user_flag") != "not_said"]
    disputed = [m for m in mistakes if m.get("user_flag") == "not_said"]
    by_type = Counter(m.get("type") for m in confirmed)
    grammar = [m for m in confirmed if m.get("family", "grammar") == "grammar"]
    style = [m for m in confirmed if m.get("family") == "style"]

    # ⚑ span-level ASR support for each correction. Weakly-supported ones are
    # still SHOWN by the panel; they simply do not become evidence here.
    by_turn = {t.get("turn"): t for t in real}
    well_supported, weakly_supported = [], []
    for m in confirmed:
        tr = by_turn.get(m.get("session_turn"))
        conf = span_confidence(tr, m.get("original")) if tr else None
        m = dict(m)
        m["span_min_word_p"] = conf
        if conf is not None and conf < config.EVALUATION_MIN_SPAN_WORD_P:
            weakly_supported.append(m)
        else:
            well_supported.append(m)

    # patterns come from well-supported corrections only
    solid_types = Counter(m.get("type") for m in well_supported)
    recurring = [{"type": t, "count": c} for t, c in solid_types.most_common()
                 if c >= 2]

    low_conf = sum(1 for t in real if t.get("stt_low_confidence"))

    return {
        "level_setting": level,
        "duration_min": duration_min,
        "turns": len(real),
        # --- speaking volume -------------------------------------------
        "learner_words": lw_total,
        "ai_words": ai_w,
        "learner_word_share": round(lw_total / (lw_total + ai_w), 3)
        if (lw_total + ai_w) else None,
        # ⚑ actual VOICED seconds, not recording length
        "learner_voiced_s": round(sum(voiced), 1) if voiced else None,
        "learner_voiced_s_median_per_turn": round(statistics.median(voiced), 2)
        if voiced else None,
        "words_per_turn_median": round(statistics.median(lw), 1) if lw else None,
        "words_per_turn_max": max(lw) if lw else None,
        "short_turns_under_5_words": sum(1 for x in lw if x < 5),
        # --- hesitation -------------------------------------------------
        "pauses_per_turn_median": round(statistics.median(pauses), 1) if pauses else None,
        "pause_seconds_total": round(sum(pause_s), 1) if pause_s else None,
        # --- vocabulary -------------------------------------------------
        "distinct_words": distinct,
        "type_token_ratio": round(distinct / len(all_words), 3) if all_words else None,
        # --- code switching (heuristic, terms listed for verification) ---
        "code_switch_turns": cs_turns,
        "code_switch_terms": [w for w, _ in cs_terms.most_common(8)],
        "code_switch_is_heuristic": True,
        # --- correction evidence ----------------------------------------
        "mistakes_confirmed": len(confirmed),
        "mistakes_grammar": len(grammar),
        "mistakes_style": len(style),
        "mistakes_disputed_by_user": len(disputed),
        "mistake_types": dict(by_type),
        "recurring_patterns": recurring,
        # corrections the ASR barely heard -- shown in the panel, but not used
        # as evidence for any statement about the learner
        "mistakes_weak_asr_support": len(weakly_supported),
        "weak_asr_corrections": [
            {"original": m.get("original"), "better": m.get("better"),
             "span_min_word_p": m.get("span_min_word_p")}
            for m in weakly_supported],
        # --- input reliability ------------------------------------------
        "asr_disputed_turns": len(disputed_turns),
        "stt_low_confidence_turns": low_conf,
    }


def measure_partner(turns, session_end) -> dict:
    """Section F -- Lena. MEASURED ONLY. The model does not grade itself."""
    real = [t for t in turns if t.get("input_source") != "synthetic"]
    texts = [(t.get("ai_text_spoken") or "").strip() for t in real]
    texts = [x for x in texts if x]
    if not texts:
        return {"measured": False}
    # ⚑ THE SIGN OF THIS METRIC FLIPPED WITH THE PRODUCT.
    # As a conversation partner, a high question rate was a persona VIOLATION
    # ("höchstens jede zweite"). As a teacher it is the job: the handoff's own
    # §7 finding is that questions INCREASE this learner's output. So the same
    # number is still counted, and still not judged by the model -- only the
    # comparison it is held against changed.
    q = sum(1 for x in texts if x.endswith("?"))
    words = [t.get("ai_words") or 0 for t in real]

    grams = Counter()
    for x in texts:
        w = _words(x)
        for i in range(len(w) - 3):
            grams[" ".join(w[i:i + 4])] += 1
    repeated = [{"phrase": g, "count": c} for g, c in grams.most_common(3) if c > 1]

    # turns where the learner's transcript was unreliable -> Lena was answering
    # something the learner may not have said
    damaged = sum(1 for t in real if t.get("stt_low_confidence"))

    return {
        "measured": True,
        "question_end_rate": round(q / len(texts), 2),
        "teaching_goal_min_question_rate": 0.6,  # teacher.json: fast jeder Beitrag
        "question_rate_meets_teaching_goal": (q / len(texts)) >= 0.6,
        "ai_words_median": round(statistics.median(words), 1) if words else None,
        "ai_words_median_first_half": (session_end or {}).get("ai_words_median_first_half"),
        "ai_words_median_second_half": (session_end or {}).get("ai_words_median_second_half"),
        "question_rate_first_half": (session_end or {}).get("question_end_rate_first_half"),
        "question_rate_second_half": (session_end or {}).get("question_end_rate_second_half"),
        "repeated_phrases": repeated,
        "turns_answering_unreliable_transcript": damaged,
        "note": ("Gemessen, nicht bewertet. Das Modell beurteilt sich nicht "
                 "selbst -- Selbstbewertung wäre kein Beleg."),
    }


def measure_teaching(turns) -> dict:
    """Section G -- did the TEACHING actually do anything? MEASURED ONLY.

    ⚑ This is the section that makes a teaching regression visible. The
    Conversation Director records what it asked for on every turn
    (`teacher_directive` in the turn record), so the effect of the single most
    important move -- the Speaking Expander -- is checkable without asking the
    model anything: after a turn where the director demanded a longer answer,
    did the learner's NEXT turn actually get longer?

    No judgment, no score, and explicitly no claim of causation: a learner can
    get longer for reasons that have nothing to do with the prompt. It is a
    counted rate, reported with its own denominator.
    """
    real = [t for t in turns if t.get("input_source") != "synthetic"]
    if not real:
        return {"measured": False}
    kinds = Counter((t.get("teacher_directive") or "keine") for t in real)

    wins = tries = 0
    for a, b in zip(real, real[1:]):
        if a.get("teacher_directive") == "expand":
            tries += 1
            if (b.get("learner_words") or 0) > (a.get("learner_words") or 0):
                wins += 1

    diffs = [t.get("teacher_difficulty") for t in real
             if t.get("teacher_difficulty") is not None]
    lw = [t.get("learner_words") or 0 for t in real]
    h = len(lw) // 2
    topics = [t.get("teacher_topic") for t in real if t.get("teacher_topic")]
    seen, order = set(), []
    for x in topics:
        if x not in seen:
            seen.add(x)
            order.append(x)

    return {
        "measured": True,
        "directives": dict(kinds),
        "expand_prompts": tries,
        "expand_followed_by_longer_turn": wins,
        "expand_success_rate": round(wins / tries, 2) if tries else None,
        "difficulty_start": diffs[0] if diffs else None,
        "difficulty_end": diffs[-1] if diffs else None,
        "words_per_turn_first_half": (round(statistics.median(lw[:h]), 1)
                                      if h else None),
        "words_per_turn_second_half": (round(statistics.median(lw[h:]), 1)
                                       if len(lw) - h else None),
        "topics": order,
        "note": ("Gezählt aus den Turn-Daten. Kein Beleg für Ursache: ein Lerner "
                 "kann auch ohne Aufforderung länger sprechen."),
    }


# ────────────────────────────────────────────────────────── judgment half
def classify_failure(exc: Exception) -> str:
    """Turn an exception into a stable, non-scary failure class.

    ⚑ LlamaCppProvider.stream_chat wraps EVERY exception in StallError, so the
    class name alone is meaningless -- a dead server and a genuine 45 s hang
    both arrive as "StallError". The real signal is inside: the wrapped error
    name and how long we waited. The production failure that motivated this
    waited 44.9 ms, which is a dead socket, not a stall.
    """
    txt = str(exc)
    waited = None
    try:
        waited = json.loads(txt).get("waited_ms")
        inner = json.loads(txt).get("error", "")
    except Exception:
        inner = txt
    low = (inner or "").lower()
    if "urlerror" in low or "refused" in low or "connection" in low \
            or "10061" in low or "remotedisconnected" in low:
        return "llm_unavailable"
    if "timeout" in low or "timed out" in low:
        return "llm_timeout"
    if waited is not None and waited < 1000:
        return "llm_unavailable"
    return "llm_error"


class SessionEvaluator:
    """Two small LLM calls, after the conversation, on the existing instance."""

    def __init__(self, llm, level: str):
        self.llm = llm
        self.level = level
        self.ms = []
        self.attempts = []          # full record of every try, for telemetry

    def _alive(self) -> bool:
        """Cheap liveness probe. Not a substitute for the generation smoke test
        during boot -- here we only need to know the socket is there before
        spending a request, so a failure can be named precisely."""
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.llm.base}/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _ask(self, prompt: str, max_tokens: int, what: str, schema: dict = None):
        """One bounded, instrumented request. At most EVALUATION_MAX_ATTEMPTS,
        and a retry only happens for failures a fresh provider could fix.

        ⚑ Every attempt records the RAW model output and why it was rejected.
        The judgment used to fail about one session in eight with nothing kept
        but the words "parse_failed", which is not enough to diagnose anything.
        Failure classes are kept apart (`bad_syntax` / `truncated` /
        `empty_content` / `schema_invalid`) because they have different causes
        and different fixes.
        """
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
        kw = {}
        if schema is not None:
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": what,
                                                     "schema": schema,
                                                     "strict": True}}
        last_class = None
        for attempt in range(1, config.EVALUATION_MAX_ATTEMPTS + 1):
            rec = {"what": what, "attempt": attempt,
                   "started_at": time.time(),
                   "provider_generation": getattr(self.llm, "generation", None),
                   "alive_before": self._alive(),
                   "constrained": schema is not None,
                   "max_tokens": max_tokens,
                   "first_token_ms": None, "total_ms": None, "tokens": None,
                   "truncated": None, "raw": None, "schema_problems": None,
                   "status": None, "failure_class": None, "restarted": False}
            t0 = time.perf_counter()
            raw, ntok = "", 0
            try:
                for delta, meta in self.llm.stream_chat(
                        msgs, max_tokens=max_tokens, temperature=0.2, top_p=0.9,
                        **kw):
                    if rec["first_token_ms"] is None and meta.get("first_token_ms"):
                        rec["first_token_ms"] = meta["first_token_ms"]
                    raw += delta or ""
                    ntok = meta.get("tokens", ntok)
                rec["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                rec["tokens"] = ntok
                # The budget being reached is the only reliable truncation
                # signal on the streaming path, which does not carry
                # finish_reason.
                rec["truncated"] = bool(ntok) and ntok >= max_tokens - 2
                rec["raw"] = raw[:2000]

                if not raw.strip():
                    # Observed with a loose schema: the whole generation goes to
                    # `reasoning_content` and never reaches `content`.
                    cls = "empty_content"
                elif (obj := _extract_json(raw)) is None:
                    cls = "truncated" if rec["truncated"] else "bad_syntax"
                elif schema is not None and (problems := validate(obj, schema)):
                    rec["schema_problems"] = problems[:8]
                    cls = "schema_invalid"
                else:
                    cls = None

                if cls:
                    rec["status"] = cls
                    rec["failure_class"] = cls
                    last_class = cls
                    self.attempts.append(rec)
                    continue          # all four are worth exactly one reroll
                rec["status"] = "ok"
                self.ms.append(rec["total_ms"])
                self.attempts.append(rec)
                return obj, None
            except Exception as e:
                rec["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                cls = classify_failure(e)
                rec["status"] = "failed"
                rec["failure_class"] = cls
                last_class = cls
                self.attempts.append(rec)
                # ⚑ DO NOT RESTART llama-server here, and do not retry a dead
                # or hung server.
                #
                # An unreachable LLM at evaluation time is a LIFECYCLE BUG, not
                # a transient fault -- it is exactly the production failure this
                # work fixed (main() was released by _shutdown_done and killed
                # the server mid-evaluation). Papering over it by spawning a
                # fresh llama-server during teardown would hide the bug, cost
                # ~5 s at the worst possible moment, and -- observed in testing
                # -- leave an orphaned process holding port 8790.
                # Retrying is reserved for parse failures, which ARE transient
                # (Gemma produced 1 unparseable reply in 25 real analyses) and
                # cost nothing but one reroll.
                break
        return None, last_class

    def judge(self, turns, m: dict) -> dict:
        real = [t for t in turns if t.get("input_source") != "synthetic"]
        # ⚑ only turns Whisper was reasonably sure about feed the judgment
        usable = [t for t in real if not t.get("stt_low_confidence")]
        lines = "\n".join(f"- {(t.get('learner_text') or '').strip()}"
                          for t in usable if (t.get("learner_text") or "").strip())
        out = {"available": False,
               "usable_turns": len(usable),
               "excluded_uncertain_turns": [t.get("turn") for t in real
                                            if t.get("stt_low_confidence")]}
        if len(usable) < config.EVALUATION_MIN_USABLE_TURNS or not lines.strip():
            out["reason"] = ("Zu wenige sicher verstandene Redebeiträge für eine "
                             "sinnvolle Einschätzung.")
            out["failure_class"] = "insufficient_evidence"
            return out

        # ⚑ EVIDENCE QUANTITY GOVERNS HOW MUCH MAY BE CLAIMED. Five turns cannot
        # support statements about someone's German; they can support statements
        # about what happened today.
        if len(usable) < 6:
            evidence_note = (
                "- ⚑ WENIG MATERIAL: nur %d verwertbare Redebeiträge. Mache "
                "WENIGE, vorsichtige Aussagen. Schreibe ausdrücklich, wenn etwas "
                "in diesem Gespräch nicht gezeigt wurde, statt es zu bewerten. "
                "Keine Aussagen über sein Deutsch allgemein." % len(usable))
        else:
            evidence_note = ("- Stütze jede Aussage auf etwas, das er in diesem "
                             "Gespräch tatsächlich gesagt hat.")
        out["evidence_level"] = "thin" if len(usable) < 6 else "adequate"

        a, fail = self._ask(PROMPT_SPEAKING.format(
            level=self.level, turns=lines[:4000],
            evidence_note=evidence_note), SPEAKING_MAX_TOKENS, "speaking",
            schema=SPEAKING_SCHEMA)
        if not a:
            out["failure_class"] = fail
            out["reason"] = FAILURE_TEXT.get(fail, FAILURE_TEXT["llm_error"])
            return out
        out.update({k: a.get(k) for k in
                    ("speaking", "sustaining", "complexity", "topics", "strengths")})

        pats = ", ".join(f"{p['type']} x{p['count']}"
                         for p in m["recurring_patterns"]) or "keine wiederkehrenden"
        ex = "; ".join(f"{x}" for x in (m.get("_examples") or [])[:3]) or "keine"
        tops = ", ".join(t.get("topic", "") for t in (a.get("topics") or [])[:4]) or "unklar"
        b, _f2 = self._ask(PROMPT_FOCUS.format(
            patterns=pats, examples=ex, topics=tops,
            wpt=m.get("words_per_turn_median")), FOCUS_MAX_TOKENS, "focus",
            schema=FOCUS_SCHEMA)
        if b:
            out["next_focus"] = (b.get("next_focus") or [])[:3]
            out["expressions"] = (b.get("expressions") or [])[:4]
        out["available"] = True
        return out


def build(engine, session_end: dict) -> dict:
    """Assemble the full evaluation. Called after the conversation ends."""
    turns = list(engine.log.turns) if engine.log else []
    mistakes = list(engine.mistakes.items) if engine.mistakes else []
    disputed_turns = [t for t in turns if t.get("asr_disputed")]

    m = measure(turns, mistakes, disputed_turns,
                (session_end or {}).get("duration_min"), engine.level)
    # only well-supported corrections may shape the next-session focus
    weak = {(w["original"], w["better"]) for w in (m.get("weak_asr_corrections") or [])}
    confirmed = [x for x in mistakes
                 if x.get("user_flag") != "not_said"
                 and (x.get("original"), x.get("better")) not in weak]
    m["_examples"] = [f"{x['original']} -> {x['better']}" for x in confirmed[:3]]

    # ⚑ FROZEN INPUT: everything the judge is shown, plus the identity of the
    # prompts and model. This is what makes a failed judgment rerunnable offline
    # without asking the user to repeat the session.
    real_turns = [t for t in turns if t.get("input_source") != "synthetic"]
    frozen = {
        "session_id": engine.log.session_id if engine.log else None,
        "level": engine.level,
        "schema": SCHEMA,
        "prompt_version": prompt_version(),
        "model": config.MODEL_DECISION.get("llm"),
        "model_status": config.MODEL_STATUS,
        "sampling": {"temperature": 0.2, "top_p": 0.9},
        "requested_at": time.time(),
        "selected_turns": [
            {"turn": t.get("turn"), "text": t.get("learner_text")}
            for t in real_turns if not t.get("stt_low_confidence")],
        "excluded_turns": [
            {"turn": t.get("turn"), "text": t.get("learner_text"),
             "reason": "stt_low_confidence",
             "avg_logprob": t.get("stt_avg_logprob")}
            for t in real_turns if t.get("stt_low_confidence")],
        "confirmed_corrections": [
            {"original": x.get("original"), "better": x.get("better"),
             "type": x.get("type"), "turn": x.get("session_turn")}
            for x in mistakes if x.get("user_flag") != "not_said"],
        "excluded_corrections": [
            {"original": x.get("original"), "better": x.get("better"),
             "reason": "user_flagged_not_said"}
            for x in mistakes if x.get("user_flag") == "not_said"],
        "measured": m,
    }

    judgment = {"available": False}
    ev_obj = None
    try:
        ev_obj = SessionEvaluator(engine.llm, engine.level)
        judgment = ev_obj.judge(turns, m)
    except Exception as e:
        judgment = {"available": False,
                    "failure_class": classify_failure(e),
                    "reason": FAILURE_TEXT.get(classify_failure(e),
                                               FAILURE_TEXT["llm_error"])}
    if ev_obj is not None:
        judgment["attempts"] = ev_obj.attempts

    examples = m.pop("_examples", [])
    frozen["measured"] = m
    return {
        "kind": "session_evaluation", "schema": SCHEMA,
        "at": time.time(),
        "session_id": engine.log.session_id if engine.log else None,
        "status": "PROVISIONAL_pending_E2",
        "prompt_version": prompt_version(),
        "evaluation_input": frozen,
        "measured": m,
        "mistake_examples": examples,
        "judgment": judgment,
        "partner": measure_partner(turns, session_end),
        "teaching": measure_teaching(turns),
        "caveats": [
            "Gemessene Werte sind gezählt; Einschätzungen sind Modellmeinung.",
            "Die Spracherkennung kann sich verhören -- als Hörfehler markierte "
            "Korrekturen zählen NICHT als Fehler.",
            "Ein einzelnes Gespräch belegt kein CEFR-Niveau.",
            "Die Fehlerkorrektur ist vorläufig (E2 noch offen).",
        ],
    }

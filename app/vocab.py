"""Vocabulary Coach: turn a word the teacher used into a card the learner keeps.

⚑ IT RUNS WHERE THE MISTAKE ANALYSIS ALREADY RUNS, and for the same reason.
llama-server runs with `-np 1`: one slot. A second background analyser with its
own thread would sit in front of the learner's next turn. So a vocabulary probe
is just another job on the MistakeAnalyzer queue -- same worker, same
wait-for-idle gate, same single-flight discipline. Nothing new competes for the
GPU and the conversation path is untouched.

⚑ DETECTION IS FREE, ENRICHMENT IS THROTTLED. `teaching.detect_vocab_offer`
finds words the teacher explicitly offered using plain regex -- no model call.
Only when something was offered (or every VOCAB_PROBE_EVERY_TURNS turns, so the
panel is not empty on a smooth conversation) is one small constrained request
spent to fill in meaning, example and a level estimate.

⚑ CONSTRAINED DECODING, NOT REPAIR. The same json_schema response_format that
app/evaluate.py verified against this pinned llama.cpp build. A card that does
not parse is dropped, never patched -- a wrong vocabulary card teaches a wrong
word, which is the most damaging error class this product has (config.py
KNOWN_RISKS).
"""
from __future__ import annotations

import json
import re
import unicodedata

from . import config

SYSTEM = ("Du bist eine erfahrene Deutschlehrerin. Du antwortest AUSSCHLIESSLICH "
          "mit gültigem JSON, ohne Text davor oder danach, ohne Markdown.")

CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "word": {"type": "string", "maxLength": 40},
        "meaning": {"type": "string", "maxLength": 90},
        "example": {"type": "string", "maxLength": 120},
        "cefr": {"type": "string", "enum": ["A1", "A2", "B1", "B2", "C1"]},
        "useful": {"type": "boolean"},
    },
    "required": ["word", "meaning", "example", "cefr", "useful"],
    "additionalProperties": False,
}

PROMPT = """Aus dieser Unterrichtsstunde (Niveau {level}).

Der Schüler sagte: "{learner}"
Die Lehrerin antwortete: "{teacher}"

{focus}

Wähle GENAU EIN deutsches Wort oder EINE Wendung aus der Lehrer-Antwort, die
sich für den Schüler zu merken lohnt. Bedingungen:
- nichts, was er offensichtlich schon kann (nicht: gut, machen, Haus, sagen)
- kein Eigenname, keine Zahl, kein Füllwort
- passend zum Niveau {level} oder eine Stufe darüber
- die Bedeutung erklärst du auf EINFACHEM Deutsch, nicht auf Englisch
- das Beispiel ist ein kurzer, natürlicher Satz mit genau diesem Wort

Wenn nichts wirklich lohnenswert ist, setze "useful" auf false.

Antworte exakt so:
{{"word":"<Wort>","meaning":"<max 10 Wörter, einfaches Deutsch>",\
"example":"<ein kurzer Satz>","cefr":"<A1|A2|B1|B2|C1>","useful":true}}"""

# Never worth a card. Function words and the words the learner already leans on.
TRIVIAL = {
    "gut", "schön", "machen", "sagen", "haben", "sein", "gehen", "kommen",
    "sehen", "geben", "wissen", "können", "wollen", "müssen", "sollen",
    "haus", "leute", "sache", "ding", "problem", "frage", "antwort", "wort",
    "deutsch", "heute", "morgen", "gestern", "jetzt", "immer", "viel", "sehr",
}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip().lower())


def _extract_json(raw: str) -> dict | None:
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


def accept(obj: dict, teacher_text: str, known: set[str]) -> dict | None:
    """Filter hard. A card is a promise that this word is real and useful."""
    if not isinstance(obj, dict) or not obj.get("useful"):
        return None
    word = (obj.get("word") or "").strip()
    meaning = (obj.get("meaning") or "").strip()
    example = (obj.get("example") or "").strip()
    cefr = (obj.get("cefr") or "").strip().upper()
    if not word or not meaning or not example:
        return None
    if len(word) > 40 or len(word.split()) > 4:
        return None
    if _norm(word) in TRIVIAL or _norm(word) in known:
        return None
    if cefr not in ("A1", "A2", "B1", "B2", "C1"):
        return None
    # ⚑ the word must actually come from what the teacher SAID. A card for a
    # word the learner never heard is not vocabulary from the lesson, it is the
    # model inventing homework.
    stem = _norm(word).split()[0][:6]
    if stem and stem not in _norm(teacher_text):
        return None
    # a card whose example does not contain the word teaches nothing
    if stem and stem not in _norm(example):
        return None
    return {"word": word, "meaning": meaning, "example": example,
            "cefr": cefr, "source": "lesson"}


def build_prompt(level: str, learner_text: str, teacher_text: str,
                 offered: list[str] | None) -> str:
    focus = ("Die Lehrerin hat dem Schüler dieses Wort angeboten: "
             + ", ".join(offered) + ". Nimm dieses Wort."
             if offered else
             "Die Lehrerin hat kein Wort ausdrücklich angeboten. "
             "Wähle selbst das nützlichste.")
    return PROMPT.format(level=level, learner=(learner_text or "")[:400],
                         teacher=(teacher_text or "")[:600], focus=focus)


def analyze(llm, level: str, learner_text: str, teacher_text: str,
            offered: list[str] | None, known: set[str]) -> dict | None:
    """One bounded, constrained request. Returns a card or None. Never raises."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(level, learner_text,
                                                     teacher_text, offered)}]
    kw = {"response_format": {"type": "json_schema",
                              "json_schema": {"name": "vocab_card",
                                              "schema": CARD_SCHEMA,
                                              "strict": True}}}
    raw = ""
    try:
        for delta, _meta in llm.stream_chat(msgs, max_tokens=config.VOCAB_MAX_TOKENS,
                                            temperature=0.2, top_p=0.9, **kw):
            raw += delta or ""
    except Exception:
        return None
    obj = _extract_json(raw)
    if obj is None:
        return None
    return accept(obj, teacher_text, known)

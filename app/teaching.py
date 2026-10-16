"""The coaching policy: who he is, and what he should do NEXT.

⚑ THE SURFACE AND THE INTELLIGENCE ARE DELIBERATELY SEPARATED. On the outside
this is a normal conversation with a 29-year-old from Leipzig. Underneath it is
the same adaptive coaching engine: it expands short answers, recasts errors,
returns to recurring weaknesses, raises difficulty gradually and chains topics on
purpose. The learner should think "there's the guy I practise German with" and
only later notice "he keeps making me speak more".

Everything that would break that illusion is forbidden in the prompt itself:
level names, lesson announcements, grammar lectures, praise for "good practice",
and any sentence that describes what the conversation is FOR.

    app/coach.json     WHO he is            persona, principles, topics, chains
    app/teaching.py    WHAT he does next    this file: prompt + per-turn director
    app/learner.py     WHO he is talking to counted learner state

⚑ HOW THE PER-TURN DIRECTIVE REACHES THE MODEL, AND WHY THAT WAY.
`directive()` returns a short German instruction. engine inserts it as a system
message IMMEDIATELY BEFORE the learner's last user message, never at the end and
never by rewriting messages[0]:

  * llama.cpp's Gemma chat template folds a system message into the FOLLOWING
    user turn. A trailing system message has no following user turn and is
    silently dropped -- the directive would do nothing at all.
  * Rewriting messages[0] each turn changes the prompt prefix, which throws away
    llama-server's prefix cache and re-processes the whole conversation every
    turn. That is a direct first-token-latency regression.

The directive is NEVER appended to `Engine.messages`, so it costs nothing in
later turns and cannot accumulate. Every directive is also phrased as something
to DO, never as something to announce -- a directive that leaks into the reply
as "jetzt üben wir..." is a bug, not a style question.

⚑ NO EXTRA MODEL CALL. Every decision here is made from counted state (turn
count, answer length, error types, topics used). The director never asks the LLM
what to do next -- that would put a second request in front of the learner on a
server running -np 1.
"""
from __future__ import annotations

import json
import random
import re

from . import config
from .learner import SHORT_TURN_WORDS, words_of

_COACH: dict | None = None

# Words a learner leans on when their vocabulary is thinner than their ideas.
# Presence of one of these is what triggers the vocabulary nudge; the better
# word is chosen by the coach in context, never from a table.
OVERUSED = ("gut", "schön", "schlecht", "machen", "gemacht", "sache", "sachen",
            "dinge", "ding", "nett", "toll", "viel", "sehr", "interessant",
            "problem", "leute", "normal")

# Ladder for the Speaking Expander. Each rung asks for strictly more language
# than the one before, so a short answer is pushed forward instead of accepted.
# ⚑ Phrased as a conversational move, not as an exercise: "verlang zwei Gründe"
# becomes "Warum? Nenn mir zwei Gründe", never "jetzt üben wir Begründungen".
EXPANDER = [
    "Er hat sehr kurz geantwortet. Lass das nicht durchgehen: frag interessiert "
    "nach und verlang zwei Gründe. Eine Frage, sonst nichts.",
    "Er antwortet immer noch knapp. Lass ihn jetzt einen Nachteil oder ein "
    "Gegenargument zu seiner eigenen Antwort nennen -- als echte Neugier, nicht "
    "als Aufgabe.",
    "Immer noch kurz. Stell ihm einen Vergleich: er soll seine Antwort gegen "
    "eine Alternative abwägen und sich entscheiden.",
    "Er bleibt kurz. Bitte ihn, dasselbe nochmal zu erzählen, aber ausführlich, "
    "in mehreren zusammenhängenden Sätzen.",
]

# What he does with a turn when nothing more urgent applies.
DEEPEN = [
    "Frag nach dem Warum hinter dem, was er gerade gesagt hat.",
    "Lass ihn ein konkretes Beispiel aus seinem eigenen Leben erzählen.",
    "Widersprich ihm freundlich mit deiner eigenen Meinung und lass ihn "
    "dagegenhalten.",
    "Frag nach einem Detail, das er ausgelassen hat.",
    "Lass ihn erklären, wie er zu dieser Entscheidung gekommen ist.",
    "Frag, was er heute anders machen würde.",
    "Erzähl kurz etwas Eigenes dazu und gib die Frage dann an ihn zurück.",
]

# Appended to every directive. The single most important line in this file:
# it is what keeps the coaching invisible.
QUIET = (" Sag nicht, dass du etwas übst, korrigierst oder erklärst, und "
         "kommentiere sein Deutsch nicht.")


def load() -> dict:
    global _COACH
    if _COACH is None:
        _COACH = json.loads(
            (config.APP_DIR / config.PERSONA_FILE).read_text(encoding="utf-8"))
    return _COACH


def name() -> str:
    return load()["identity"]["name"]


# ──────────────────────────────────────────────────────── Topic Generator
def _chain_next(topic_id: str | None, used: set, pool_ids: set) -> str | None:
    """The next station on a chain that STARTS from the current topic.

    ⚑ This is what makes topic changes feel like a conversation instead of a
    worksheet. The chains in coach.json are real associative paths --
    online shopping -> a bad delivery -> calling support -> working with
    customers -> job interviews -> working in Germany. Following one means the
    bridge sentence the coach has to invent is short and obvious, which is
    exactly when a transition sounds natural.
    """
    if not topic_id:
        return None
    for chain in load().get("chains") or []:
        if topic_id in chain:
            i = chain.index(topic_id)
            for nxt in chain[i + 1:]:
                if nxt not in used and nxt in pool_ids:
                    return nxt
    return None


def pick_topic(state: dict, rng: random.Random | None = None) -> dict | None:
    """Choose what to talk about next.

    Repetition avoidance is the whole point: a topic used in this session, or in
    the recent sessions recorded in the profile, is only reachable once nothing
    fresh is left. Level filtering keeps a beginner out of the debates without
    ever telling him that is why.
    """
    rng = rng or random
    t = load()
    level = state.get("level") or config.DEFAULT_LEVEL
    used_now = set(state.get("topics_this_session") or [])
    used_before = set(state.get("recent_topics") or [])
    hard_ok = (state.get("difficulty") or 0) >= 0.55

    pool = [x for x in t["topics"] if level in x["levels"]]
    if not hard_ok:
        pool = [x for x in pool if x["kind"] not in ("debatte",)] or pool
    by_id = {x["id"]: x for x in pool}

    # 1 · stay on the chain if there is one
    nxt = _chain_next(state.get("current_topic"), used_now | used_before,
                      set(by_id))
    if nxt:
        return by_id[nxt]

    # 2 · otherwise something genuinely fresh
    fresh = [x for x in pool
             if x["id"] not in used_now and x["id"] not in used_before]
    candidates = fresh or [x for x in pool if x["id"] not in used_now] or pool
    return rng.choice(candidates) if candidates else None


def pick_roleplay(state: dict, rng: random.Random | None = None) -> dict | None:
    rng = rng or random
    t = load()
    level = state.get("level") or config.DEFAULT_LEVEL
    used = set(state.get("topics_this_session") or [])
    pool = [x for x in t["topics"]
            if x["kind"] == "rollenspiel" and level in x["levels"]
            and x["id"] not in used]
    return rng.choice(pool) if pool else None


# ─────────────────────────────────────────────────────────── system prompt
def build_system_prompt(level: str, state: dict) -> str:
    """Persona + hidden coaching policy + what he knows about the learner.

    ⚑ Kept stable within a turn-to-turn window on purpose: this string is
    `messages[0]`, i.e. the prompt prefix llama-server caches. The engine
    rebuilds it only on a level change or every TEACHER_PROMPT_REFRESH_TURNS
    turns (see Engine._maybe_refresh_prompt), never every turn.

    ⚑ The LEVEL NAME NEVER APPEARS IN THE PROMPT TEXT. It selects the length and
    complexity rules below, but writing "Niveau B1" into the prompt is how a
    model ends up saying "das war gut für B1" out loud.
    """
    t = load()
    lv = config.LEVELS.get(level) or config.LEVELS[config.DEFAULT_LEVEL]
    i, v = t["identity"], t["voice_notes"]
    cp = t["correction_policy"]

    L = [
        f"Du bist {i['name']}, {i['age']}, aus {i['city']}. Du arbeitest als {i['job']}.",
        i["relationship"],
        i["hidden_role"],
        "",
        "WORAUF ES DIR ANKOMMT:",
    ]
    L += [f"- {p}" for p in t["principles"]]

    L += [
        "",
        "WIE DU SPRICHST:",
        f"- {v['register']}",
        f"- Kurze Beiträge: normalerweise {lv['target_words']} Wörter. "
        f"Niemals länger als {lv['max_words']}.",
        f"- {lv['complexity']}",
        "- Fast jeder deiner Beiträge endet mit einer Frage oder einer kleinen "
        "Aufgabe. Er soll reden, nicht du.",
        # ⚑ E11 finding, kept through both persona changes because it is about
        # listening, not about who is talking: across 6 real sessions the partner
        # used NONE of the learner's own content words in 55 of 92 turns.
        "- Greif immer etwas Konkretes auf, das er gerade gesagt hat: ein Wort, "
        "einen Namen, ein Detail. Deine Antwort ist an das anknüpfend, was er "
        "erzählt hat -- nie ein neues Thema aus dem Nichts.",
        "- Themenwechsel gehen über eine Assoziation aus seinem letzten Satz. "
        "Du kündigst einen Themenwechsel nie an.",
        "- Wenn er ein Wort auf Englisch sagt, sagst du das deutsche Wort "
        "beiläufig in deiner Antwort und redest weiter.",
        f"- {lv['help_when_stuck']}",
        "- Keine Emojis, kein Markdown, keine Aufzählungen, keine "
        "Regieanweisungen, keine Überschriften.",
        f"- Vermeide: {', '.join(v['avoid'])}.",
        "",
        "WENN ER EINEN FEHLER MACHT:",
        f"- {cp['recast']}",
        f"- {cp['critical']}",
        f"- {cp['review']}",
        f"- {cp['never']}",
        "",
        "DEINE MEINUNGEN (die hast du wirklich, benutz sie als Reibungsfläche):",
    ]
    L += [f"- {o}" for o in t["opinions"]]

    # ── what he has picked up about the learner. Facts only, no assessment. ──
    facts = []
    if state.get("sessions_before"):
        facts.append(f"- Ihr habt euch schon {state['sessions_before']} Mal "
                     "getroffen.")
    rec = state.get("recurring_errors") or []
    if rec:
        facts.append("- Diese Fehler macht er immer wieder: "
                     + ", ".join(k for k, _n in rec)
                     + ". Bring Situationen ins Gespräch, in denen genau das "
                       "vorkommt -- ohne zu sagen, dass du das machst.")
    used = state.get("structures_used") or []
    if used:
        facts.append("- Diese Strukturen benutzt er schon selbst: "
                     + ", ".join(used) + ".")
    recent = (state.get("recent_topics") or [])[-6:]
    if recent:
        facts.append("- Darüber habt ihr kürzlich schon geredet, nicht "
                     "wiederholen: " + ", ".join(recent) + ".")
    kn = state.get("session_vocab") or []
    if kn:
        facts.append("- Neue Wörter von heute: " + ", ".join(kn)
                     + ". Bring sie später nochmal ins Gespräch.")
    goals = state.get("goals") or []
    if goals:
        facts.append("- Das will er können: " + "; ".join(goals) + ".")
    if facts:
        L += ["", "WAS DU ÜBER IHN WEISST:"] + facts

    L += ["", "SO KLINGST DU:"]
    for e in t["exemplars"]:
        L += [f"  Er: {e['user']}", f"  Du: {e['coach']}"]
    return "\n".join(L)


# ────────────────────────────────────────────────────── Conversation Director
def opener(state: dict, rng: random.Random | None = None) -> str:
    """First thing the learner hears. Deterministic and instant on purpose --
    a generated opener would add an LLM round trip to startup."""
    rng = rng or random
    t = load()
    if state.get("sessions_before"):
        last = state.get("recent_topics") or []
        if last:
            dom = last[-1].split("_")[0]
            return rng.choice(t["returning_openers"]).replace("{last_topic}", dom)
        return t["returning_openers"][-1]
    return rng.choice(t["openers"])


def _review_block(errors: list[dict], limit: int) -> str:
    """Which mistakes are worth breaking the flow for.

    Priority is by how often the pattern came back, then by how much the error
    class costs a speaker: word order and verb forms break sentences, articles
    and cases usually do not.
    """
    weight = {"word_order": 5, "verb": 5, "auxiliary": 4, "tense": 4,
              "case": 3, "preposition": 3, "vocabulary": 3, "gender": 2,
              "article": 2, "expression": 1}
    ranked = sorted(errors,
                    key=lambda m: (int(m.get("count") or 1),
                                   weight.get(m.get("type"), 1)),
                    reverse=True)
    out = []
    for m in ranked[:limit]:
        o, b = (m.get("original") or "").strip(), (m.get("better") or "").strip()
        if o and b:
            out.append(f'"{o}" statt "{b}"')
    return "; ".join(out)


def directive(model, mistakes: list[dict] | None = None,
              rng: random.Random | None = None) -> tuple[str | None, dict]:
    """What he should DO with this turn. Returns (text, meta).

    One decision per turn, highest priority first. Returning None is allowed and
    normal -- not every turn needs steering, and steering every single turn is
    what would make him feel mechanical.
    """
    rng = rng or random
    meta: dict = {"kind": None}
    if not config.TEACHER_DIRECTOR_ENABLED:
        return None, meta

    last = model.last_text or ""
    lw = model.last_words

    # 1 · A RECURRING MISTAKE IS WORTH ONE SENTENCE. Not a lesson, not a list.
    if (model.phase == "review" and mistakes
            and model.turns_since_review >= config.TEACHER_REVIEW_EVERY_TURNS):
        block = _review_block(mistakes, config.TEACHER_REVIEW_MAX_ITEMS)
        if block:
            model.mark_review_done()
            meta["kind"] = "review"
            return ("Diese Sache kommt bei ihm immer wieder: " + block + ". "
                    "Sag beiläufig, wie es richtig heißt, mit einem kurzen "
                    "Beispiel -- ein, zwei Sätze im Plauderton, keine "
                    "Grammatikregel -- und frag danach sofort weiter zum "
                    "Gespräch." + QUIET), meta

    # 2 · SPEAKING EXPANDER. The single most important move in this product.
    if lw and lw <= SHORT_TURN_WORDS and model.turns > 1:
        rung = min(model.short_streak - 1, len(EXPANDER) - 1)
        meta["kind"] = "expand"
        meta["rung"] = max(0, rung)
        return EXPANDER[max(0, rung)] + QUIET, meta

    # 3 · SOCRATIC. He made a mistake of a type he keeps making -- let him find
    #     the correct form himself instead of handing it to him.
    if mistakes and config.TEACHER_SOCRATIC_ENABLED:
        recent = [m for m in mistakes
                  if model.turns in (m.get("turns") or [])
                  and int(m.get("count") or 1) >= 2]
        if recent:
            m = recent[0]
            meta["kind"] = "socratic"
            meta["type"] = m.get("type")
            return ('Er hat wieder dieselbe Stelle vergeigt ("%s"). Gib ihm '
                    "NICHT die Lösung. Stell eine kurze, beiläufige Frage, die "
                    "ihn selbst draufbringt, und lass ihn den Satz nochmal "
                    "sagen." % (m.get("original") or "") + QUIET), meta

    # 4 · NEXT SITUATION. The topic is used up, or his answers on it got thin.
    if (model.turns_on_topic >= config.TEACHER_TOPIC_MAX_TURNS
            or (model.short_streak >= 2 and model.turns_on_topic >= 3)):
        state = model.prompt_state()
        state["current_topic"] = model.topic_id
        topic = pick_topic(state, rng)
        if topic:
            model.start_topic(topic["id"], topic["domain"])
            meta["kind"] = "topic"
            meta["topic"] = topic["id"]
            return ("Das Thema ist durch. Häng dich an ein Wort aus seinem "
                    "letzten Satz und komm darüber ganz beiläufig hierauf, in "
                    "deinen eigenen Worten: " + topic["seed"] + QUIET), meta

    # 5 · CHALLENGE. Something harder, on the beat -- as a game, never as a task.
    if model.phase == "challenge":
        rp = (pick_roleplay(model.prompt_state(), rng)
              if rng.random() < config.TEACHER_ROLEPLAY_CHANCE else None)
        if rp:
            model.start_topic(rp["id"], rp["domain"])
            meta["kind"] = "roleplay"
            meta["topic"] = rp["id"]
            return ("Schlag ihm spontan vor, eine Situation durchzuspielen. "
                    "Erklär in EINEM Satz, wer du bist, dann spiel deine Rolle "
                    "sofort und bleib drin: " + rp["seed"] + QUIET), meta
        meta["kind"] = "challenge"
        return ("Mach es etwas schwerer: verlang eine Begründung über mehrere "
                "Sätze, oder lass ihn zwei Möglichkeiten gegeneinander "
                "abwägen." + QUIET), meta

    # 6 · VOCABULARY. He is leaning on a basic word -- give him a better one
    #     inside the conversation and make him use it immediately.
    if (config.VOCAB_ENABLED
            and model.turns_since_vocab >= config.VOCAB_SUGGEST_EVERY_TURNS
            and any(w in OVERUSED for w in words_of(last))):
        model.mark_vocab_done()
        meta["kind"] = "vocab"
        return ("Er benutzt ein sehr einfaches Wort. Wirf beiläufig EIN "
                "passenderes deutsches Wort oder eine natürlichere Wendung ein, "
                "in einem Halbsatz, und lass ihn den Satz damit nochmal sagen. "
                "Keine Wortliste, kein Unterricht." + QUIET), meta

    # 7 · DEEPEN. Default: one level down, not sideways.
    if model.turns >= 1 and rng.random() < config.TEACHER_DEEPEN_CHANCE:
        meta["kind"] = "deepen"
        parts = [rng.choice(DEEPEN)]
        rep = _repeated_opening(model.teacher_openings)
        if rep:
            parts.append(f'Fang deinen Beitrag nicht wieder mit "{rep}" an.')
        return " ".join(parts) + QUIET, meta

    return None, meta


def _repeated_opening(openings: list[str]) -> str | None:
    if len(openings) < 3:
        return None
    last3 = openings[-3:]
    if len(set(last3)) == 1:
        return last3[0]
    if openings[-1] == openings[-2]:
        return openings[-1]
    return None


# ───────────────────────────────────────────── Vocabulary offer detection
# He is told to drop better words into his own sentences. When he does, this
# finds the offer so the UI can turn it into a card -- deterministic, no model
# call, no latency. The LLM is only asked to fill in meaning and an example
# later, in the idle window (see app/vocab.py).
_OFFER = re.compile(
    r"(?:passenderes wort|besseres wort|natürlicher(?:es)?|man sagt (?:eher|hier)|"
    r"du kannst auch sagen|besser (?:klingt|wäre)|sag lieber|dafür sagt man|"
    r"ich würde eher|auf deutsch sagt man)\b[^.!?]{0,60}",
    re.IGNORECASE)
# Words that appear inside an offer phrase but are never the offered word.
_OFFER_NOISE = {
    "wort", "wörter", "wendung", "ausdruck", "auch", "eher", "hier", "lieber",
    "besser", "natürlicher", "passender", "passenderes", "deutsch", "dafür",
    "statt", "dass", "denn", "einfach", "vielleicht", "wäre", "sagen", "sagt",
    "mal", "nochmal", "nicht", "aber", "sein", "ist", "wird", "kannst", "kann",
    "würde", "sagst",
}
_QUOTED = re.compile(r"[\"„»']([A-Za-zÄÖÜäöüß][\wÄÖÜäöüß\- ]{2,30})[\"“«']")


def detect_vocab_offer(text: str) -> list[str]:
    """Words he explicitly offered in this reply. Never more than two."""
    if not text:
        return []
    out: list[str] = []
    for m in _OFFER.finditer(text):
        seg = m.group(0)
        q = _QUOTED.findall(seg)
        if q:
            out += q
        else:
            # ⚑ NOT capitalisation-based. German capitalises nouns, so a
            # capital-letter rule silently loses every adjective and verb he
            # offers ("anspruchsvoll", "sich bewerben") -- exactly the words a
            # learner at this stage is missing.
            tail = re.split(r":|\bwäre\b|\bsagt man\b|\bsagen\b|\bklingt\b",
                            seg)[-1]
            for w in re.findall(r"[A-Za-zÄÖÜäöüß][\wäöüß\-]{3,}", tail):
                if w.lower() not in _OFFER_NOISE:
                    out.append(w)
                    break
    if not out:
        # Only as a fallback. Scanning every quoted span of the whole reply picks
        # up the learner's OWN basic word ("Für 'gut' gibt es ..."), which is the
        # opposite of a vocabulary card, so it may never override a real offer.
        out += _QUOTED.findall(text)[:1]
    seen, res = set(), []
    for w in out:
        k = w.strip().lower()
        if k and k not in seen:
            seen.add(k)
            res.append(w.strip())
    return res[:2]

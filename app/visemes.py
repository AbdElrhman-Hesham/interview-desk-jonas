"""A coarse mouth-shape timeline for one spoken chunk. No audio, no model.

⚑ WHY IT IS DERIVED FROM TEXT AND NOT FROM AUDIO. Every sample of speech in this
product is played by Python (`app/audio_out.SpeechPlayer`); the browser is a
view and performs no audio at all. That is a load-bearing property -- it is what
lets the desktop shell host the UI in a WebView with no media permissions -- so
"analyse the waveform in the browser" is not available and must not be added.

What IS available at the moment a chunk is handed to the speaker: the exact text
Piper was given, and the exact number of samples it produced. That is enough for
a believable mouth: German orthography is close enough to its phonology that a
grapheme-to-viseme map produces the right OPEN/WIDE/ROUND/CLOSED sequence, and
the true duration means the sequence stays in sync with the audio even though
nothing measures it.

⚑ IT IS AN APPROXIMATION ON PURPOSE. The goal is "the person in front of me is
talking", not phoneme accuracy. Cost is a few microseconds per chunk, zero GPU,
and it degrades to a neutral mouth if anything here fails.

Viseme codes (kept short -- they cross the SSE wire on every chunk):
    X  closed / rest        M B P, silence
    A  open                 a, ä
    E  wide                 e, i, ie
    O  rounded              o, u, au, eu, ö, ü
    F  teeth on lip         f, v, w, pf
    S  narrow, tense        s, sch, ch, z, tsch, j
    T  tongue, small open   t d n l r g k h
"""
from __future__ import annotations

import re

from . import config

# Order matters: longest digraphs first, or "sch" is read as s + c + h.
_RULES: list[tuple[str, str]] = [
    ("sch", "S"), ("tsch", "S"), ("chs", "S"), ("ch", "S"), ("sp", "S"),
    ("st", "S"), ("pf", "F"), ("ph", "F"), ("qu", "O"),
    ("au", "O"), ("eu", "O"), ("äu", "O"), ("ei", "E"), ("ai", "E"),
    ("ie", "E"), ("ee", "E"), ("aa", "A"), ("oo", "O"),
    ("a", "A"), ("ä", "A"),
    ("e", "E"), ("i", "E"), ("y", "E"),
    ("o", "O"), ("u", "O"), ("ö", "O"), ("ü", "O"),
    ("m", "X"), ("b", "X"), ("p", "X"),
    ("f", "F"), ("v", "F"), ("w", "F"),
    ("s", "S"), ("z", "S"), ("ß", "S"), ("j", "S"),
    ("t", "T"), ("d", "T"), ("n", "T"), ("l", "T"), ("r", "T"),
    ("g", "T"), ("k", "T"), ("h", "T"), ("c", "T"), ("x", "S"),
]
_MAXLEN = max(len(k) for k, _ in _RULES)

# Vowels carry the visible mouth movement; consonants between them are passed
# over quickly. These weights turn a flat character sequence into something with
# the rhythm of speech.
_WEIGHT = {"A": 2.2, "O": 2.0, "E": 1.8, "S": 1.1, "F": 1.0, "T": 0.9, "X": 0.8}
_PUNCT_PAUSE = {".": 2.4, "!": 2.4, "?": 2.4, ",": 1.2, ";": 1.4, ":": 1.4,
                "-": 0.8, "…": 2.4}


def units(text: str) -> list[tuple[str, float]]:
    """(viseme, relative_weight) for one chunk of German text."""
    out: list[tuple[str, float]] = []
    s = (text or "").lower()
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            out.append(("X", 0.7))
            i += 1
            continue
        if c in _PUNCT_PAUSE:
            out.append(("X", _PUNCT_PAUSE[c]))
            i += 1
            continue
        hit = None
        for L in range(_MAXLEN, 0, -1):
            frag = s[i:i + L]
            if len(frag) < L:
                continue
            for k, v in _RULES:
                if k == frag:
                    hit = (v, L)
                    break
            if hit:
                break
        if hit is None:
            i += 1
            continue
        code, L = hit
        # a doubled consonant is one closure, not two
        if out and out[-1][0] == code and code in ("X", "T", "S", "F"):
            out[-1] = (code, out[-1][1] + 0.3)
        else:
            out.append((code, _WEIGHT.get(code, 1.0)))
        i += L
    return out


def timeline(text: str, duration_ms: float) -> list[list]:
    """[[start_ms, code], ...] covering `duration_ms`, ending closed.

    Shapes shorter than VISEME_MIN_MS are merged away: below ~50 ms a mouth
    reads as jitter rather than articulation, and it costs frames for nothing.
    """
    if not config.VISEMES_ENABLED or duration_ms <= 0:
        return []
    us = units(text)
    if not us:
        return []
    total_w = sum(w for _c, w in us) or 1.0
    ms_per_w = duration_ms / total_w

    out: list[list] = []
    t = 0.0
    for code, w in us:
        dur = w * ms_per_w
        if out and dur < config.VISEME_MIN_MS and out[-1][1] == code:
            t += dur
            continue
        if out and dur < config.VISEME_MIN_MS:
            # too short to see: let the previous shape hold through it
            t += dur
            continue
        out.append([round(t, 1), code])
        t += dur
    if not out:
        out = [[0.0, "A"]]
    out.append([round(min(t, duration_ms), 1), "X"])       # close the mouth
    return out

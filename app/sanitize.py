"""Deterministic TTS safety sanitizer. FROZEN BEFORE B2. Disposable lab code.

⚑ PARITY RULE: this exact function is applied to EVERY candidate's output,
identically, with no per-model branches. It exists so that TTS does not crash or
read markup aloud -- NOT to hide any model's problems.

⚑ EVERYTHING IT REMOVES IS COUNTED AND REPORTED as a model-output compatibility
failure against the candidate that produced it. Sanitizing is not forgiving:
emitting emoji, markdown or stage directions is a real product defect (the
system prompt explicitly forbids all of them), and the count is evidence.

The raw, unsanitized text is always persisted alongside the sanitized text, so
nothing is silently lost.
"""
from __future__ import annotations

import re

EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F900-\U0001F9FF☀-➿️♀♂‍]+")
# A SINGLE-asterisk span is a stage direction and is DELETED (*lacht*).
# A double/triple-asterisk span is emphasis and is UNWRAPPED (**sehr** -> sehr).
# The lookarounds stop this from matching the inner *sehr* of **sehr**, which
# would leave orphaned asterisks behind.
STAGE_DIRECTION = re.compile(r"(?<!\*)\*(?!\*)[^*\n]{1,80}(?<!\*)\*(?!\*)")
UNDERSCORE_EMPH = re.compile(r"_([^_\n]{1,80})_")
MD_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
MD_BULLET = re.compile(r"(?m)^\s{0,3}[-*•]\s+")
MD_NUMBERED = re.compile(r"(?m)^\s{0,3}\d+[.)]\s+")
CODE_FENCE = re.compile(r"```[\s\S]*?```|`([^`\n]*)`")
BRACKETED = re.compile(r"\[[^\]\n]{1,80}\]")               # [Pause], [lacht]
MD_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
BOLD_ITALIC = re.compile(r"\*{2,3}([^*\n]+)\*{2,3}")
MULTISPACE = re.compile(r"[ \t]{2,}")
MULTINEWLINE = re.compile(r"\n{2,}")


def sanitize(text: str) -> tuple[str, dict]:
    """Return (speakable_text, counts_of_what_was_removed).

    Counts are per-category so the report can say exactly which compatibility
    failure each candidate produced, and how often.
    """
    if not text:
        return "", {}
    counts: dict[str, int] = {}

    def sub(pat, repl, key, s):
        n = len(pat.findall(s))
        if n:
            counts[key] = counts.get(key, 0) + n
        return pat.sub(repl, s)

    s = text
    s = sub(MD_LINK, r"\1", "markdown_link", s)
    s = sub(CODE_FENCE, r"\1", "code_span", s)
    # emphasis BEFORE stage directions: otherwise **sehr** is seen as the stage
    # direction *sehr* and deleting it strips the word, leaving stray asterisks
    s = sub(BOLD_ITALIC, r"\1", "markdown_emphasis", s)
    s = sub(STAGE_DIRECTION, "", "stage_direction", s)
    s = sub(BRACKETED, "", "bracketed_aside", s)
    s = sub(UNDERSCORE_EMPH, r"\1", "underscore_emphasis", s)
    s = sub(MD_HEADING, "", "markdown_heading", s)
    s = sub(MD_BULLET, "", "markdown_bullet", s)
    s = sub(MD_NUMBERED, "", "markdown_numbered", s)
    s = sub(EMOJI, "", "emoji", s)

    s = MULTINEWLINE.sub("\n", s)
    s = MULTISPACE.sub(" ", s)
    s = "\n".join(ln.strip() for ln in s.splitlines())
    s = s.strip()

    # If sanitizing emptied the turn, that is itself a total failure -- surface
    # it rather than sending silence to the synthesizer.
    if not s and text.strip():
        counts["emptied_by_sanitizer"] = 1
    return s, counts


def merge_counts(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


if __name__ == "__main__":
    cases = [
        "Ach echt? Mit Freunden? 😊",
        "*lacht* Ja klar. **Sehr** gut!",
        "- Punkt eins\n- Punkt zwei",
        "Schau mal [hier](http://x.de) rein.",
        "[Pause] Also, wie war's?",
        "Ganz normaler Satz ohne alles.",
        "😂😂😂",
    ]
    for c in cases:
        out, n = sanitize(c)
        print(f"{c!r}\n  -> {out!r}   {n}")

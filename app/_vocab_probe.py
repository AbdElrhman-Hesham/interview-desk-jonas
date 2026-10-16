"""Measure the rate of Gemma's wrong-vocabulary error.

The 'Drogerie' for 'pharmacy' error was seen in E1 Stage A (S7 seed 11) and
again in MVP smoke run 1, but NOT in run 2 -- so it is stochastic. Anecdote is
not evidence of a rate. This runs the same word-finding prompts N times at the
production sampling settings and counts.

⚑ Wrong vocabulary is the most damaging error class for a language-learning
product: the learner has no way to know they were taught the wrong word.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.providers import LlamaCppProvider  # noqa: E402
from app.engine import Engine  # noqa: E402

# (learner utterance, correct German, wrong answers seen or plausible)
PROBES = [
    ("Ich war gestern in der... pharmacy.", ["apotheke"], ["drogerie"]),
    ("Wie sagt man 'prescription' auf Deutsch?", ["rezept"], []),
    ("Ich brauche ein neues... how do you say 'charger'?", ["ladegerät", "ladekabel"], []),
    ("Ich muss noch zur... post office.", ["post", "postamt"], ["postfiliale"]),
    ("Wo ist die nächste... pharmacy?", ["apotheke"], ["drogerie"]),
]
N = 8


def main():
    llm = LlamaCppProvider(config.LLM_MODEL_PATH, config.LLAMA_BUILD_DIR,
                           config.LLM_PORT, config.LLM_CTX, config.LLM_NGL,
                           config.LLM_SERVER_ARGS, config.REQ_TIMEOUT_S)
    llm.start()
    e = Engine.__new__(Engine)
    e.level = "B1"
    system = Engine._system_prompt(e)

    print("=" * 78)
    print(f"  VOCABULARY ACCURACY PROBE — {config.MODEL_DECISION['llm']}")
    print(f"  {N} samples per probe, production sampling {config.LLM_SAMPLING}")
    print("=" * 78)

    grand = Counter()
    for utt, correct, wrong in PROBES:
        got = Counter()
        samples = []
        for _ in range(N):
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": utt}]
            out = "".join(d for d, _m in llm.stream_chat(msgs, max_tokens=90))
            low = out.lower()
            if any(w in low for w in correct):
                got["correct"] += 1
            elif any(w in low for w in wrong):
                got["WRONG"] += 1
                samples.append(out.strip()[:90])
            else:
                got["no_word_supplied"] += 1
                samples.append(out.strip()[:90])
        grand.update(got)
        rate = got["WRONG"] / N
        mark = "  <-- WRONG WORD" if got["WRONG"] else ""
        print(f"\n  {utt}")
        print(f"    correct={got['correct']}/{N}  wrong={got['WRONG']}/{N}  "
              f"no-word={got['no_word_supplied']}/{N}{mark}")
        for s in samples[:3]:
            print(f"      · {s}")

    tot = sum(grand.values())
    print("\n" + "=" * 78)
    print(f"  TOTAL  correct={grand['correct']}/{tot}  "
          f"WRONG={grand['WRONG']}/{tot} ({100*grand['WRONG']/tot:.0f}%)  "
          f"no-word={grand['no_word_supplied']}/{tot}")
    print("=" * 78)
    llm.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

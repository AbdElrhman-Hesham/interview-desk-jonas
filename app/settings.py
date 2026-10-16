"""Persisted user preferences. Local JSON, nothing more.

Only the handful of settings a real user actually needs between sessions:
German level, microphone, transcript visibility. Model paths, sampling
parameters, GPU layers and compute types are developer configuration and stay
in config.py, deliberately out of the product surface.

Stored next to the session logs in %LOCALAPPDATA%, never in the repo and never
in the OneDrive-synced project folder.
"""
from __future__ import annotations

import json
import os
import tempfile

from . import config

PATH = config.DATA_DIR / "settings.json"

DEFAULTS = {
    "level": config.DEFAULT_LEVEL,
    "input_device": None,        # None = auto-pick (see devices.default_input)
    "show_transcript": False,    # transcript is deliberately secondary
    # ⚑ ON by default -- hands-free is the normal experience, not an option.
    # A settings.json written before this key existed simply inherits True.
    "handsfree": config.HANDSFREE_ENABLED_DEFAULT,
    "show_mistakes": True,       # passive mistakes panel
    # ⚑ V2 panels. Both default ON: in the teacher product the vocabulary and
    # the progress readout ARE part of the lesson, not developer instrumentation.
    "show_vocab": True,          # vocabulary cards from the lesson
    "show_progress": True,       # counted session progress
    "panel": "vokabeln",         # which drawer tab was last open
    "interview_language": "de",  # last choice on the bilingual start screen
}


def load() -> dict:
    s = dict(DEFAULTS)
    try:
        raw = json.loads(PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k in DEFAULTS:
                if k in raw:
                    s[k] = raw[k]
    except FileNotFoundError:
        pass
    except Exception:
        # A corrupt settings file must never stop the app from launching.
        pass
    if s["level"] not in config.LEVELS:
        s["level"] = config.DEFAULT_LEVEL
    if s["interview_language"] not in ("de", "en"):
        s["interview_language"] = "de"
    return s


def save(**changes) -> dict:
    """Merge and persist. Atomic, so a crash mid-write cannot corrupt it."""
    s = load()
    for k, v in changes.items():
        if k in DEFAULTS:
            s[k] = v
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(PATH.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, PATH)
    except Exception:
        pass
    return s

"""First run: turn an installed application into a working one.

The installer ships the application (~82 MiB). Everything large is provisioned
here, once, into `%LOCALAPPDATA%\\Lena`:

    runtime\\site-packages   the engine's Python packages   ~2.3 GiB
    bin\\llama124            llama.cpp + CUDA runtime       ~1.1 GiB
    models\\                 Gemma, Whisper, Piper          ~6.4 GiB

⚑ THIS IS NOT A SECOND DOWNLOADER. Every artifact is fetched by
`lab.setup.bootstrap.fetch()` — the same pinned-URL, SHA-256-verified, resumable
implementation that `setup.cmd` uses and that `_selftest.py` proves against the
real network (fresh download, flipped byte, truncated file, resumed partial,
wrong checksum, dead URL, zip-slip). The manifest is the same file. Only the
DESTINATIONS differ, because an installed application must not write into
Program Files.

The user never runs `setup.cmd`. They install Lena and start it; this runs
behind the splash with named states and, where the number is real, byte counts.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

APP_NAME = "Lena"


def data_root() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME


def site_packages() -> Path:
    return data_root() / "runtime" / "site-packages"


def bin_dir() -> Path:
    return data_root() / "bin"


def models_dir() -> Path:
    return data_root() / "models"


def packages_ready() -> bool:
    """Engine runtime plus the PDF dependencies introduced in version 2."""
    sp = site_packages()
    return all((sp / name).exists() for name in ("ctranslate2", "reportlab", "PIL"))


def is_provisioned() -> bool:
    """Cheap check used on EVERY launch. Deliberately a few file stats."""
    try:
        from app import config
    except Exception:
        return False
    return all([
        packages_ready(),
        (bin_dir() / "llama124" / "llama-server.exe").exists(),
        Path(config.LLM_MODEL_PATH).exists(),
        all(Path(voice).exists() and Path(str(voice) + ".json").exists()
            for voice in config.TTS_VOICES.values()),
    ])


def activate():
    """Put the provisioned engine packages on sys.path.

    ⚑ Called before any engine import. The shell's own dependencies live in the
    application directory; the engine's live in the data directory. One
    interpreter, two package directories — validated: numpy, ctranslate2,
    faster_whisper, onnxruntime, sounddevice, soundfile and piper all import
    from the relocated directory, with CUDA visible.
    """
    sp = str(site_packages())
    if sp not in sys.path and Path(sp).is_dir():
        sys.path.insert(0, sp)


class FirstRun:
    """Drives provisioning and reports named states to a callback."""

    def __init__(self, on_state=None):
        self.on_state = on_state or (lambda s, d="": None)

    def say(self, state: str, detail: str = ""):
        self.on_state(state, detail)

    # ---------------------------------------------------------------- steps
    def run(self) -> tuple[bool, str]:
        """Returns (ok, message). Safe to call again after any failure."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from lab.setup import bootstrap as bs
        except Exception as e:
            return False, f"Die Einrichtung konnte nicht starten ({type(e).__name__})."

        import json
        mf = json.loads((Path(bs.__file__).parent / "manifest.json")
                        .read_text(encoding="utf-8"))
        state = bs.State(data_root() / "setup-state.json")

        class Args:
            reverify = False
            keep_corrupt = False
        args = Args()

        data_root().mkdir(parents=True, exist_ok=True)

        # 1 ── disk, before anything is downloaded ------------------------
        self.say("System wird geprüft…")
        need = mf["disk"]["required_free_bytes"]
        import shutil as _sh
        free = _sh.disk_usage(data_root()).free
        if free < need:
            return False, (f"Zu wenig Speicherplatz.\n"
                           f"Benötigt: {need / 2**30:.0f} GB · "
                           f"verfügbar: {free / 2**30:.0f} GB.\n"
                           f"Schaffe Platz und starte Lena erneut.")

        # 2 ── the engine's Python packages ------------------------------
        if not packages_ready():
            self.say("Laufzeitumgebung wird vorbereitet…",
                     "Einmalig, etwa 2 GB.")
            # Existing Lena installations already contain the heavy speech
            # stack.  Upgrade those with only the two small report packages;
            # a clean installation still uses the complete pinned lock file.
            ok, msg = self._install_packages(
                report_only=(site_packages() / "ctranslate2").is_dir())
            if not ok:
                return False, msg

        # The next download step imports huggingface_hub from the freshly
        # provisioned data site-packages. Activate it now, not after run()
        # returns, otherwise a clean packaged first run reports a false network
        # error even though the packages were installed successfully.
        activate()

        # 3 ── llama.cpp + CUDA runtime ----------------------------------
        for art in [a for a in mf["artifacts"] if a["stage"] == "binaries"]:
            target = bin_dir() / Path(art["extract_to"]).name
            if (target / art["sentinel"]).exists():
                continue
            self.say("Sprachmodell-Programm wird geladen…",
                     f"{art['size'] / 2**20:.0f} MB")
            try:
                archive = bin_dir() / Path(art["archive"]).name
                bs.fetch(art, archive, state, args)
                self.say("Sprachmodell-Programm wird entpackt…")
                target.mkdir(parents=True, exist_ok=True)
                import zipfile
                with zipfile.ZipFile(archive) as z:
                    for m in z.namelist():
                        p = (target / m).resolve()
                        if not str(p).startswith(str(target.resolve())):
                            return False, "Ein heruntergeladenes Archiv war fehlerhaft."
                    z.extractall(target)
            except Exception as e:
                return False, self._net_message(e)

        # 4 ── the models ------------------------------------------------
        labels = {
            "llm_gemma": "Gesprächsmodell wird geladen…",
            "tts_voice": "Deutsche Stimme wird vorbereitet…",
            "tts_voice_config": "Deutsche Stimme wird vorbereitet…",
            "tts_voice_en": "Englische Stimme wird vorbereitet…",
            "tts_voice_en_config": "Englische Stimme wird vorbereitet…",
        }
        for art in [a for a in mf["artifacts"] if a["stage"] == "models"]:
            dest = models_dir() / Path(art["dest"]).name
            if art["id"] in ("tts_voice", "tts_voice_config",
                              "tts_voice_en", "tts_voice_en_config"):
                dest = models_dir() / "piper" / Path(art["dest"]).name
            self.say(labels.get(art["id"], "Dateien werden geladen…"),
                     f"{art['size'] / 2**30:.2f} GB" if art["size"] > 2**30
                     else f"{art['size'] / 2**20:.0f} MB")
            try:
                bs.fetch(art, dest, state, args)
            except Exception as e:
                return False, self._net_message(e)

        # 5 ── Whisper snapshot ------------------------------------------
        for snap in mf.get("hf_snapshots", []):
            self.say("Spracherkennung wird geladen…",
                     f"{snap['approx_bytes'] / 2**30:.2f} GB")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=snap["repo"], revision=snap["revision"])
            except Exception as e:
                return False, self._net_message(e)

        self.say("Dateien werden geprüft…")
        return True, ""

    # ------------------------------------------------------------- helpers
    def _install_packages(self, report_only: bool = False) -> tuple[bool, str]:
        """pip the pinned lock into the data directory, with this interpreter."""
        lock = Path(__file__).resolve().parents[1] / "lab" / "setup" / "requirements.lock"
        target = site_packages()
        target.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        cmd = [sys.executable, "-m", "pip", "install",
               "--disable-pip-version-check", "--no-warn-script-location",
               "--only-binary=:all:", "--target", str(target)]
        if report_only:
            cmd += ["pillow==12.3.0", "reportlab==5.0.1"]
        else:
            cmd += ["-r", str(lock)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as e:
            return False, f"Die Laufzeitumgebung konnte nicht installiert werden ({type(e).__name__})."
        if r.returncode != 0:
            tail = ((r.stderr or "") + (r.stdout or ""))[-400:]
            if "network" in tail.lower() or "resolve" in tail.lower() \
                    or "retries exceeded" in tail.lower():
                return False, ("Keine Internetverbindung.\n"
                               "Lena braucht sie nur für die Ersteinrichtung.\n"
                               "Verbinde dich und starte Lena erneut.")
            return False, ("Die Laufzeitumgebung konnte nicht installiert werden.\n"
                           "Starte Lena erneut — begonnene Downloads werden "
                           "fortgesetzt.")
        return True, ""

    @staticmethod
    def _net_message(e: Exception) -> str:
        t = f"{type(e).__name__}: {e}".lower()
        if "checksum" in t or "mismatch" in t:
            return ("Eine heruntergeladene Datei war beschädigt.\n"
                    "Starte Lena erneut — die Datei wird neu geladen.")
        if "space" in t or "enospc" in t:
            return "Zu wenig Speicherplatz auf der Festplatte."
        return ("Der Download wurde unterbrochen.\n"
                "Prüfe die Internetverbindung und starte Lena erneut — "
                "bereits geladene Dateien bleiben erhalten.")

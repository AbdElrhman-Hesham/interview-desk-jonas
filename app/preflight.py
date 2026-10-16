"""Startup preflight: find the broken prerequisite BEFORE the 20-second boot.

`setup.cmd` provisions a machine. This is the other half: on every normal
`start.cmd`, check in well under a second that the things the conversation needs
are actually there, and if they are not, say so in a sentence the learner can
act on.

⚑ THE POINT IS THE MESSAGE, NOT THE CHECK. Without this, a missing Piper voice
surfaces after llama-server and Whisper have already loaded — fifteen seconds of
waiting for a `FileNotFoundError`; a missing Whisper snapshot silently starts a
1.5 GiB download on what the user thought was a normal launch; a stale device
index throws mid-sentence, when they are already talking. Each of those is
detectable in milliseconds from a file stat or a device query.

SEVERITY
  HARD  the conversation cannot start. Abort with a remedy.
  WARN  it can start, but something is degraded or was silently repaired.
  INFO  worth printing, no judgement.

⚑ Findings carry a German `message` for the product surface and an English
`detail` for the log. Tracebacks, CUDA dumps and port numbers belong in
`detail`, never in `message`.

⚑ This module must stay CHEAP and IMPORT-LIGHT: file stats, a device query and
a socket probe. It must never load a model, and it must never download
anything — provisioning stays explicit, in setup.cmd.
"""
from __future__ import annotations

import os
import socket
import time
from pathlib import Path

from . import config

HARD, WARN, INFO = "HARD", "WARN", "INFO"

SETUP_HINT = "Führe einmal setup.cmd aus."


class Finding:
    __slots__ = ("severity", "code", "message", "detail", "remedy")

    def __init__(self, severity, code, message, detail="", remedy=""):
        self.severity = severity
        self.code = code
        self.message = message      # German, product surface
        self.detail = detail        # English/technical, log only
        self.remedy = remedy        # German, what to do next

    @property
    def user_text(self) -> str:
        return f"{self.message} {self.remedy}".strip()

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code,
                "message": self.message, "detail": self.detail,
                "remedy": self.remedy}

    def __repr__(self):
        return f"<{self.severity} {self.code}>"


# ────────────────────────────────────────────────────────────── the checks
def _artifacts(out: list[Finding]):
    """The four files without which nothing can happen. Stat only."""
    checks = [
        (config.LLM_MODEL_PATH, "llm_model_missing",
         "Das Sprachmodell fehlt.", 1_000_000_000),
        (config.LLAMA_BUILD_DIR / "llama-server.exe", "llama_binary_missing",
         "Das Programm für das Sprachmodell fehlt.", 1000),
    ]
    for language, voice in config.TTS_VOICES.items():
        checks.extend([
            (Path(voice), f"tts_voice_{language}_missing",
             "Eine Interviewstimme fehlt.", 1_000_000),
            (Path(str(voice) + ".json"), f"tts_config_{language}_missing",
             "Die Konfiguration einer Interviewstimme fehlt.", 100),
        ])
    for path, code, msg, min_size in checks:
        try:
            st = path.stat()
        except OSError:
            out.append(Finding(HARD, code, msg, f"missing: {path}", SETUP_HINT))
            continue
        if st.st_size < min_size:
            # A file truncated by a failed copy or an interrupted download is
            # worse than a missing one: it fails deep inside a loader instead.
            out.append(Finding(
                HARD, code + "_truncated",
                msg.replace("fehlt", "ist beschädigt"),
                f"{path} is {st.st_size} bytes, expected at least {min_size}",
                "Führe setup.cmd aus — die Datei wird geprüft und neu geladen."))


def _whisper(out: list[Finding]):
    """⚑ A missing Whisper snapshot must NOT become a silent 1.5 GiB download.

    faster-whisper resolves the `large-v3-turbo` alias through huggingface_hub,
    which downloads on demand. On a normal launch that is a surprise, not a
    feature: the learner pressed start expecting a conversation. Provisioning
    belongs to setup.cmd, so detect the empty cache and say so.
    """
    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    hub = home / "hub" if (home / "hub").exists() else home
    hits = list(hub.glob("models--*faster-whisper-large-v3-turbo*")) if hub.exists() else []
    if not hits:
        out.append(Finding(
            HARD, "stt_model_missing",
            "Die Spracherkennung fehlt.",
            f"no faster-whisper-large-v3-turbo snapshot under {hub}",
            SETUP_HINT))
        return
    # A snapshot directory with no model.bin is an interrupted download.
    for h in hits:
        if any(p.name == "model.bin" for p in (h / "snapshots").rglob("*")) or \
                any(p.name == "model.bin" for p in h.rglob("*")):
            return
    out.append(Finding(
        HARD, "stt_model_incomplete",
        "Die Spracherkennung ist unvollständig.",
        f"snapshot present but no model.bin under {hits[0]}",
        "Führe setup.cmd aus — der Download wird fortgesetzt."))


def _gpu(out: list[Finding]):
    """CUDA is not optional in this architecture; there is no CPU path."""
    try:
        import pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() < 1:
            raise RuntimeError("no CUDA device")
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        free_mb = round(mem.free / 1048576)
        total_mb = round(mem.total / 1048576)
        out.append(Finding(INFO, "gpu", f"Grafikkarte: {name}",
                           f"{name}, {free_mb} MiB free of {total_mb} MiB"))
        # ⚑ WARN, not HARD. The measured peak at HEAD is 4469 MB and another
        # application holding VRAM right now may release it. Refusing to start
        # on an inferred bound would be worse than trying and reporting.
        if free_mb < 5200:
            out.append(Finding(
                WARN, "vram_low",
                "Wenig freier Grafikspeicher — andere Programme belegen ihn.",
                f"{free_mb} MiB free, ~4500 MiB needed at peak",
                "Schließe speicherhungrige Programme, falls der Start scheitert."))
    except Exception as e:
        out.append(Finding(
            HARD, "gpu_missing",
            "Keine nutzbare NVIDIA-Grafikkarte gefunden.",
            f"{type(e).__name__}: {e}",
            "Lena braucht eine NVIDIA-GPU mit aktuellem Treiber."))


def _ports(out: list[Finding]):
    """Distinguish 'Lena is already running' from 'something else is here'."""
    for port, code, what in ((config.SERVER_PORT, "ui_port_busy", "Fenster"),
                             (config.LLM_PORT, "llm_port_busy", "Sprachmodell")):
        if _port_free(port):
            continue
        owner = _is_our_process(port)
        if port == config.SERVER_PORT:
            # ⚑ Only claim "Lena is already running" when the process could
            # plausibly BE Lena. Telling the user to switch to a Lena window
            # that does not exist is worse than admitting we do not know.
            looks_like_lena = bool(owner) and "python" in owner.lower()
            out.append(Finding(
                HARD, code,
                "Lena läuft bereits." if looks_like_lena else
                "Der benötigte Anschluss ist von einem anderen Programm belegt.",
                f"port {port} occupied by {owner or 'an unknown process'}",
                "Wechsle zum bereits geöffneten Lena-Fenster."
                if looks_like_lena else
                f"Beende das Programm „{owner}“ und starte Lena neu."
                if owner else "Beende das andere Programm und starte Lena neu."))
        else:
            # llama-server on 8790 without our UI on 8791 is a leftover from a
            # crashed run. LlamaCppProvider refuses to start on a busy port, so
            # name it now rather than 20 seconds into the boot.
            out.append(Finding(
                HARD, code,
                "Ein Rest einer früheren Sitzung läuft noch."
                if "llama-server" in (owner or "") else
                "Der benötigte Anschluss ist belegt.",
                f"port {port} occupied by {owner or 'an unknown process'}",
                "Beende llama-server.exe im Task-Manager und starte Lena neu."
                if "llama-server" in (owner or "") else
                "Beende das Programm auf diesem Anschluss und starte Lena neu."))


def _port_free(port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.4)
    try:
        s.connect(("127.0.0.1", port))
        return False
    except Exception:
        return True
    finally:
        s.close()


def _is_our_process(port: int) -> str | None:
    """Name of the process listening on `port`, best effort.

    ⚑ Identification only. Startup never kills anything: the runtime already
    owns llama-server's lifecycle, and a second process manager competing with
    it -- or worse, killing an unrelated program to free a port -- is not a
    trade this product should make.
    """
    try:
        import psutil
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN \
                    and c.pid:
                return psutil.Process(c.pid).name()
    except Exception:
        pass
    return None


def _storage(out: list[Finding]):
    """Session logs and recordings must be writable, or turns vanish silently."""
    try:
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        probe = config.SESSIONS_DIR / f".write-probe-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:
        out.append(Finding(
            HARD, "data_dir_unwritable",
            "Lena kann ihre Sitzungsdaten nicht speichern.",
            f"{config.SESSIONS_DIR}: {type(e).__name__}: {e}",
            "Prüfe die Schreibrechte für den Ordner."))
        return
    try:
        import shutil
        free = shutil.disk_usage(config.DATA_DIR).free
        if free < 512 * 1024 * 1024:
            out.append(Finding(
                WARN, "disk_low",
                "Wenig freier Speicherplatz auf der Festplatte.",
                f"{free / 1048576:.0f} MiB free at {config.DATA_DIR}",
                "Ein langes Gespräch könnte nicht vollständig gespeichert werden."))
    except Exception:
        pass


def _path_length(out: list[Finding]):
    """The MAX_PATH trap that Item 2 found, on the run path this time."""
    venv = config.ROOT / "lab" / ".venv"
    n = len(str(venv.resolve())) if venv.exists() else len(str(config.ROOT))
    if n > 116:
        out.append(Finding(
            WARN, "path_too_long",
            "Der Projektordner liegt sehr tief verschachtelt.",
            f"venv root {n} chars, limit 116 (MAX_PATH 260 minus 143 for the "
            f"deepest packaged file)",
            "Falls Lena nicht startet: verschiebe den Ordner nach C:\\Lena."))


def _audio(out: list[Finding], requested_device):
    """Device availability, and recovery when the saved microphone is gone.

    Returns the device index that should actually be used. A saved index is
    just a number: unplug the headset it referred to and that number now means
    a different device, or nothing at all. Failing at startup is bad; failing
    silently mid-sentence is worse.
    """
    from . import devices
    inputs = devices.list_inputs()
    if not inputs:
        out.append(Finding(
            HARD, "no_microphone",
            "Es wurde kein Mikrofon gefunden.",
            "sounddevice reported no input devices",
            "Schließe ein Mikrofon an und starte Lena neu."))
        return requested_device

    have = {d["index"] for d in inputs}
    device = requested_device
    if requested_device is not None and requested_device not in have:
        fallback = devices.default_input(inputs)
        out.append(Finding(
            WARN, "microphone_gone",
            "Das zuletzt benutzte Mikrofon ist nicht mehr da.",
            f"saved input_device {requested_device} not among {sorted(have)}; "
            f"falling back to {fallback}",
            f"Lena benutzt jetzt: {devices.describe(fallback)}."))
        device = fallback

    chosen = next((d for d in inputs if d["index"] == device), None)
    if chosen and chosen.get("warning"):
        # devices.default_input already avoids narrowband devices; this only
        # fires when the user picked one deliberately or it is all there is.
        out.append(Finding(
            WARN, "microphone_narrowband",
            f"Das Mikrofon „{chosen['name']}“ liefert nur Telefonqualität.",
            f"device {device} default rate {chosen.get('sample_rate')} Hz",
            "Die Spracherkennung wird dadurch deutlich schlechter."))

    try:
        import sounddevice as sd
        sd.query_devices(kind="output")
    except Exception as e:
        out.append(Finding(
            HARD, "no_speaker",
            "Es wurde keine Audioausgabe gefunden.",
            f"{type(e).__name__}: {e}",
            "Schließe Kopfhörer oder Lautsprecher an und starte Lena neu."))
    return device


def _settings(out: list[Finding]):
    """Report what settings.load() had to repair, instead of repairing silently.

    A corrupt settings file is already survivable -- load() falls back to
    defaults. What was missing is the user being told, and the broken file
    being kept instead of overwritten on the next save.
    """
    from . import settings
    path = settings.PATH
    if not path.exists():
        return
    import json
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"top level is {type(raw).__name__}, expected object")
    except Exception as e:
        backup = path.with_suffix(f".broken-{time.strftime('%Y%m%d-%H%M%S')}.json")
        try:
            path.replace(backup)
            detail = f"{type(e).__name__}: {e}; moved to {backup.name}"
            remedy = "Die alte Datei wurde gesichert."
        except Exception as e2:
            detail = f"{type(e).__name__}: {e}; backup failed: {e2}"
            remedy = ""
        out.append(Finding(
            WARN, "settings_corrupt",
            "Die Einstellungen waren beschädigt und wurden zurückgesetzt.",
            detail, remedy))
        return
    unknown = [k for k in raw if k not in settings.DEFAULTS]
    if unknown:
        # An older or newer schema. Unknown keys are ignored by load() and
        # preserved on disk until the next save; say so rather than pretending.
        out.append(Finding(
            INFO, "settings_unknown_keys",
            "Einige gespeicherte Einstellungen sind unbekannt und werden ignoriert.",
            f"unknown keys: {', '.join(sorted(unknown))}"))


# ─────────────────────────────────────────────────────────────── entry point
def run(requested_device=None) -> tuple[list[Finding], object]:
    """Every check, cheapest first. Returns (findings, device_to_use).

    Never raises: a preflight that crashes is worse than the failure it looks
    for. Anything unexpected becomes a WARN and the launch continues.
    """
    out: list[Finding] = []
    device = requested_device
    for fn in (_settings, _storage, _path_length, _artifacts, _whisper, _gpu,
               _ports):
        try:
            fn(out)
        except Exception as e:
            out.append(Finding(WARN, "check_error",
                               "Eine Startprüfung konnte nicht ausgeführt werden.",
                               f"{fn.__name__}: {type(e).__name__}: {e}"))
    try:
        device = _audio(out, requested_device)
    except Exception as e:
        out.append(Finding(WARN, "check_error",
                           "Die Audioprüfung konnte nicht ausgeführt werden.",
                           f"_audio: {type(e).__name__}: {e}"))
    return out, device


def blockers(findings) -> list[Finding]:
    return [f for f in findings if f.severity == HARD]


def format_console(findings) -> str:
    """Operator-facing summary. Technical detail belongs here, not in the UI."""
    lines = []
    for f in findings:
        if f.severity == INFO and f.code == "gpu":
            continue                      # already printed in the header
        tag = {"HARD": "FEHLER", "WARN": "HINWEIS", "INFO": "INFO"}[f.severity]
        lines.append(f"  [{tag}] {f.message}"
                     + (f" {f.remedy}" if f.remedy else ""))
        if f.detail:
            lines.append(f"          ({f.detail})")
    return "\n".join(lines)

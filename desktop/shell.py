"""Lena — the desktop window.

This is the application's entry point once it is installed. It is a SHELL: it
opens a window, starts the existing engine inside this same process, and shows
the page the engine already serves. It contains no product logic, no audio, no
model handling and no conversation code, and it must stay that way.

    runtime\\pythonw.exe  desktop\\shell.py

WHY A WEBVIEW AND NOT A BROWSER
  `app/static/index.html` does no audio at all -- no getUserMedia, no
  AudioContext, no MediaRecorder. Every microphone and speaker operation is in
  Python. So the window needs no media permissions, which is what makes hosting
  the existing UI in a WebView a small change instead of a project.

WHY THE ENGINE RUNS IN THIS PROCESS
  Closing the window must run the ordered shutdown that the evaluation depends
  on (log frozen -> analyzer drained -> evaluation -> persisted -> broadcast ->
  only then the LLM may stop). Owning the engine in-process makes that a
  function call instead of a race against a dying subprocess. `llama-server`
  remains a child subprocess, so the CUDA isolation E0 established is untouched.

⚑ NOTHING HERE MAY WEAKEN RUNTIME CONTAINMENT. The watchdog, the real-generation
liveness check, the lifecycle restart, the port handling and app/preflight.py
all still run; this file only decides how their results are shown.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

# ── environment, before anything heavy is imported ──────────────────────────
# ⚑ PYTHONNOUSERSITE: an embedded CPython with `import site` still puts the
# USER's %APPDATA%\Python\Python310\site-packages on sys.path. On the machine
# this was built on, that leaked a TensorFlow install into the runtime. Without
# this the shipped product silently inherits whatever the user pip-installed.
os.environ.setdefault("PYTHONNOUSERSITE", "1")
# ⚑ pythonnet must bind the IN-BOX .NET Framework, not .NET Core. 4.8 ships with
# Windows 10 1903+ and Windows 11; .NET Core may simply not be there.
os.environ.setdefault("PYTHONNET_RUNTIME", "netfx")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ⚑ PACKAGED vs DEVELOPMENT, decided by what is on disk, not by a flag someone
# has to remember to pass. An installed tree carries BUILD.json and ships no
# `lab/models`; the development tree is the other way round. In development the
# shell must stay a thin window over the existing checkout, with assets where
# `setup.cmd` already put them.
PACKAGED = (ROOT / "BUILD.json").exists() and not (ROOT / "lab" / "models").is_dir()
if PACKAGED:
    # Program Files is read-only after install; assets live per-user.
    os.environ.setdefault(
        "LENA_ASSET_ROOT",
        str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Lena"))

APP_NAME = "Interview Desk"
WINDOW_TITLE = "Interview Desk — Call Center Interview"
MIN_SIZE = (860, 620)
DEFAULT_SIZE = (1180, 820)


# ── logging: on disk for developers, never in the user's face ───────────────
def _log_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


LOG_PATH = _log_dir() / "desktop.log"


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    # stdout exists under python.exe (developer run) and is a no-op under
    # pythonw.exe (installed run), so this costs nothing either way.
    try:
        print(line, flush=True)
    except Exception:
        pass


# ── the splash the user sees while the stack warms ──────────────────────────
# Deliberately not a progress bar: boot is 16-26 s of model loading with no
# honest percentage available, and a fake one is a lie. Named states only.
SPLASH = """<!doctype html><meta charset="utf-8">
<style>
  :root { color-scheme: light dark; }
  html,body { height:100%; margin:0; }
  body { display:flex; align-items:center; justify-content:center;
         background:#f6f5f3; color:#2b2b2b;
         font:16px/1.55 "Segoe UI",system-ui,sans-serif; }
  @media (prefers-color-scheme: dark){ body{ background:#191919; color:#ececec; } }
  .box { text-align:center; max-width:34rem; padding:2rem; }
  .name { font-size:2.1rem; font-weight:300; letter-spacing:.04em; margin:0 0 .2rem; }
  .sub { opacity:.55; margin:0 0 2.1rem; font-size:.95rem; }
  .state { font-size:1.02rem; margin:0 0 .5rem; }
  .detail { opacity:.5; font-size:.85rem; min-height:1.2em; white-space:pre-line; }
  .dots span { display:inline-block; width:7px; height:7px; margin:0 3px;
    border-radius:50%; background:currentColor; opacity:.25;
    animation:p 1.25s infinite ease-in-out; }
  .dots span:nth-child(2){ animation-delay:.16s } .dots span:nth-child(3){ animation-delay:.32s }
  @keyframes p { 0%,80%,100%{opacity:.22;transform:translateY(0)} 40%{opacity:.75;transform:translateY(-3px)} }
  .err { color:#b3261e; text-align:left; white-space:pre-line; }
  @media (prefers-color-scheme: dark){ .err{ color:#ff8a80; } }
  .remedy { margin-top:1rem; opacity:.75; font-size:.9rem; text-align:left; white-space:pre-line; }
</style>
<div class="box">
  <p class="name">Interview Desk</p>
  <p class="sub">German & English call-center interviews</p>
  <p class="state" id="state">Wird gestartet…</p>
  <p class="detail" id="detail"></p>
  <p class="dots" id="dots"><span></span><span></span><span></span></p>
</div>
<script>
  window.setState = (s, d) => {
    document.getElementById('state').textContent = s;
    document.getElementById('detail').textContent = d || '';
  };
  window.setError = (title, body) => {
    document.getElementById('dots').style.display = 'none';
    const st = document.getElementById('state');
    st.textContent = title; st.className = 'state err';
    document.getElementById('detail').innerHTML =
      '<span class="remedy">' + (body || '').replace(/</g,'&lt;') + '</span>';
  };
</script>"""


class Shell:
    def __init__(self):
        self.window = None
        self.server = None
        self.ready = threading.Event()
        self.failed: tuple[str, str] | None = None
        self._shutdown_done = threading.Event()
        self._closing = threading.Event()

    # -------------------------------------------------------------- splash
    def say(self, state: str, detail: str = ""):
        log(f"state: {state}  {detail}")
        if self.window is None or self._closing.is_set():
            return
        try:
            self.window.evaluate_js(
                f"window.setState({state!r}, {detail!r})")
        except Exception:
            pass

    def fail(self, title: str, body: str):
        log(f"FAILED: {title} | {body}")
        self.failed = (title, body)
        if self.window is None or self._closing.is_set():
            return
        try:
            self.window.evaluate_js(f"window.setError({title!r}, {body!r})")
        except Exception:
            pass

    # -------------------------------------------------------------- backend
    def start_backend(self):
        """Provision if needed, then run the existing server in this process."""
        try:
            # ── FIRST RUN vs NORMAL USE, kept strictly apart ──────────────
            # The check is a few file stats on every launch; provisioning only
            # happens when something is genuinely missing. `start.cmd` never
            # downloaded anything and neither does a normal launch here.
            import firstrun
            if PACKAGED and not firstrun.is_provisioned():
                log("not provisioned; running first-run setup")
                self.say("Interview Desk wird vorbereitet…")
                ok, msg = firstrun.FirstRun(on_state=self.say).run()
                if not ok:
                    self.fail("Interview Desk konnte nicht eingerichtet werden.",
                              f"{msg}\n\nDetails: {LOG_PATH}")
                    return
                log("first-run setup complete")
            if PACKAGED:
                firstrun.activate()

            from app import config, preflight

            self.say("System wird geprüft…")
            findings, device = preflight.run(None)
            for f in findings:
                log(f"preflight {f.severity} {f.code}: {f.message} | {f.detail}")
            hard = preflight.blockers(findings)
            if hard:
                # ⚑ The product-level error. The German message and its remedy
                # come from preflight; the technical detail stays in the log.
                self.fail(hard[0].message,
                          "\n".join(f.remedy for f in hard if f.remedy)
                          + f"\n\nDetails: {LOG_PATH}")
                return

            self.say("Interviewmodell wird geladen…",
                     "Beim ersten Start dauert das etwas länger.")

            from app import server as srv
            self.server = srv

            # Reuse the real entry point, minus the browser it would open.
            sys.argv = [sys.argv[0], "--no-browser"]
            self._run_server(srv)
        except Exception as e:
            log("backend crashed:\n" + traceback.format_exc())
            self.fail("Interview Desk konnte nicht starten.",
                      f"Ein unerwarteter Fehler ist aufgetreten.\n"
                      f"Details: {LOG_PATH}")

    def _run_server(self, srv):
        """server.main() with our event hook, so the splash can narrate boot."""
        original = srv.broadcast

        def hooked(ev):
            kind = ev.get("kind")
            if kind == "boot":
                self.say(ev.get("detail") or "Wird gestartet…")
            elif kind == "boot_failed":
                self.fail(ev.get("message") or "Start fehlgeschlagen.",
                          f"Details: {LOG_PATH}")
            elif kind == "ready":
                self.ready.set()
            return original(ev)

        srv.broadcast = hooked
        try:
            rc = srv.main()
            log(f"server.main() returned {rc}")
        finally:
            srv.broadcast = original
            self._shutdown_done.set()

    # ------------------------------------------------------------- lifecycle
    def wait_and_show(self):
        """Poll until the engine serves, then swap the splash for the UI."""
        from app import config
        url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}/"
        deadline = time.perf_counter() + 240
        while time.perf_counter() < deadline:
            if self.failed:
                return
            if self.ready.is_set():
                # The engine says ready; make sure the page is actually served
                try:
                    urllib.request.urlopen(url, timeout=3).read(64)
                except Exception:
                    time.sleep(0.3)
                    continue
                log(f"engine ready, loading {url}")
                try:
                    self.window.load_url(url)
                except Exception:
                    log("load_url failed:\n" + traceback.format_exc())
                return
            time.sleep(0.25)
        if not self.failed:
            self.fail("Interview Desk braucht ungewöhnlich lange zum Starten.",
                      f"Bitte schließe das Fenster und versuche es erneut.\n"
                      f"Details: {LOG_PATH}")

    def on_closing(self):
        """Window close -> start the product's own ordered shutdown.

        ⚑ RETURNS IMMEDIATELY, ON PURPOSE. This runs on the GUI thread, and
        pywebview's message loop is that thread: waiting here for the shutdown
        to finish deadlocks the window instead of closing it (observed — the
        process had to be killed after 150 s). So the window closes at once and
        `main()` waits for the shutdown afterwards, on the main thread, with no
        GUI loop left to starve.

        ⚑ The shutdown itself must NOT be short-circuited: it is the path that
        produces the post-session evaluation, and E10 established that killing
        the LLM before it finishes is what once surfaced as "StallError".
        """
        log("window closing; starting ordered shutdown")
        self._closing.set()
        try:
            if self.server is not None:
                threading.Thread(
                    target=lambda: self.server._shutdown("user_ended"),
                    daemon=True).start()
        except Exception:
            log("shutdown start error:\n" + traceback.format_exc())
        return True

    def wait_for_shutdown(self, timeout: float = 120.0):
        """Called from main() after the window is gone. Keeps the process alive
        until the evaluation has been written, then lets it exit."""
        if self.server is None:
            return
        log("waiting for ordered shutdown to finish")
        if not self._shutdown_done.wait(timeout=timeout):
            log(f"shutdown did not finish within {timeout}s; exiting anyway")
        else:
            log("ordered shutdown finished")


# ⚑ Matches `AppMutex` in installer.iss, so an upgrade can detect a running
# Lena and ask the user to close it rather than replacing files underneath it.
# Held for the process's lifetime; Windows releases it when the process exits.
SINGLE_INSTANCE_MUTEX = "InterviewDeskSingleInstance"
_mutex_handle = None


def claim_single_instance() -> bool:
    """True if THIS process is the one instance. Never blocks."""
    global _mutex_handle
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.CreateMutexW.restype = wintypes.HANDLE
        _mutex_handle = k32.CreateMutexW(None, False, SINGLE_INSTANCE_MUTEX)
        return k32.GetLastError() != 183          # ERROR_ALREADY_EXISTS
    except Exception:
        return True            # a mutex problem must never block a launch


def already_running() -> bool:
    """One instance only. A second model stack would fight for VRAM and 8790."""
    try:
        from app import config
        with urllib.request.urlopen(
                f"http://{config.SERVER_HOST}:{config.SERVER_PORT}/api/config",
                timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def activate_existing_window() -> bool:
    """Restore and focus the real app instead of showing a second-instance box."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, WINDOW_TITLE)
        if not hwnd:
            return False
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def main() -> int:
    log("=" * 60)
    log(f"{APP_NAME} desktop shell starting; python {sys.version.split()[0]}")
    log(f"root={ROOT}")

    if not claim_single_instance() or already_running():
        # ⚑ Do NOT boot a second engine. Tell the user and leave the running
        # instance alone; its window is the one they want.
        log("another instance is already serving; refusing to start a second")
        if activate_existing_window():
            log("existing Interview Desk window restored and focused")
            return 0
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None,
                "Interview Desk is already open or is finishing a report.\n\n"
                "البرنامج مفتوح بالفعل أو ينهي حفظ التقرير.",
                APP_NAME, 0x40)
        except Exception:
            pass
        return 0

    import webview

    shell = Shell()
    shell.window = webview.create_window(
        WINDOW_TITLE, html=SPLASH,
        width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1],
        min_size=MIN_SIZE, resizable=True, confirm_close=False,
        background_color="#f6f5f3")
    shell.window.events.closing += shell.on_closing

    def worker():
        threading.Thread(target=shell.start_backend, daemon=True).start()
        shell.wait_and_show()

    try:
        webview.start(worker, private_mode=False, debug=False)
    except Exception:
        log("webview failed:\n" + traceback.format_exc())
        return 1
    shell.wait_for_shutdown()
    log("window closed; exiting")

    # ⚑ HARD EXIT, DELIBERATELY. `webview.start()` returns and `main()` finishes,
    # but the process does NOT end: the WinForms/CLR host keeps foreground
    # threads alive, and the observed result was a Python process still running
    # after the window was gone -- with the port released and llama-server
    # already reaped, but the process itself never exiting.
    #
    # This is safe HERE and only here, because it runs strictly after
    # `wait_for_shutdown()`: the session log is closed, the mistake analyzer is
    # drained, the evaluation is written and fsync'd, `llama-server` is stopped
    # and the socket is closed. The only thing `os._exit` skips is
    # `server._emergency_close`, whose entire job is to close a log that this
    # path has already closed.
    #
    # Leaving a stray interpreter behind would be worse than skipping an atexit
    # hook that has nothing left to do.
    sys.stdout.flush() if sys.stdout else None
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

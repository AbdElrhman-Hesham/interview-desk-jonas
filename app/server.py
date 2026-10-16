"""Minimal local server: static page + Server-Sent Events + a few POST routes.

Deliberately stdlib-only -- no FastAPI, no websockets, no build step. The browser
is a thin VIEW; all audio and all state live in Python (audio stays in Python
for device control and deterministic buffers).

⚑ The UI must never see a traceback, an HTTP 500 body, a port number or the
words "llama-server". Failures are translated into product language before they
leave this module; the technical detail goes to the session log instead.
"""
from __future__ import annotations

import atexit
import json
import queue
import signal
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import (config, devices, evaluate as evaluator, interview as interview_mode,
               interview_report, preflight, settings as prefs, teaching)
from .engine import Engine, State

STATIC = config.APP_DIR / "static"
_subscribers: list[queue.Queue] = []
_sub_lock = threading.Lock()
engine: Engine | None = None
_recent: list[dict] = []          # replay buffer so a late/reloaded UI catches up
_RECENT_MAX = 60


def broadcast(event: dict):
    with _sub_lock:
        if event.get("kind") == "interview_reset":
            _recent.clear()
        elif event.get("kind") in ("boot", "ready", "state", "boot_failed",
                                 "transcript", "recovering", "turn_error",
                                 "mistake", "handsfree", "disputed",
                                 "evaluating", "vocab", "progress",
                                 "level_suggestion"):
            _recent.append(event)
            del _recent[:-_RECENT_MAX]
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(event)
            except Exception:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                  # keep the console clean

    def _trusted(self) -> bool:
        """Only this app's own window (and the desktop shell) may talk to us.

        Binding to 127.0.0.1 is not enough: any page in the learner's browser can
        send a `text/plain` POST here without a CORS preflight, and a DNS-rebinding
        page arrives with its own Host name. Both would otherwise read the learner
        profile or report, or write into settings and the teacher prompt.
        """
        port = self.server.server_address[1]
        local = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = (self.headers.get("Host") or "").strip().lower()
        origin = (self.headers.get("Origin") or "").strip().lower()
        return host in local and (not origin or origin in {"http://" + h for h in local})

    # ------------------------------------------------------------------- GET
    def do_GET(self):
        if not self._trusted():
            return self.send_error(403)
        if self.path in ("/", "/index.html"):
            return self._file(STATIC / "index.html", "text/html; charset=utf-8")
        if self.path == "/events":
            return self._sse()
        if self.path == "/api/config":
            s = prefs.load()
            return self._json({
                "levels": list(config.LEVELS),
                "level": engine.level if engine else s["level"],
                "model_status": config.MODEL_STATUS,
                "flags": {k: v[1] for k, v in config.FEEDBACK_FLAGS.items()},
                "show_transcript": s["show_transcript"],
                "handsfree": engine.handsfree if engine else s["handsfree"],
                "show_mistakes": s["show_mistakes"],
                "show_vocab": s["show_vocab"],
                "show_progress": s["show_progress"],
                "panel": s["panel"],
                "teacher": teaching.name(),
                "sessions_before": (engine.learner.profile.data.get("sessions", 0)
                                    if engine and engine.learner else 0),
                "endpoint_silence_s": config.ENDPOINT_SILENCE_S,
                "state": engine.state if engine else State.INITIALIZING,
                "boot_error": engine.boot_error if engine else None,
                "interview_language": s.get("interview_language", "de"),
                "interview": (engine.interview.snapshot()
                              if engine and engine.interview else {"active": False}),
            })
        if self.path == "/api/mistakes":
            m = engine.mistakes if engine else None
            return self._json({
                "items": (m.items if m else []),
                "stats": (m.stats if m else {}),
                # ⚑ surfaced so the UI can label it honestly
                "status": "PROVISIONAL_pending_E2",
            })
        if self.path == "/api/vocab":
            m = engine.mistakes if engine else None
            return self._json({"items": (m.vocab if m else [])})
        if self.path == "/api/learner":
            # ⚑ COUNTED ONLY. Nothing in this response is a judgment about the
            # learner -- that is the post-session evaluation's job, and it says
            # so itself. Here: turns, words, seconds, cards, corrections.
            lm = engine.learner if engine else None
            prof = lm.profile.data if lm else {}
            return self._json({
                "session": (lm.snapshot() if lm else {}),
                "profile": {
                    "sessions": prof.get("sessions", 0),
                    "total_turns": prof.get("total_turns", 0),
                    "level": prof.get("level"),
                    "difficulty": prof.get("difficulty"),
                    "recurring_errors": (lm.profile.recurring_errors if lm else []),
                    "vocabulary_total": len(prof.get("vocabulary") or []),
                    "topics_done": (prof.get("topics_done") or [])[-10:],
                    "history": (prof.get("history") or [])[-8:],
                },
            })
        if self.path == "/api/devices":
            s = prefs.load()
            return self._json({
                "devices": devices.list_inputs(),
                "selected": s["input_device"],
                "auto": devices.default_input(),
            })
        if self.path.startswith("/api/questions"):
            # /api/questions?lang=de  (or en)
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            lang = (qs.get("lang") or ["de"])[0]
            return self._json({
                "language": lang,
                "questions": interview_mode.question_bank(lang),
            })
        if self.path == "/api/interview/report/download":
            return self._serve_report()
        self.send_error(404)

    def _file(self, p: Path, ctype: str):
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse(self):
        q: queue.Queue = queue.Queue(maxsize=500)
        with _sub_lock:
            _subscribers.append(q)
            backlog = list(_recent)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            # replay so a reloaded page immediately knows the current state
            for ev in backlog:
                self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15)
                    self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except Exception:
            pass
        finally:
            with _sub_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    def _serve_report(self):
        """Secure download of the current interview's PDF report only."""
        if engine is None or engine._last_report_path is None:
            return self._json({"ok": False, "reason": "no_report"}, 404)
        from pathlib import Path as _P
        path = _P(engine._last_report_path).resolve()
        if not path.suffix.lower() == ".pdf" or not path.is_file():
            return self._json({"ok": False, "reason": "not_found"}, 404)
        # Security: the resolved path must be inside a known reports directory.
        try:
            allowed = interview_report.report_dir().resolve()
            if allowed not in path.parents:
                return self._json({"ok": False, "reason": "forbidden"}, 403)
        except Exception:
            return self._json({"ok": False, "reason": "forbidden"}, 403)
        try:
            data = path.read_bytes()
        except Exception:
            return self._json({"ok": False, "reason": "read_error"}, 500)
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""   # drain first: an unread body turns the 403 into a reset
        if not self._trusted():
            return self.send_error(403)
        try:
            body = json.loads(raw or b"{}") if n else {}
        except Exception:
            body = {}
        p = self.path

        if engine is None or engine.boot_error:
            return self._json({"ok": False, "reason": "not_ready"}, 200)

        try:
            if p == "/api/interview/start":
                params, err = interview_mode.validate_start_params(body)
                if err:
                    return self._json({"ok": False, "reason": err})
                language = params["language"]
                snap = engine.start_interview(
                    language,
                    selected_ids=params.get("selected_ids"),
                    count=params.get("count"))
                prefs.save(interview_language=language)
                return self._json({"ok": True, "interview": snap})
            if p == "/api/ptt/start":
                return self._json({"ok": engine.start_recording(),
                                   "state": engine.state})
            if p == "/api/ptt/stop":
                return self._json({"ok": engine.stop_recording(),
                                   "state": engine.state})
            if p == "/api/flag":
                engine.flag(body.get("code", "?"), body.get("note", ""))
                return self._json({"ok": True})
            if p == "/api/level":
                lv = body.get("level")
                if engine.set_level(lv):
                    prefs.save(level=lv)
                    broadcast({"kind": "level", "level": lv})
                return self._json({"ok": True, "level": engine.level})
            if p == "/api/dispute":
                rec = engine.dispute_transcript(body.get("turn"))
                return self._json({"ok": rec is not None, "turn": (rec or {}).get("turn")})
            if p == "/api/vocab/save":
                rec = (engine.mistakes.save_vocab(body.get("id", ""),
                                                  bool(body.get("saved", True)))
                       if engine.mistakes else None)
                return self._json({"ok": rec is not None})
            if p == "/api/goal":
                # One line the learner can set about what he wants to practise.
                # It goes into the teacher's prompt, so it has to be short.
                g = (body.get("goal") or "").strip()[:120]
                if engine.learner is not None:
                    goals = engine.learner.profile.data.setdefault("goals", [])
                    if g and g not in goals:
                        goals.append(g)
                        del goals[:-3]
                    elif not g:
                        goals.clear()
                    engine.learner.profile.save()
                    if engine.messages:
                        engine.messages[0] = {"role": "system",
                                              "content": engine._system_prompt()}
                return self._json({"ok": True,
                                   "goals": (engine.learner.profile.data.get("goals")
                                             if engine.learner else [])})
            if p == "/api/mistake/flag":
                rec = (engine.mistakes.flag(body.get("id", ""), body.get("flag", ""))
                       if engine.mistakes else None)
                return self._json({"ok": rec is not None})
            if p == "/api/handsfree":
                on = engine.set_handsfree(bool(body.get("enabled")))
                prefs.save(handsfree=on)
                return self._json({"ok": True, "enabled": on})
            if p == "/api/settings":
                saved = prefs.save(**{k: v for k, v in body.items()
                                      if k in prefs.DEFAULTS})
                return self._json({"ok": True, "settings": saved})
            if p == "/api/interview/finish":
                # Finalize the interview (evaluate + PDF) without shutting
                # down the whole app. The Home button needs the engine alive.
                threading.Thread(target=_finish_interview, daemon=True).start()
                return self._json({"ok": True})
            if p == "/api/interview/home":
                # Return to the setup screen for a new interview
                if engine:
                    engine.reset_interview()
                return self._json({"ok": True})
            if p == "/api/end":
                threading.Thread(target=_shutdown, daemon=True).start()
                return self._json({"ok": True})
        except Exception as e:
            # product language out; technical detail into the log only
            if engine and engine.log:
                engine.log.event("api_error", path=p,
                                 detail=f"{type(e).__name__}: {e}")
            return self._json({"ok": False, "reason": "internal"}, 200)
        self.send_error(404)


_httpd: ThreadingHTTPServer | None = None
# ⚑ TWO events, not one. This was a real production bug.
#
# There used to be a single `_shutdown_done`, set at the TOP of _shutdown() for
# idempotency. main() waits on that event and then runs `finally: engine.stop()`,
# which stops llama-server. So the moment _shutdown() began, main() was released
# and killed the LLM WHILE _shutdown() was still running the evaluation. The
# evaluation's very first request hit a dead server and came back as a
# connection error in 44.9 ms -- surfaced to the user as "StallError", because
# LlamaCppProvider.stream_chat wraps every exception in StallError.
#
# `_shutdown_started` gives idempotency; `_shutdown_done` now means "the whole
# shutdown sequence, including evaluation, has finished, the LLM may die".
_shutdown_started = threading.Event()
_shutdown_done = threading.Event()


def _shutdown(reason: str = "user_ended"):
    """Ordered teardown. The LLM stays alive until the evaluation is finished.

    Invariant:
      conversation ends
        -> conversation data becomes immutable (log closed)
        -> background LLM work settles (mistake analysis drained)
        -> evaluation gets the LLM to itself
        -> evaluation persisted
        -> UI receives it
        -> only then may the LLM/server stop
    """
    if _shutdown_started.is_set():
        return
    _shutdown_started.set()
    import time
    summary, evaluation = {}, None
    try:
        # 1. freeze the conversation data
        try:
            if engine:
                summary = engine.log.close(engine.llm.stalls, reason=reason) \
                    if (engine.log and not engine.log.closed) else {}
        except Exception:
            pass

        # 2. let outstanding background LLM work settle BEFORE evaluating.
        # llama-server runs -np 1; a mistake analysis still in flight would sit
        # in front of the evaluation on the single slot.
        try:
            if engine and engine.mistakes:
                engine.mistakes.drain(timeout=config.EVALUATION_DRAIN_S)
        except Exception:
            pass

        # 3. evaluation, with the LLM guaranteed alive
        try:
            interview_ready = bool(engine and engine.interview
                                   and engine.interview.answers)
            standard_ready = bool(engine and engine.log
                                  and len(engine.log.turns) >= config.EVALUATION_MIN_TURNS)
            if (engine and engine.llm and config.EVALUATION_ENABLED
                    and reason == "user_ended" and engine.log
                    and (interview_ready or standard_ready)):
                broadcast({"kind": "evaluating"})
                t0 = time.perf_counter()
                evaluation = (interview_report.finalize(engine, summary)
                              if engine.interview is not None
                              else evaluator.build(engine, summary))
                evaluation["eval_wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                _persist_evaluation(evaluation)
        except Exception as e:
            try:
                if engine and engine.log:
                    engine.log.event("evaluation_error",
                                     detail=f"{type(e).__name__}: {e}")
            except Exception:
                pass

        # 4. hand the result to the UI
        broadcast({"kind": "session_end", "summary": summary,
                   "evaluation": evaluation})
        time.sleep(1.0)
    finally:
        # 5. ONLY NOW may the LLM stop. Releasing main() before this point is
        # what killed the evaluation mid-request.
        _shutdown_done.set()

    try:
        if engine:
            engine.stop(reason=reason)
    except Exception:
        pass
    if _httpd:
        threading.Thread(target=_httpd.shutdown, daemon=True).start()


_finish_lock = threading.Lock()


def _finish_interview():
    """Evaluate and create the PDF without tearing the engine down.

    This is the non-destructive counterpart to _shutdown(). The LLM stays
    alive so the user can press Home and start a fresh interview.
    """
    import time as _time
    if not _finish_lock.acquire(blocking=False):
        return  # already running
    try:
        summary, evaluation = {}, None
        # 0. block new recording / turns immediately
        if engine:
            engine.begin_finishing()
        # 1. close the session log (makes data immutable)
        try:
            if engine and engine.log and not engine.log.closed:
                summary = engine.log.close(
                    engine.llm.stalls if engine.llm else [],
                    reason="interview_finished")
        except Exception:
            pass
        # 2. drain background LLM work
        try:
            if engine and engine.mistakes:
                engine.mistakes.drain(timeout=config.EVALUATION_DRAIN_S)
        except Exception:
            pass
        # 3. evaluate
        try:
            if (engine and engine.llm and engine.interview
                    and engine.interview.answers):
                broadcast({"kind": "evaluating"})
                t0 = _time.perf_counter()
                evaluation = interview_report.finalize(
                    engine, summary, open_pdf=False)
                evaluation["eval_wall_ms"] = round(
                    (_time.perf_counter() - t0) * 1000, 1)
                # Store for the secure download endpoint
                engine._last_report_path = evaluation.get("report_path")
                _persist_evaluation(evaluation)
        except Exception as e:
            try:
                if engine and engine.log:
                    engine.log.event("evaluation_error",
                                     detail=f"{type(e).__name__}: {e}")
            except Exception:
                pass
        # 4. broadcast to the UI (includes per-question feedback)
        broadcast({"kind": "session_end", "summary": summary,
                   "evaluation": evaluation})
    finally:
        _finish_lock.release()


def _persist_evaluation(ev: dict):
    """Two homes: inside the session file, and one append-only evaluations
    file. The second exists so a future progress view can read every session's
    structured evaluation without reparsing every log -- the schema is stored
    now precisely so we do not regret it later. Nothing reads it yet."""
    try:
        if engine and engine.log:
            with engine.log.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
                f.flush()
    except Exception:
        pass
    try:
        import os
        with (config.DATA_DIR / "evaluations.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass



def _emergency_close():
    """Last line of defence: if the process dies without /api/end, the session
    must still be closed and MARKED abnormal rather than silently truncated."""
    try:
        if engine and engine.log and not engine.log.closed:
            engine.log.close(engine.llm.stalls if engine.llm else [],
                             reason="crash")
    except Exception:
        pass


def main():
    global engine, _httpd
    import argparse
    ap = argparse.ArgumentParser(prog="Interview Desk")
    ap.add_argument("--level", default=None, choices=list(config.LEVELS))
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--handsfree", action="store_true",
                    help="force automatic turn-taking (already the default)")
    ap.add_argument("--ptt", action="store_true",
                    help="start in push-to-talk instead of hands-free")
    args = ap.parse_args()

    if args.list_devices:
        for d in devices.list_inputs():
            mark = "  <-- Standard" if d["index"] == devices.default_input() else ""
            warn = f"   ⚠ {d['warning']}" if d["warning"] else ""
            print(f"{d['index']:>3}  {d['name'][:48]:<50} "
                  f"{d['sample_rate'] or '?':>6} Hz{mark}{warn}")
        return 0

    saved = prefs.load()
    level = args.level or saved["level"]
    device = args.device if args.device is not None else saved["input_device"]
    if device is None:
        device = devices.default_input()
    if args.device is not None:
        prefs.save(input_device=args.device)

    print("=" * 66)
    print("  Interview Desk — private call-center interview simulator")
    print("=" * 66)

    # ── preflight ────────────────────────────────────────────────────────
    # ⚑ Runs BEFORE the ~20 s boot, so a missing file is a sentence instead of
    # a FileNotFoundError fifteen seconds in. It may also CHANGE the device:
    # a saved microphone index is meaningless once that microphone is gone.
    findings, device = preflight.run(device)
    report = preflight.format_console(findings)
    hard = preflight.blockers(findings)

    print(f"  Niveau:    {level}")
    print(f"  Mikrofon:  {devices.describe(device)}")
    if report:
        print(report)

    if device != (args.device if args.device is not None else saved["input_device"]) \
            and any(f.code == "microphone_gone" for f in findings):
        # Persist the recovery, so the next launch does not repeat the warning
        # for a device that is not coming back.
        prefs.save(input_device=device)

    if hard:
        print()
        print("  " + "=" * 62)
        print("  LENA KANN NICHT STARTEN")
        print("  " + "=" * 62)
        for f in hard:
            print(f"  {f.message}")
            if f.remedy:
                print(f"    -> {f.remedy}")
        print()
        # If the window can still be served, show the same thing there rather
        # than leaving the learner with a console they may never look at.
        if not any(f.code == "ui_port_busy" for f in hard):
            try:
                _httpd = ThreadingHTTPServer((config.SERVER_HOST, config.SERVER_PORT),
                                             Handler)
                threading.Thread(target=_httpd.serve_forever, daemon=True).start()
                broadcast({"kind": "boot_failed",
                           "message": hard[0].message,
                           "detail": "\n".join(f.user_text for f in hard),
                           "code": hard[0].code})
                url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}/"
                if not args.no_browser:
                    webbrowser.open(url)
                print(f"  Details im Fenster: {url}")
                print("  Strg+C zum Beenden.")
                try:
                    _shutdown_done.wait(timeout=120)
                except KeyboardInterrupt:
                    pass
            except Exception:
                pass
        return 1

    atexit.register(_emergency_close)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_a: _shutdown("shutdown"))
        except Exception:
            pass

    # hands-free unless explicitly disabled
    handsfree = True if args.handsfree else (False if args.ptt else saved["handsfree"])
    print(f"  Modus:     {'Freihand (automatisch)' if handsfree else 'Push-to-Talk'}")
    engine = Engine(level=level, device=device, on_event=broadcast,
                    handsfree=handsfree)
    _httpd = ThreadingHTTPServer((config.SERVER_HOST, config.SERVER_PORT), Handler)
    threading.Thread(target=_httpd.serve_forever, daemon=True).start()
    url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}/"
    print(f"  Fenster:   {url}")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        engine.start()
        print("  Bereit. Leertaste gedrückt halten zum Sprechen.\n")
        _shutdown_done.wait()
    except RuntimeError as e:
        # boot failed; the UI already has a readable message via boot_failed.
        # Keep serving briefly so the user actually sees it instead of the
        # window dying on them.
        print(f"\n  Start fehlgeschlagen: {e}\n")
        print("  Das Fenster zeigt Details. Strg+C zum Beenden.")
        try:
            _shutdown_done.wait(timeout=300)
        except KeyboardInterrupt:
            pass
        return 1
    except KeyboardInterrupt:
        _shutdown("shutdown")
    finally:
        try:
            # _shutdown_done means the evaluation is finished, so stopping the
            # engine here can no longer cut it off.
            engine.stop(reason="shutdown" if not _shutdown_started.is_set()
                        else "user_ended")
        except Exception:
            pass
        if engine and engine.log:
            print("  Sitzung gespeichert:", engine.log.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The regression gate. One command, one PASS/FAIL.

    regress.cmd

Every check below already existed as a script somebody ran by hand. This file
adds no test logic of its own — it decides WHICH of them constitute the gate,
what each one needs in order to be meaningful, and how to run them so the
result is trustworthy. The checks themselves stay where they were written, next
to the experiment that produced them.

WHY SUBPROCESSES, NOT IMPORTS
  Each engine-level check boots llama-server on port 8790, loads faster-whisper
  onto the GPU and loads Piper. Two CUDA runtimes must not share one process
  (that is why llama.cpp is a subprocess in production), several harnesses patch
  module-level globals (`sd.InputStream`, `sd.OutputStream`,
  `config.ENDPOINT_NO_SPEECH_TIMEOUT_S`), and a check that crashes must not take
  the runner down with it. One process per check is the only isolation that
  actually holds. The cost is a fresh ~20 s boot per check; the gate is a
  minutes-long thing by nature.

WHAT THIS RUNNER GUARANTEES
  * a check that cannot run is SKIPPED and named, never quietly passed;
  * a run with any skip is INCOMPLETE, not PASS;
  * port 8790 is free before each check and reclaimed after it, so one check's
    leaked llama-server cannot fail the next one and be misread as a regression;
  * every check's full output is on disk, per run, whether it passed or not;
  * harness sessions are stamped LENA_HARNESS so a regression run can never
    manufacture something that later reads as real learner evidence.

EXIT CODES   0 = PASS   1 = FAIL   2 = INCOMPLETE (something was skipped)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

RESULTS_DIR = ROOT / "lab" / "results" / "regression"


# ═══════════════════════════════════════════════════════════ prerequisites
# A requirement is a named predicate returning (ok, detail). If it fails, every
# check that declares it is SKIPPED with that detail — never silently dropped,
# never counted as a pass.

def _req_llm():
    exe = config.LLAMA_BUILD_DIR / "llama-server.exe"
    if not config.LLM_MODEL_PATH.exists():
        return False, f"LLM model missing: {config.LLM_MODEL_PATH}"
    if not exe.exists():
        return False, f"llama-server.exe missing: {exe}"
    return True, f"{config.LLM_MODEL_PATH.name} + {config.LLAMA_BUILD_DIR.name}"


def _req_piper():
    p = Path(config.TTS_VOICE)
    if not p.exists():
        return False, f"Piper voice missing: {p}"
    if not p.with_suffix(".onnx.json").exists():
        return False, f"Piper voice config missing: {p.with_suffix('.onnx.json')}"
    return True, p.name


def _req_stt():
    try:
        import faster_whisper  # noqa: F401
    except Exception as e:
        return False, f"faster-whisper not importable: {type(e).__name__}: {e}"
    return True, config.STT_MODEL


def _req_gpu():
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        if n < 1:
            return False, "no CUDA device reported by NVML"
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        if isinstance(name, bytes):
            name = name.decode()
        return True, name
    except Exception as e:
        return False, f"NVML unavailable: {type(e).__name__}: {e}"


def _req_audio_out():
    try:
        import sounddevice as sd
        return True, sd.query_devices(kind="output")["name"]
    except Exception as e:
        return False, f"no default output device: {type(e).__name__}: {e}"


def _req_audio_in():
    try:
        import sounddevice as sd
        return True, sd.query_devices(kind="input")["name"]
    except Exception as e:
        return False, f"no default input device: {type(e).__name__}: {e}"


# ⚑ A QUIET ROOM IS A PREREQUISITE, exactly like a GPU.
#
# handsfree_halfduplex asserts that silence produces no turn: the mic is opened
# and nothing must happen. If the room is not actually silent the engine behaves
# CORRECTLY -- the speech gate fires, the endpointer switches from its no-speech
# branch to its trailing-silence branch, and the mic stays open -- and the check
# fails for a reason that has nothing to do with the code.
#
# That is not hypothetical. On run 20260810-214015 this check failed with
# silence_times_out=False and manual_fallback_works=False; the capture it wrote
# (s20260810-214149-t001.wav) is 11.65 s of real room sound, peak frame RMS
# 0.151 against a 0.004 gate and 1.84 s voiced against a 0.30 s speech-start
# threshold. The engine was right; the room was loud. On a quiet re-run the
# check passed. A gate that fails on ambient noise trains people to ignore it,
# so the condition is measured up front and named instead.
#
# The threshold is DERIVED FROM THE FAILURE MECHANISM, not guessed.
# `Engine._auto_listen_loop` accumulates voiced time and never decays it: once
# `voiced_total >= ENDPOINT_MIN_SPEECH_S` it declares speech started and leaves
# the no-speech branch for good. So what matters is not how loud one moment is,
# it is how much voiced time the room accumulates across the whole listening
# window. Probe long enough to estimate that rate, project it over the window
# the check actually uses, and require a safety factor.
#
# Measured for reference: E6 put this machine's silent room at max frame RMS
# 0.00293 (below the 0.004 floor) and 0.00 s voiced. The capture from the
# failing run accumulated 1.84 s voiced in 11.65 s = 0.158 voiced-s per second,
# roughly 3x over the line below.
_QUIET_PROBE_S = 3.0
# verify_handsfree.py shortens the shipping 45 s no-speech timeout to this.
_QUIET_WINDOW_S = 6.0
_QUIET_SAFETY = 1.5


def _req_quiet_room():
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as e:
        return False, f"cannot probe the room: {type(e).__name__}: {e}"
    try:
        rec = sd.rec(int(_QUIET_PROBE_S * config.RECORD_SR),
                     samplerate=config.RECORD_SR, channels=1, dtype="float32")
        sd.wait()
    except Exception as e:
        return False, f"microphone probe failed: {type(e).__name__}: {e}"
    x = np.asarray(rec, dtype=np.float32).reshape(-1)
    n = int(0.02 * config.RECORD_SR)
    if x.size < n:
        return False, "microphone probe returned no samples"
    f = x[: x.size // n * n].reshape(-1, n)
    rms = np.sqrt((f.astype(np.float64) ** 2).mean(axis=1))
    voiced_s = float((rms > config.SPEECH_RMS_FLOOR).sum()) * 0.02
    rate = voiced_s / _QUIET_PROBE_S
    projected = rate * _QUIET_WINDOW_S
    budget = config.ENDPOINT_MIN_SPEECH_S / _QUIET_SAFETY
    detail = (f"{_QUIET_PROBE_S:.0f}s probe: {voiced_s:.2f}s voiced @ floor "
              f"{config.SPEECH_RMS_FLOOR} (max frame {rms.max():.5f}) -> "
              f"{projected:.2f}s projected over the {_QUIET_WINDOW_S:.0f}s "
              f"listening window, budget {budget:.2f}s")
    if projected > budget:
        return False, (detail + f". At that rate the room alone reaches the "
                       f"{config.ENDPOINT_MIN_SPEECH_S}s speech-start threshold "
                       f"before the no-speech timeout fires, so the engine would "
                       f"be RIGHT to keep listening and the check would fail for "
                       f"a reason that is not the code. Quieten the room and "
                       f"re-run.")
    return True, detail


# The hands-free cycle check replays the user's OWN recordings through the
# capture callback. Without them it would be testing nothing real.
def _req_learner_wavs():
    got = sorted(config.AUDIO_DIR.glob("s20260808-181219-*.wav"))
    if len(got) < 3:
        return False, (f"need >=3 real learner recordings in {config.AUDIO_DIR} "
                       f"matching s20260808-181219-*.wav, found {len(got)}")
    return True, f"{len(got)} clips in {config.AUDIO_DIR.name}"


# The degradation check asserts exact numbers from one frozen real session.
_FIXTURE_SESSION = "s20260810-165738"


def _req_fixture_session():
    p = config.SESSIONS_DIR / f"{_FIXTURE_SESSION}.jsonl"
    if not p.exists():
        return False, (f"frozen evaluation fixture missing: {p}. The degradation "
                       f"check asserts exact measured values from this session.")
    return True, f"{_FIXTURE_SESSION}.jsonl"


REQUIREMENTS = {
    "llm": _req_llm, "piper": _req_piper, "stt": _req_stt, "gpu": _req_gpu,
    "audio_out": _req_audio_out, "audio_in": _req_audio_in,
    "quiet_room": _req_quiet_room,
    "learner_wavs": _req_learner_wavs, "fixture_session": _req_fixture_session,
}

# ⚑ Re-measured immediately before the check that needs it, not once up front.
#
# A model file present at 08:47 is still present at 08:49. A QUIET ROOM IS NOT
# LIKE THAT. Observed: the up-front probe passed, three other checks ran for two
# minutes, and by the time handsfree_halfduplex started the room was loud —
# 3.98 s voiced in its 12.67 s capture, peak frame 0.069 against a 0.004 gate.
# The check failed for a reason that was not the code, which is exactly what the
# prerequisite exists to prevent.
#
# The fix is timing, NOT a looser threshold: re-probe just before the check, so
# a noisy room produces an honest SKIP instead of a spurious FAIL.
JUST_IN_TIME = {"quiet_room"}


# ═══════════════════════════════════════════════════════════════ the gate
@dataclass
class Check:
    name: str
    script: str
    timeout_s: float
    needs: tuple[str, ...]
    covers: str
    harness: str | None = None          # value for LENA_HARNESS
    result: dict = field(default_factory=dict)

    @property
    def hardware(self) -> str:
        """What this check needs from the machine and the room, not from disk."""
        hw = [n for n in self.needs
              if n in ("gpu", "audio_out", "audio_in", "quiet_room")]
        return ", ".join(hw) if hw else "none"


CHECKS: list[Check] = [
    Check("invariants", "lab/regression/check_invariants.py", 120, (),
          "frozen config, the BUILT conversation prompt, shutdown ordering, "
          "telemetry honesty — the hand-checklist from the handoff"),

    Check("smoke", "app/_smoke.py", 420, ("gpu", "llm", "piper", "stt"),
          "core loop end to end: streaming LLM -> sanitizer -> chunking -> "
          "Piper, plus telemetry honesty (unmeasured stages stay null) and "
          "0 stalls",
          harness="regress_smoke"),

    Check("tts_continuity", "lab/experiments/e04_tts_continuity/verify_engine.py",
          420, ("gpu", "llm", "piper", "stt", "audio_out"),
          "E4: 0 playback underflows on the REAL device through the real "
          "engine — the chopping bug's regression guard",
          harness="regress_tts_continuity"),

    Check("handsfree_halfduplex", "lab/experiments/e05_handsfree/verify_handsfree.py",
          420, ("gpu", "llm", "piper", "stt", "audio_out", "audio_in", "quiet_room"),
          "E5 half-duplex invariants: mic never open while Lena speaks, silence "
          "invents no turn, echo invents no turn, manual PTT still works",
          harness="regress_handsfree_halfduplex"),

    Check("handsfree_cycle", "lab/experiments/e05_handsfree/verify_cycle.py",
          600, ("gpu", "llm", "piper", "stt", "audio_out", "learner_wavs"),
          "E5 lifecycle: the mic re-arms on EVERY turn and the endpointer ends "
          "every turn, driven by the user's own recorded German",
          harness="regress_handsfree_cycle"),

    Check("evaluation_degradation", "lab/experiments/e10_evaluation/verify_degradation.py",
          600, ("gpu", "llm"),
          "E10: with the LLM dead, measured statistics survive, the failure is "
          "classified llm_unavailable, no exception name reaches the user and "
          "the retry is bounded — then the same session evaluates when alive",
          harness="regress_evaluation_degradation"),

    Check("shutdown_evaluation", "lab/experiments/e10_evaluation/verify_shutdown_path.py",
          600, ("gpu", "llm", "piper", "stt"),
          "E10: the real server shutdown path produces, persists and broadcasts "
          "the evaluation before the LLM is allowed to stop (the StallError race)",
          harness="regress_shutdown_evaluation"),

    Check("startup_preflight", "lab/experiments/e14_startup/verify_preflight.py",
          300, (),
          "E14: every realistic startup prerequisite broken in isolation -- each "
          "is detected, classified HARD/WARN/INFO, and answered with a German "
          "sentence carrying a remedy and no traceback"),

    Check("launch_path", "lab/experiments/e14_startup/verify_launch.py",
          420, ("gpu", "llm", "piper", "stt"),
          "E14: `python -m app.server` -- the path start.cmd actually runs -- "
          "boots, ends through /api/end and exits 0, with no warnings on a "
          "healthy machine. The only check that exercises server.main().",
          harness="e14_verify_launch"),
]

# Deliberately NOT in the gate — recorded so the omissions are decisions, not
# oversights. See lab/regression/README.md for the reasoning.
EXCLUDED = {
    "lab/experiments/e07_mistakes/verify_mistakes.py":
        "returns 0 unconditionally; it is a precision STUDY for a human to read, "
        "not a pass/fail invariant, and the panel is PROVISIONAL pending E2",
    "lab/experiments/e08_stt_forensics/verify_trim_and_decode.py":
        "measurement, not a gate: it reports DISAGREEMENT between decode "
        "configurations with no ground truth, and its trim comparison is moot "
        "now that SPEECH_TRIM_ENABLED is False (asserted by `invariants`)",
    "lab/experiments/e04_tts_continuity/measure_gaps.py":
        "isolated-path measurement, superseded in the gate by verify_engine.py "
        "which proves the same property in production code",
    "lab/experiments/e06_noise/noise_validation.py":
        "hardware characterisation of one microphone in one room; the result is "
        "a calibration, not an invariant a code change can break",
    "lab/experiments/e00_stack/*, e01_llm/*":
        "settled model-selection experiments; re-running costs GPU hours and "
        "produces no new information",
    "lab/experiments/e11_conversation/probe_ab.py":
        "A/B measurement of two prompts; its output is a number to interpret, "
        "not a threshold. The prompt property it protects is asserted cheaply "
        "by `invariants` instead",
}


# ═══════════════════════════════════════════════════════════ port hygiene
def _port_busy_pid(port: int) -> int | None:
    """PID listening on `port`, or None. Used to reclaim a leaked llama-server."""
    try:
        import psutil
    except Exception:
        return None
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN:
                return c.pid
    except Exception:
        pass
    return None


def _port_free(port: int) -> bool:
    import socket
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        return False
    except Exception:
        return True
    finally:
        s.close()


# ⚑ TWO GATE RUNS MUST NOT OVERLAP.
#
# Learned the hard way: a background series that was believed dead was still
# looping while a second series started. Both spawn llama-server on port 8790,
# so checks failed with "port occupied before start" in ways that looked like
# product regressions and cost a full diagnosis to rule out. The port reclaim
# cannot fix this -- it frees the port, and the other run immediately takes it
# again.
#
# The lock is advisory and self-healing: a stale file from a killed run is
# ignored once its PID is gone.
LOCK = RESULTS_DIR / ".lock"


def _acquire_lock() -> tuple[bool, str]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        prev = json.loads(LOCK.read_text(encoding="utf-8"))
    except Exception:
        prev = None
    if prev and prev.get("pid"):
        try:
            import psutil
            if psutil.pid_exists(prev["pid"]):
                p = psutil.Process(prev["pid"])
                if "python" in (p.name() or "").lower():
                    return False, (f"another regression run is active "
                                   f"(pid {prev['pid']}, started {prev.get('at')}). "
                                   f"Two runs share port {config.LLM_PORT} and will "
                                   f"fail each other's checks. Wait for it, or stop "
                                   f"it, then re-run.")
        except Exception:
            pass                      # stale lock, or psutil missing -> proceed
    try:
        LOCK.write_text(json.dumps({"pid": os.getpid(),
                                    "at": dt.datetime.now().isoformat(timespec="seconds")}),
                        encoding="utf-8")
    except Exception:
        pass
    return True, ""


def _release_lock():
    try:
        if LOCK.exists() and json.loads(LOCK.read_text(encoding="utf-8")).get("pid") == os.getpid():
            LOCK.unlink()
    except Exception:
        pass


def _kill_stray_llama() -> str:
    """Kill llama-server processes belonging to THIS project, between checks.

    The port probe alone is not enough: a llama-server that is still starting up
    has not bound the port yet, so the probe says "free" and the next check then
    loses the race to it. Between checks there is no legitimate llama-server, so
    any that exists is a leak.

    ⚑ Scoped by executable path to `lab/bin`. A llama-server the user is running
    from somewhere else is none of this runner's business.
    """
    try:
        import psutil
    except Exception:
        return ""
    ours = str((ROOT / "lab" / "bin").resolve()).lower()
    killed = []
    for p in psutil.process_iter(["pid", "name", "exe"]):
        try:
            if "llama-server" not in (p.info.get("name") or "").lower():
                continue
            exe = (p.info.get("exe") or "").lower()
            if exe and not exe.startswith(ours):
                continue              # not ours -- leave it alone
            p.kill()
            killed.append(p.info["pid"])
        except Exception:
            pass
    if killed:
        try:
            psutil.wait_procs([psutil.Process(pid) for pid in killed
                               if psutil.pid_exists(pid)], timeout=10)
        except Exception:
            pass
        return f"killed leaked llama-server pid(s) {killed}"
    return ""


def _reclaim_port(port: int, wait_s: float = 25.0) -> tuple[bool, str]:
    """Wait for the port, then kill whatever still holds it.

    LlamaCppProvider.start() refuses to start on an occupied port. A check that
    died before its teardown leaves llama-server.exe running, and every later
    check would then fail for a reason that has nothing to do with the code
    under test. Reclaiming is loud on purpose.
    """
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if _port_free(port):
            return True, ""
        time.sleep(0.5)
    pid = _port_busy_pid(port)
    if pid is None:
        return False, f"port {port} still occupied and the owner PID is unknown"
    try:
        import psutil
        p = psutil.Process(pid)
        name = p.name()
        p.kill()
        p.wait(timeout=10)
    except Exception as e:
        return False, f"port {port} held by pid {pid}, kill failed: {e}"
    ok = _port_free(port)
    return ok, (f"reclaimed port {port} from leaked {name} (pid {pid})" if ok
                else f"killed pid {pid} but port {port} is still busy")


# ═══════════════════════════════════════════════════════════════ execution
def _run_check(c: Check, log_path: Path, stream: bool) -> dict:
    env = dict(os.environ)
    # German text and box-drawing characters go through these pipes; the
    # Windows console default codepage would raise UnicodeEncodeError inside
    # the child and be misreported as a product failure.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if c.harness:
        env["LENA_HARNESS"] = c.harness
    else:
        env.pop("LENA_HARNESS", None)

    cmd = [sys.executable, "-X", "utf8", str(ROOT / c.script)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", bufsize=1)

    q: queue.Queue = queue.Queue()

    def pump():
        try:
            for line in proc.stdout:            # type: ignore[union-attr]
                q.put(line.rstrip("\n"))
        except Exception as e:
            q.put(f"[runner] output pump died: {type(e).__name__}: {e}")
        finally:
            q.put(None)

    th = threading.Thread(target=pump, daemon=True)
    th.start()

    lines: list[str] = []
    deadline = t0 + c.timeout_s
    timed_out = False
    while True:
        try:
            line = q.get(timeout=0.5)
        except queue.Empty:
            if time.perf_counter() > deadline:
                timed_out = True
                break
            continue
        if line is None:
            break
        lines.append(line)
        if stream:
            print(f"    | {line}", flush=True)

    if timed_out:
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        rc = None
    else:
        rc = proc.wait()

    wall = time.perf_counter() - t0
    log_path.write_text(
        f"# {c.name}\n# {' '.join(cmd)}\n# exit={rc} wall={wall:.1f}s\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")

    if timed_out:
        status = "TIMEOUT"
    elif rc == 0:
        status = "PASS"
    elif rc == 2:
        # ⚑ EXIT 2 = INCONCLUSIVE, not failure. A check may discover that its
        # own preconditions were violated while it ran -- e05 half-duplex needs
        # a silent room and the room is not under our control. Reporting that as
        # a pass would be a lie; reporting it as a failure would blame the code.
        # It is a skip, so the run is INCOMPLETE and nobody reads it as green.
        status = "SKIP"
    else:
        status = "FAIL"
    return {"status": status, "exit_code": rc, "wall_s": round(wall, 1),
            "log": str(log_path), "n_lines": len(lines),
            "tail": lines[-25:]}


# ═══════════════════════════════════════════════════════════════════ main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Lena regression gate — one PASS/FAIL over the established checks.")
    ap.add_argument("--only", help="comma-separated check names (see --list)")
    ap.add_argument("--skip", help="comma-separated check names to leave out; "
                                   "the run is reported INCOMPLETE, never PASS")
    ap.add_argument("--list", action="store_true", help="list the gate and exit")
    ap.add_argument("--quiet", action="store_true",
                    help="do not stream child output (logs are still written)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first failing check")
    args = ap.parse_args()

    if args.list:
        print("\n  REGRESSION GATE\n")
        for c in CHECKS:
            print(f"  {c.name:<24} {c.script}")
            print(f"  {'':<24} hardware: {c.hardware}   timeout: {c.timeout_s:.0f}s")
            print(f"  {'':<24} {c.covers}\n")
        print("  NOT in the gate:\n")
        for k, v in EXCLUDED.items():
            print(f"  - {k}\n      {v}\n")
        return 0

    selected = list(CHECKS)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = want - {c.name for c in CHECKS}
        if unknown:
            print(f"unknown check(s): {', '.join(sorted(unknown))}")
            return 1
        selected = [c for c in CHECKS if c.name in want]
    dropped = set()
    if args.skip:
        dropped = {s.strip() for s in args.skip.split(",") if s.strip()}
        selected = [c for c in selected if c.name not in dropped]

    got_lock, why = _acquire_lock()
    if not got_lock:
        print("\n  REGRESSION: ABORTED\n")
        print(f"  {why}\n")
        return 1

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  LENA REGRESSION GATE")
    print(f"  run {run_id}   python {sys.version.split()[0]}   {sys.executable}")
    print(f"  logs: {out_dir}")
    print("=" * 78)

    # ---- prerequisites, once, up front -----------------------------------
    needed = sorted({n for c in selected for n in c.needs} - JUST_IN_TIME)
    prereq: dict[str, tuple[bool, str]] = {}
    print("\n  PREREQUISITES")
    for name in needed:
        ok, detail = REQUIREMENTS[name]()
        prereq[name] = (ok, detail)
        print(f"    {'ok ' if ok else 'MISSING'}  {name:<16} {detail}")
    deferred = sorted({n for c in selected for n in c.needs} & JUST_IN_TIME)
    for name in deferred:
        print(f"    ...      {name:<16} measured just before the check that needs it")
    if not needed and not deferred:
        print("    (none)")

    free, note = _reclaim_port(config.LLM_PORT, wait_s=2.0)
    if note:
        print(f"    {'ok ' if free else 'PROBLEM'}  port {config.LLM_PORT:<11} {note}")
    if not free:
        print(f"\n  ABORT: llama-server port {config.LLM_PORT} is occupied and could "
              f"not be reclaimed.\n  Nothing that boots an LLM can run. "
              f"Stop the process holding it and re-run.")
        _release_lock()
        return 1

    # ---- run --------------------------------------------------------------
    t_start = time.perf_counter()
    for c in selected:
        missing = [f"{n}: {prereq[n][1]}" for n in c.needs
                   if n in prereq and not prereq[n][0]]
        header = f"\n{'-' * 78}\n  {c.name}   [hardware: {c.hardware}]\n  {c.covers}\n{'-' * 78}"
        print(header)
        # Re-measure the volatile prerequisites NOW, not two minutes ago.
        if not missing:
            for n in c.needs:
                if n in JUST_IN_TIME:
                    ok_jit, detail_jit = REQUIREMENTS[n]()
                    print(f"  [{n}] {detail_jit}")
                    if not ok_jit:
                        missing.append(f"{n}: {detail_jit}")
        if missing:
            c.result = {"status": "SKIP", "reason": missing, "wall_s": 0.0}
            print("  SKIP — prerequisite not met:")
            for m in missing:
                print(f"    - {m}")
            continue

        # Kill leaked servers FIRST, then wait for the port. Doing it in the
        # other order loses to a server that has not bound yet.
        stray = _kill_stray_llama()
        if stray:
            print(f"  [port] {stray}")
        ok, note = _reclaim_port(config.LLM_PORT, wait_s=25.0)
        if note:
            print(f"  [port] {note}")
        if not ok:
            c.result = {"status": "FAIL", "reason": [note], "wall_s": 0.0,
                        "exit_code": None, "tail": [note]}
            print(f"  FAIL — {note}")
            continue

        c.result = _run_check(c, out_dir / f"{c.name}.log", stream=not args.quiet)
        print(f"  => {c.result['status']}  ({c.result['wall_s']}s, "
              f"exit={c.result['exit_code']}, log={Path(c.result['log']).name})")
        if args.quiet and c.result["status"] != "PASS":
            print("  ---- last lines ----")
            for line in c.result.get("tail", []):
                print(f"    | {line}")

        # reclaim before the next check rather than after the last one, so a
        # leaked server is attributed to the check that leaked it
        ok, note = _reclaim_port(config.LLM_PORT, wait_s=25.0)
        if note:
            print(f"  [port] {note}")

        if args.fail_fast and c.result["status"] != "PASS":
            print("\n  --fail-fast: stopping here")
            break

    total_wall = time.perf_counter() - t_start

    # ---- verdict ----------------------------------------------------------
    ran = [c for c in selected if c.result]
    not_run = [c for c in selected if not c.result]
    failed = [c for c in ran if c.result["status"] in ("FAIL", "TIMEOUT")]
    skipped = [c for c in ran if c.result["status"] == "SKIP"]
    passed = [c for c in ran if c.result["status"] == "PASS"]

    print("\n" + "=" * 78)
    print("  RESULT")
    print("=" * 78)
    for c in selected:
        st = c.result.get("status", "NOT RUN") if c.result else "NOT RUN"
        wall = c.result.get("wall_s", 0.0) if c.result else 0.0
        print(f"  {st:<8} {c.name:<24} {wall:>7.1f}s   [{c.hardware}]")
        for r in (c.result.get("reason") or []) if c.result else []:
            print(f"           {r}")

    excluded_by_flag = sorted(dropped | {c.name for c in CHECKS if c not in selected})
    if excluded_by_flag:
        print(f"\n  not selected: {', '.join(excluded_by_flag)}")

    if failed or not_run:
        verdict, code = "FAIL", 1
    elif skipped or excluded_by_flag:
        verdict, code = "INCOMPLETE", 2
    else:
        verdict, code = "PASS", 0

    print(f"\n  {len(passed)}/{len(selected)} checks passed   "
          f"{len(failed)} failed   {len(skipped)} skipped   "
          f"total {total_wall / 60:.1f} min")
    if verdict == "INCOMPLETE":
        print("  INCOMPLETE means the gate did not fully run. It is NOT a pass.")
    print(f"\n  REGRESSION: {verdict}\n")

    report = {
        "run_id": run_id, "at": dt.datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict, "exit_code": code,
        "total_wall_s": round(total_wall, 1),
        "python": sys.version.split()[0], "executable": sys.executable,
        "prerequisites": {k: {"ok": v[0], "detail": v[1]} for k, v in prereq.items()},
        "checks": [{"name": c.name, "script": c.script, "hardware": c.hardware,
                    "covers": c.covers, **(c.result or {"status": "NOT RUN"})}
                   for c in selected],
        "not_selected": excluded_by_flag,
        "excluded_by_design": EXCLUDED,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    latest = RESULTS_DIR / "latest.json"
    latest.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"  report: {out_dir / 'report.json'}")
    _release_lock()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

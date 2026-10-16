# Interview Desk & Jonas

A local, offline voice app for practising German: job-interview rounds with an assessment report, and conversation coaching with a coach persona. Speech recognition (faster-whisper), the language model (Gemma via llama.cpp) and speech synthesis (Piper) all run on the learner's Windows PC; the UI is a local web page in a desktop shell.

## Status

Personal local application. The model choice is recorded in `app/config.py` as **provisional**: it was not validated by blind rating or by a native speaker, and the known vocabulary-error measurements are listed there. No hosted service exists.

## What is here

| Path | Contents |

|---|---|

| `app/` | Local server (`server.py`, bound to 127.0.0.1 with Host/Origin checks), conversation engine, interview flow and report, assessment, providers for STT/LLM/TTS, personas (`coach.json`, `persona.json`, `teacher.json`), settings and static UI. |

| `desktop/` | Desktop shell, first-run setup flow, installer build script and icon. |

| `tests/` | Synthetic tests for interview selection/assessment and the local-server guard. |

| `setup.cmd`, `start.cmd`, `regress.cmd`, `build_desktop.cmd` | Windows entry points. |

## Not included

- Models and binaries (Gemma GGUF, Whisper, Piper voices, llama.cpp build) — about 6 GB.

- Provisioned `lab/models`, `lab/bin`, `lab/.venv`, setup state and private reports. The source in `lab/setup/` (bootstrap, verification, pinned artifact manifest and `requirements.lock`) and `lab/regression/` is included. Setup requires provisioning the omitted runtime dependencies; this publication did not download or run them.

- Learner data: sessions, recordings, settings and evaluation history (stored under `%LOCALAPPDATA%\GermanPartner` at runtime).

## Running

Requires Windows, Python 3.10, an NVIDIA GPU for the LLM, and the models above placed under `lab/models` (or `LENA_ASSET_ROOT`). Python packages used by the source include numpy, sounddevice, soundfile, faster-whisper, piper, psutil, pynvml, reportlab and pywebview.

## Verification

```

python -B -m pytest -p no:cacheprovider tests/test_local_server_guard.py tests/test_interview_assessment.py
# 12 synthetic tests passed; this is not the full model/voice suite

```

These tests use synthetic requests only; they start no microphone, model or GPU.

## License

No license file is included; no open-source license is granted. Model licences are those of their upstream projects.

## Portfolio

https://abdelrahman-hesham-portfolio.vercel.app


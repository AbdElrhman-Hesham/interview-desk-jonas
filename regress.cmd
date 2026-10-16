@echo off
REM ===========================================================================
REM  Lena - regression gate
REM
REM  One command, one PASS/FAIL over the established checks:
REM    invariants . smoke . TTS continuity . hands-free half-duplex .
REM    hands-free lifecycle . evaluation degradation . shutdown evaluation
REM
REM  Takes several minutes: each check boots its own llama-server, Whisper and
REM  Piper on purpose, because that isolation is what makes the result mean
REM  something.
REM
REM  Speakers and a microphone must be available - some checks exercise the
REM  real audio devices and are SKIPPED (never silently passed) without them.
REM  Nobody has to speak; no clicking; nothing to watch.
REM
REM    regress.cmd                       run the whole gate
REM    regress.cmd --list                show the gate and what is excluded
REM    regress.cmd --only invariants     run one check (see --list for names)
REM    regress.cmd --quiet               only show output of failing checks
REM    regress.cmd --fail-fast           stop at the first failure
REM
REM  EXIT  0 = PASS   1 = FAIL   2 = INCOMPLETE (a check was skipped)
REM ===========================================================================
setlocal
cd /d "%~dp0"
title Lena - Regression

set "PY=lab\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo.
  echo   Die Python-Umgebung fehlt:  %PY%
  echo   Erwartet wird ein venv unter lab\.venv mit Python 3.10.
  echo.
  pause
  exit /b 1
)

"%PY%" -X utf8 lab\regression\run.py %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo   REGRESSION: PASS
if "%RC%"=="1" echo   REGRESSION: FAIL          - siehe lab\results\regression\
if "%RC%"=="2" echo   REGRESSION: INCOMPLETE    - ein Check wurde uebersprungen
echo.
endlocal & exit /b %RC%

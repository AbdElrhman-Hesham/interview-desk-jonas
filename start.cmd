@echo off
REM ===========================================================================
REM  Jonas - dein lokaler Gespraechspartner fuer Deutsch
REM
REM  Doppelklick genuegt. Alles laeuft lokal auf diesem Rechner.
REM  Optional:  start.cmd --level A2        Niveau setzen
REM             start.cmd --list-devices    Mikrofone anzeigen
REM             start.cmd --device 9        bestimmtes Mikrofon
REM
REM  Modell: Gemma 4 E4B QAT q4_0  (PROVISORISCH - keine finale Wahl)
REM ===========================================================================
setlocal
cd /d "%~dp0"
title Jonas - Deutsch sprechen

set "PY=lab\.venv\Scripts\python.exe"

REM ⚑ Setup is EXPLICIT. If the environment is missing, point at setup.cmd --
REM  never start provisioning from here. The learner pressed start expecting a
REM  conversation, not a 7 GiB download.
if not exist "%PY%" (
  echo.
  echo   ==================================================================
  echo    Lena ist auf diesem Rechner noch nicht eingerichtet.
  echo   ==================================================================
  echo.
  echo    Es fehlt die Python-Umgebung:  %PY%
  echo.
  echo    Fuehre einmal  setup.cmd  aus. Danach genuegt immer start.cmd.
  echo.
  pause
  exit /b 1
)

if not exist "lab\models\gemma-4-E4B_q4_0-it.gguf" (
  echo.
  echo   ==================================================================
  echo    Das Sprachmodell fehlt.
  echo   ==================================================================
  echo.
  echo    Fuehre einmal  setup.cmd  aus. Danach genuegt immer start.cmd.
  echo.
  pause
  exit /b 1
)

"%PY%" -m app.server %*
set "RC=%ERRORLEVEL%"

REM Only pause when something went wrong, so a normal end closes cleanly.
if not "%RC%"=="0" (
  echo.
  echo   Lena wurde mit Fehlercode %RC% beendet.
  echo.
  pause
)
endlocal & exit /b %RC%

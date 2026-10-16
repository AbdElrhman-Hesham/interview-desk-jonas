@echo off
REM ===========================================================================
REM  Lena - Ersteinrichtung  /  first-run setup
REM
REM  Auf einem frischen Windows-Rechner EINMAL ausfuehren. Danach immer nur
REM  noch start.cmd.
REM
REM    setup.cmd                    alles einrichten und pruefen
REM    setup.cmd --verify-only      nur pruefen, nichts herunterladen
REM    setup.cmd --preflight-only   nur die Rechner-Voraussetzungen pruefen
REM    setup.cmd --reverify         alle Dateien neu pruefsummen
REM    setup.cmd --recreate-venv    Python-Umgebung neu aufbauen
REM
REM  Download rund 7,0 GiB. Ein erneuter Lauf laedt nichts doppelt herunter.
REM  Alles bleibt lokal auf diesem Rechner.
REM ===========================================================================
setlocal
cd /d "%~dp0"
title Lena - Setup

set "VENVPY=lab\.venv\Scripts\python.exe"

REM --- verify-only runs inside the venv, which must already exist -----------
if /I "%~1"=="--verify-only" (
  if not exist "%VENVPY%" (
    echo.
    echo   Es gibt noch keine Python-Umgebung unter lab\.venv.
    echo   Fuehre zuerst  setup.cmd  ohne Optionen aus.
    echo.
    pause
    exit /b 1
  )
  "%VENVPY%" -X utf8 lab\setup\verify.py
  set "RC=%ERRORLEVEL%"
  goto :done
)

REM --- find any Python able to run the bootstrap (stdlib only) --------------
REM  The bootstrap then locates the pinned interpreter for the venv itself.
set "BOOTPY="
if exist "%VENVPY%" set "BOOTPY=%VENVPY%"

if not defined BOOTPY (
  py -3.10 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "BOOTPY=py -3.10"
)
if not defined BOOTPY (
  py -3 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "BOOTPY=py -3"
)
if not defined BOOTPY (
  python -c "import sys" >nul 2>&1
  if not errorlevel 1 set "BOOTPY=python"
)

if not defined BOOTPY (
  echo.
  echo   ==================================================================
  echo    SETUP KANN NICHT STARTEN - kein Python gefunden
  echo   ==================================================================
  echo.
  echo    Auf diesem Rechner ist kein Python installiert.
  echo.
  echo    Dieses Projekt benoetigt Python 3.10 ^(geprueft: 3.10.11^):
  echo      https://www.python.org/downloads/release/python-31011/
  echo.
  echo    Im Installer "Add python.exe to PATH" ankreuzen.
  echo    Danach dieses Fenster schliessen und setup.cmd erneut starten.
  echo.
  pause
  exit /b 1
)

%BOOTPY% -X utf8 lab\setup\bootstrap.py %*
set "RC=%ERRORLEVEL%"

:done
echo.
if "%RC%"=="0" (
  echo   Fertig. Gespraech starten mit:  start.cmd
) else (
  echo   Setup wurde mit Fehlercode %RC% beendet - siehe Meldung oben.
)
echo.
pause
endlocal & exit /b %RC%

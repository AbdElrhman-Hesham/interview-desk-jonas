@echo off
REM ===========================================================================
REM  Build the Interview Desk desktop application and its Windows installer.
REM
REM    build_desktop.cmd                 full build
REM    build_desktop.cmd --skip-runtime  reuse the staged runtime (faster)
REM    build_desktop.cmd --no-installer  stage only, do not run Inno Setup
REM
REM  Output:  desktop\dist\Interview Desk-Setup-<version>.exe
REM
REM  Every input is pinned: the embeddable CPython is checked against a
REM  SHA-256, the shell dependencies are pinned versions, and first-run assets
REM  come from lab\setup\manifest.json. Nothing is copied by hand.
REM ===========================================================================
setlocal
cd /d "%~dp0"
set "PY=lab\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo.
  echo   Die Entwicklungsumgebung fehlt: %PY%
  echo   Fuehre zuerst setup.cmd aus.
  echo.
  pause
  exit /b 1
)
"%PY%" -X utf8 desktop\build.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo   Build fehlgeschlagen mit Code %RC%.
  echo.
)
endlocal & exit /b %RC%

@echo off
REM ===========================================================================
REM  Lena - launcher for the INSTALLED application.
REM
REM  Deliberately trivial: pythonw.exe so no console window ever appears, and
REM  the shell decides everything else. Nothing here knows about models, ports
REM  or Python packages.
REM ===========================================================================
setlocal
cd /d "%~dp0.."
start "" "%~dp0..\runtime\pythonw.exe" "%~dp0shell.py" %*
endlocal

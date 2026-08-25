@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if defined BASELINE_PY (
  set "RUN_PY=%BASELINE_PY%"
) else (
  set "RUN_PY=%~dp0..\YOLO-Master\.venv\Scripts\python.exe"
)
if not exist "%RUN_PY%" (
  echo ERROR: baseline Python not found: %RUN_PY%
  echo Set BASELINE_PY to the deployed YOLO-Master environment.
  exit /b 2
)
"%RUN_PY%" -m pytest tests -q --no-header --tb=short
exit /b %ERRORLEVEL%
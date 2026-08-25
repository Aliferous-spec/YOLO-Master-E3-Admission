@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if defined BASELINE_PY (
  set "RUN_PY=%BASELINE_PY%"
) else (
  set "RUN_PY=%~dp0..\YOLO-Master\.venv\Scripts\python.exe"
)
if not exist "%RUN_PY%" (
  echo ERROR: baseline Python not found: %RUN_PY%
  echo Set BASELINE_PY to the deployed YOLO-Master environment, e.g.:
  echo   set BASELINE_PY=C:\path\YOLO-Master\.venv\Scripts\python.exe
  exit /b 2
)
echo Running E3 admission Smoke with baseline Python: %RUN_PY%
"%RUN_PY%" scripts\run_e3_smoke.py --config configs\e3_smoke.yaml
exit /b %ERRORLEVEL%
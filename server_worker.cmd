@echo off
setlocal enableextensions

cd /d "%~dp0"
set "PYTHONPATH=%CD%"
if "%PYTHONPATH:~-1%"=="\" set "PYTHONPATH=%PYTHONPATH:~0,-1%"

for %%I in ("%~dp0best_lcat_model (1).h5") do set "LCAT_MODEL_PATH=%%~fI"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run run_lcat.bat first.
  pause
  exit /b 1
)

echo LCAT model: %LCAT_MODEL_PATH%
echo Starting at http://127.0.0.1:8000/
echo.

"%~dp0.venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
endlocal

@echo off
setlocal enableextensions enabledelayedexpansion

REM One-click runner for LCAT (Windows)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.10 -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv. Install Python 3.10 from python.org then run again.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"

echo Installing requirements...
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  pause
  exit /b 1
)

for %%I in ("%~dp0best_lcat_model (1).h5") do set "LCAT_MODEL_PATH=%%~fI"

echo Freeing port 8000 if needed...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" 2>nul
ping -n 2 127.0.0.1 >nul

echo.
echo Starting LCAT server in a separate window...
echo Do NOT close the window titled "LCAT Server".
echo.
start "LCAT Server" cmd /k call "%~dp0server_worker.cmd"

echo Waiting for server (up to 90 seconds)...
set "READY=0"
for /l %%i in (1,1,90) do (
  powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/health' -TimeoutSec 2).StatusCode; exit 0 } catch { exit 1 }" >nul 2>&1
  if !errorlevel! equ 0 (
    set "READY=1"
    goto :open_app
  )
  ping -n 2 127.0.0.1 >nul
)

echo.
echo ERROR: Server did not start.
echo Look at the "LCAT Server" window for the error message.
echo.
pause
exit /b 1

:open_app
start "" "http://127.0.0.1:8000/"
echo.
echo LCAT is running.
echo   - App URL:  http://127.0.0.1:8000/
echo   - Stop it:  close the "LCAT Server" window
echo.
echo Do not use http://localhost:8000/ on some PCs - use 127.0.0.1 instead.
echo.
pause
exit /b 0

endlocal

@echo off
cd /d "%~dp0"
set "PY=python"
where python >nul 2>nul
if errorlevel 1 set "PY=C:\Users\ZhuanZ\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not exist "%PY%" (
  echo [Error] Python not found. Please install Python 3 first.
  pause
  exit /b 1
)
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8080"
"%PY%" web\server.py
pause

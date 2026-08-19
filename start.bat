@echo off
setlocal
cd /d "%~dp0"

set "PY=python"
set "PYW=pythonw"
where python >nul 2>nul
if errorlevel 1 (
  set "PY=C:\Users\ZhuanZ\.workbuddy\binaries\python\versions\3.13.12\python.exe"
  set "PYW=C:\Users\ZhuanZ\.workbuddy\binaries\python\versions\3.13.12\pythonw.exe"
)
if not exist "%PY%" (
  echo [Error] 未找到 Python，请先安装 Python 3。
  pause
  exit /b 1
)
if not exist "%PYW%" set "PYW=%PY%"

set PORT=8080
set HOST=127.0.0.1
set URL=http://%HOST%:%PORT%

REM 端口已被占用且处于 LISTENING -> 视为已运行，直接打开浏览器
netstat -ano 2>nul | findstr "%HOST%:%PORT%" | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo [提示] 服务器可能已在运行，正在打开 %URL%
  start "" "%URL%"
  pause
  exit /b 0
)

echo 正在启动服务器（后台运行，关闭此窗口不会停止它）...
echo. > server_run.log
start "" "%PYW%" web\start_server.py

REM 等待端口就绪（最多 20 秒），再开浏览器
set /a n=0
:wait
netstat -ano 2>nul | findstr "%HOST%:%PORT%" | findstr "LISTENING" >nul
if not errorlevel 1 goto open
timeout /t 1 /nobreak >nul
set /a n+=1
if %n% lss 20 goto wait

echo [Error] 服务器未能在 20 秒内启动，日志内容如下：
echo ----------------------------------------
type server_run.log
echo ----------------------------------------
pause
exit /b 1

:open
echo 服务器已就绪，正在打开浏览器...
start "" "%URL%"
echo 服务已在后台运行。关闭此窗口不会停止它。
echo 停止服务：任务管理器结束 pythonw.exe 后重启，或重启电脑。
pause

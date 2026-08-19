#!/usr/bin/env python3
"""无窗口启动器（供 start.bat 后台调用，pythonw.exe 运行）。

- 将 stdout/stderr 重定向到 server_run.log（pythonw 无控制台，print 会崩）
- 启动直连后端 server.py
- 启动失败时写入日志并弹窗提示
"""
import ctypes
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 项目根目录
WEB_DIR = Path(__file__).resolve().parent      # web 目录
LOG = ROOT / "server_run.log"

# 重定向 stdout/stderr -> 日志文件（pythonw.exe 下 sys.stdout/stderr 为 None）
# 说明：必须长期持有日志文件句柄，不能用 with 作用域
_log_fh = open(LOG, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
sys.stdout = _log_fh
sys.stderr = _log_fh


def _log(msg: str):
    _log_fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    _log_fh.flush()


def _fatal(msg: str):
    _log(msg)
    try:
        ctypes.windll.user32.MessageBoxW(0, msg, "景点讲解 AI - 启动失败", 0x10)
    except Exception:  # noqa: S110  # 无桌面会话时弹窗失败不影响启动
        pass
    os._exit(1)


def main():
    os.chdir(ROOT)
    if str(WEB_DIR) not in sys.path:
        sys.path.insert(0, str(WEB_DIR))

    try:
        import server
    except Exception:
        import traceback
        _fatal("导入 server 失败：\n" + traceback.format_exc())

    _log("启动中…")
    try:
        server.main()
    except BaseException:
        import traceback
        _fatal("服务器异常退出：\n" + traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        _fatal("启动器异常：\n" + traceback.format_exc())

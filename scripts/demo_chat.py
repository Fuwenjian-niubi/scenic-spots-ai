#!/usr/bin/env python3
"""
景点讲解 · 流式文本客户端
调用 AnythingLLM stream-chat 端点(SSE),边生成边打印,降低感知延迟。

用法:
  python demo_chat.py "介绍一下陈家祠的历史"
环境变量:
  ALLM_BASE    AnythingLLM 地址, 默认 http://localhost:3001
  ALLM_KEY     API Key(AnythingLLM 设置页 → API Keys 生成)
  ALLM_SLUG    工作区 slug, 默认 scenic-spots
"""

import json
import os
import sys

import requests

BASE = os.environ.get("ALLM_BASE", "http://localhost:3001")
API_KEY = os.environ.get("ALLM_KEY", "YOUR_API_KEY")
WS_SLUG = os.environ.get("ALLM_SLUG", "scenic-spots")


def stream_chat(message: str, session_id: str = "demo"):
    """流式对话,逐步 yield 增量文本"""
    url = f"{BASE}/api/v1/workspace/{WS_SLUG}/stream-chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"message": message, "mode": "chat", "sessionId": session_id}

    with requests.post(url, headers=headers, json=payload,
                       stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                print(f"\n[错误] {obj['error']}", file=sys.stderr)
                break
            text = obj.get("textResponse", "")
            if text:
                print(text, end="", flush=True)
                yield text
            if obj.get("close"):
                # 打印引用来源
                for s in obj.get("sources", []):
                    print(f"\n  └ 来源: {s.get('title', '')}", file=sys.stderr)
                break


def main():
    if API_KEY == "YOUR_API_KEY":
        print("请先设置 API Key: 环境变量 ALLM_KEY", file=sys.stderr)
        sys.exit(1)
    message = sys.argv[1] if len(sys.argv) > 1 else "这个景点有什么历史故事?"
    print(f"问: {message}\n答: ", end="", flush=True)
    for _ in stream_chat(message):
        pass
    print("\n")


if __name__ == "__main__":
    main()

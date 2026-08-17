#!/usr/bin/env python3
"""
景点讲解 · 实时语音客户端
AnythingLLM 流式文本 + Edge-TTS 流式合成(免费中文神经音色)。

原理: 文本边生成边按句子切分,首句到达即开始 TTS 合成并逐块播放,
实现「边说边合成」的近实时语音讲解,无需等待完整回答。

依赖: pip install edge-tts pygame requests
用法:
  python demo_voice.py "给我讲讲广州塔"
环境变量: 同 demo_chat.py (ALLM_BASE / ALLM_KEY / ALLM_SLUG)
"""

import asyncio
import os
import queue
import re
import sys
import threading

import edge_tts
from demo_chat import stream_chat

VOICE = os.environ.get("TTS_VOICE", "zh-CN-YunxiNeural")  # 男声; 女声用 XiaoxiaoNeural
OUT_FILE = "讲解音频.mp3"

# 中文断句标点
_SENT_END = re.compile(r"[。！？!?；;\n]")


def sentences_from(text: str, buf: str) -> tuple:
    """累计文本,切出完整句子,返回 (句子列表, 剩余缓冲)"""
    buf += text
    parts = _SENT_END.split(buf)
    # 最后一个元素是未闭合的缓冲
    complete = parts[:-1]
    remain = parts[-1]
    return [p.strip() for p in complete if p.strip()], remain


async def synth_and_play(sentence: str, out_file):
    """单句流式合成并逐块写出/播放"""
    audio_chunks = []
    comm = edge_tts.Communicate(sentence, VOICE)
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    data = b"".join(audio_chunks)
    with open(out_file, "ab") as f:
        f.write(data)
    return data


def _play(data: bytes):
    """可选播放(需 pygame)。无 pygame 则只落盘。"""
    try:
        import pygame
        pygame.mixer.init()
        import io
        sound = pygame.mixer.Sound(io.BytesIO(data))
        sound.play()
        while pygame.mixer.get_busy():
            pygame.time.wait(50)
    except Exception:
        pass  # 未装 pygame 时静默降级为只保存文件


async def run(message: str):
    # 清空旧文件
    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)

    q: queue.Queue[str] = queue.Queue()

    def producer():
        buf = ""
        for delta in stream_chat(message):
            done, buf = sentences_from(delta, buf)
            for s in done:
                q.put(s)
        if buf.strip():
            q.put(buf.strip())
        q.put(None)  # 结束标记

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    # 首句到达即开播,后续句子边到边合成
    while True:
        sent = await asyncio.to_thread(q.get)
        if sent is None:
            break
        print(f"\n▶ {sent}", flush=True)
        data = await synth_and_play(sent, OUT_FILE)
        await asyncio.to_thread(_play, data)

    print(f"\n[完成] 语音已保存至 {OUT_FILE}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    message = sys.argv[1] if len(sys.argv) > 1 else "给我讲讲广州塔"
    try:
        asyncio.run(run(message))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

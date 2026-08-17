#!/usr/bin/env python3
"""
端到端连通性验证 + 无 Docker 兜底实现
智谱 embedding-3(向量化) + 本地余弦检索 + DeepSeek 流式讲解

用途：
  1. 验证两个 Key 与完整 RAG 链路是否打通（AnythingLLM 就绪前的立即验证）
  2. 作为不依赖 AnythingLLM/Docker 的轻量兜底讲解引擎

零第三方依赖（仅标准库）。运行：python scripts/run_e2e.py
"""
import json
import math
import os
import re
import sys
import time
import urllib.request

# ---------- 配置（与 docker-compose.yml 保持一致） ----------
# 从环境变量读取，避免泄露真实 Key。用法：
#   set DEEPSEEK_API_KEY=xxx && set ZHIPU_API_KEY=xxx && python scripts/run_e2e.py
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ZHIPU_KEY = os.environ.get("ZHIPU_API_KEY", "")

ZHIPU_EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "sample-data", "景点示例.md")


def http_json(url, payload, key, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def embed(text: str) -> list:
    """智谱 embedding-3 → 2048 维向量"""
    d = http_json(ZHIPU_EMBED_URL, {"model": "embedding-3", "input": text},
                  ZHIPU_KEY, timeout=60)
    return d["data"][0]["embedding"]


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def parse_entries(text: str):
    """按 ## 标题拆分景点条目，返回 [(名称, 正文)]"""
    parts = re.split(r"(?m)^##\s+", text)
    entries = []
    for p in parts[1:]:
        lines = p.strip().split("\n")
        name = lines[0].strip()
        body = f"{name}\n" + "\n".join(lines[1:]).strip()
        entries.append((name, body))
    return entries


def deepseek_stream(messages):
    """DeepSeek 流式生成，逐段 yield 增量文本"""
    payload = {"model": "deepseek-chat", "messages": messages,
               "stream": True, "temperature": 0.7}
    req = urllib.request.Request(
        DEEPSEEK_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}",
                 "Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    for raw in resp:
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = obj["choices"][0].get("delta", {}).get("content", "")
        if delta:
            yield delta


def run(question: str):
    if not DEEPSEEK_KEY or not ZHIPU_KEY:
        sys.exit("缺少 API Key：请设置环境变量 DEEPSEEK_API_KEY 和 ZHIPU_API_KEY")
    with open(SAMPLE, encoding="utf-8") as f:
        text = f.read()
    entries = parse_entries(text)
    print(f"[1] 文档解析: {len(entries)} 个景点条目")

    t0 = time.time()
    vectors = [embed(body) for _, body in entries]
    print(f"[2] 向量化完成: {len(vectors)} 条 × 2048 维, 耗时 {time.time()-t0:.2f}s")

    print(f"[3] 提问: {question}")
    qv = embed(question)
    scored = sorted(((cosine(qv, v), i) for i, v in enumerate(vectors)),
                    reverse=True)
    top = scored[:2]
    print("[4] 检索 Top-2: " +
          ", ".join(f"{entries[i][0]}({s:.3f})" for s, i in top))

    context = "\n\n".join(f"【{entries[i][0]}】\n{entries[i][1]}"
                          for _, i in top)
    messages = [
        {"role": "system",
         "content": "你是专业的景点讲解员。仅根据提供的景点资料，用简洁、准确、生动的中文回答游客问题；资料中没有的信息要如实说明。"},
        {"role": "user",
         "content": f"景点资料：\n{context}\n\n游客提问：{question}"},
    ]

    print("[5] DeepSeek 流式讲解：")
    t1 = time.time()
    first = False
    full = ""
    for delta in deepseek_stream(messages):
        if not first:
            print(f"  ⏱ 首字延迟 TTFT: {time.time()-t1:.3f}s")
            first = True
        full += delta
        sys.stdout.write(delta)
        sys.stdout.flush()
    print(f"\n  ⏱ 总耗时: {time.time()-t1:.3f}s | 字数: {len(full)}")
    return full


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "广州塔的门票多少钱？"
    run(q)

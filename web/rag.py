#!/usr/bin/env python3
"""
RAG 层：嵌入 / 检索 / 语义缓存 / 文档解析 / DeepSeek 流式。
依赖 storage.py（路径）与 crypto.py（运行时密钥）；零第三方依赖。
"""
import json
import math
import pickle
import re
import threading
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

from crypto import _runtime_keys
from storage import CACHE_FILE, _md_files, _removed

# ---------- 模型配置 ----------
ZHIPU_EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
EMBED_MODEL = "embedding-3"
RESP_CACHE_SIZE = 256

SYSTEM_PROMPT = (
    "你是专业的景点讲解员。仅根据提供的景点资料，用简洁、准确、生动的中文回答游客问题；"
    "资料中没有的信息要如实说明。可结合对话上下文回答游客的追问。"
)
SYSTEM_MSG = {"role": "system", "content": SYSTEM_PROMPT}


def http_json(url, payload, key, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def embed(text: str) -> list:
    d = http_json(ZHIPU_EMBED_URL, {"model": EMBED_MODEL, "input": text},
                  _runtime_keys["zhipu"])
    return d["data"][0]["embedding"]


def parse_spots_text(text: str, fallback_name: str = ""):
    """解析 Markdown → [(名称, 正文)]：有 ## 标题按标题拆分，否则整文件作为单个条目"""
    parts = re.split(r"(?m)^##\s+", text)
    if len(parts) > 1:
        out = []
        for p in parts[1:]:
            lines = p.strip().split("\n")
            name = lines[0].strip()
            body = f"{name}\n" + "\n".join(lines[1:]).strip()
            if name and body:
                out.append((name, body))
        return out
    body = text.strip()
    return [(fallback_name or "未命名景点", body)] if body else []


# ---------- 向量缓存（支持增量，惰性初始化：须等 crypto 密钥就绪后首次 ensure） ----------
_vectors_lock = threading.Lock()
_vectors = None


def ensure_vectors():
    """首次调用时构建/加载向量缓存并返回；后续直接返回。调用方无需再单独加锁。"""
    global _vectors
    if _vectors is not None:
        return _vectors
    with _vectors_lock:
        if _vectors is None:
            _vectors = load_or_build_cache()
    return _vectors


def _persist_cache(entries=None):
    """将当前向量(或给定 entries)与 .md 文件清单写入磁盘缓存"""
    if entries is None:
        entries = list(_vectors)
    files_meta = {str(f): f.stat().st_mtime for f in _md_files()}
    CACHE_FILE.write_bytes(pickle.dumps({
        "model": EMBED_MODEL, "entries": entries, "files": files_meta}))


def load_or_build_cache() -> list:
    entries, files_meta = [], {}
    if CACHE_FILE.exists():
        try:
            cache = pickle.loads(CACHE_FILE.read_bytes())
            if cache.get("model") == EMBED_MODEL and "files" in cache:
                entries = list(cache.get("entries") or [])
                files_meta = dict(cache.get("files") or {})
                # 丢弃源文件已不存在的条目（如曾硬删除的上传文档），避免陈旧向量残留
                entries = [e for e in entries if Path(e[3]).exists()]
                print(f"[缓存] 磁盘命中, 加载 {len(entries)} 条")
        except Exception as e:
            print(f"[缓存] 加载失败: {e}, 重建中")

    md_files = _md_files()
    current = {str(f): f.stat().st_mtime for f in md_files}
    new_files = [f for f in md_files if files_meta.get(str(f)) != current[str(f)]]

    if new_files:
        raw = []  # [(名称, 正文, 源文件路径)]
        for f in new_files:
            src = str(f)
            for name, body in parse_spots_text(
                    f.read_text(encoding="utf-8", errors="ignore"), f.stem):
                raw.append((name, body, src))
        if raw:
            print(f"[嵌入] 正在向量化 {len(raw)} 个新景点条目 ...")
            t0 = time.time()
            for name, body, src in raw:
                entries.append((name, body, embed(body), src))
            print(f"[嵌入] 完成, 耗时 {time.time() - t0:.2f}s")
        for f in new_files:
            files_meta[str(f)] = current[str(f)]
        _persist_cache(entries)

    removed_spots = set(_removed().get("spots", []))
    if removed_spots:
        entries = [e for e in entries if e[0] not in removed_spots]
    return entries


# ---------- 检索 ----------
def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(question: str, top_k: int = 4, threshold: float = 0.35):
    qv = embed(question)
    snapshot = list(ensure_vectors())
    scored = sorted(((cosine(qv, v), n, b) for n, b, v, _ in snapshot),
                    key=lambda s: s[0], reverse=True)
    top = scored[:top_k]
    passed = [(n, b) for s, n, b in top if s >= threshold]
    return passed, top


def _safe_history(raw):
    """校验并裁剪前端传来的多轮历史，只保留合法 user/assistant 文本，防止注入/过大。"""
    out = []
    for h in (raw or [])[-12:]:
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        content = h.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if len(content) > 4000:
            content = content[:4000]
        out.append({"role": role, "content": content})
    return out


# ---------- 语义响应缓存 (LRU)，value 结构 {text, sources} ----------
_resp_cache = OrderedDict()
_cache_lock = threading.Lock()

# 语气助词/虚词：归一化时移除，提高「广州塔的门票」这类问法的缓存命中率。
# 注意不含「么」——「怎么/什么」里的「么」是实义语素，删除会破坏语义。
_OMIT = "的了呢吗吧啊呀哦嘛"


def _norm(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?\"'“”‘’]|[" + _OMIT + r"]",
                  "", s.lower())


def cache_get(k):
    with _cache_lock:
        if k in _resp_cache:
            _resp_cache.move_to_end(k)
            return _resp_cache[k]
    return None


def cache_put(k, text, sources=None):
    with _cache_lock:
        _resp_cache[k] = {"text": text, "sources": sources or []}
        _resp_cache.move_to_end(k)
        if len(_resp_cache) > RESP_CACHE_SIZE:
            _resp_cache.popitem(last=False)


# ---------- DeepSeek 流式 ----------
def deepseek_stream(messages):
    payload = {"model": "deepseek-chat", "messages": messages,
               "stream": True, "temperature": 0.7}
    req = urllib.request.Request(
        DEEPSEEK_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {_runtime_keys['deepseek']}",
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


def sse_event(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")

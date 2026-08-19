#!/usr/bin/env python3
"""
景点讲解 · Web 后端（直连版，零第三方依赖）
智谱 embedding-3 + 本地余弦检索 + DeepSeek 流式讲解
无需 Docker / AnythingLLM / Ollama

启动: python web/server.py    访问: http://127.0.0.1:8080
"""

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import pickle
import re
import secrets
import threading
import time
import urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


# ---------- 加载 .env（零第三方依赖，纯标准库） ----------
def load_dotenv():
    """读取项目根目录的 .env，将 KEY=VALUE 注入 os.environ（已存在的变量优先，便于 export 覆盖）。

    查找顺序：server.py 上级目录（项目根） → web 目录 → 当前工作目录。
    """
    here = Path(__file__).resolve().parent
    candidates = [here.parent / ".env", here / ".env", Path(".env")]
    env_path = next((p for p in candidates if p.is_file()), None)
    if env_path is None:
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val

load_dotenv()

# ---------- 配置 ----------
PORT = 8080
# API Key 从环境变量读取，避免泄露真实 Key。也可在网页「设置」中录入（加密存储于 .api_config.json）。
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ZHIPU_KEY = os.environ.get("ZHIPU_API_KEY", "")
ZHIPU_EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
EMBED_MODEL = "embedding-3"

SYSTEM_PROMPT = (
    "你是专业的景点讲解员。仅根据提供的景点资料，用简洁、准确、生动的中文回答游客问题；"
    "资料中没有的信息要如实说明。可结合对话上下文回答游客的追问。"
)
SYSTEM_MSG = {"role": "system", "content": SYSTEM_PROMPT}

WEB_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = WEB_DIR.parent / "sample-data"
CACHE_FILE = WEB_DIR / ".vector_cache.pkl"
RESP_CACHE_SIZE = 256

# ---------- 上传与密钥配置 ----------
UPLOAD_DIR = WEB_DIR / "uploads"
CONFIG_FILE = WEB_DIR / ".api_config.json"
SECRET_FILE = WEB_DIR / ".secret_key"
SPOT_DOCS_FILE = WEB_DIR / ".spot_docs.json"   # 文档 → 景区 映射
REMOVED_FILE = WEB_DIR / ".removed.json"       # {"spots": [...], "files": [...]} 删除/排除清单
SPOTS_FILE = WEB_DIR / ".spots.json"           # 手动创建的景点文件夹名列表
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTS = {".doc", ".docx", ".pdf", ".md",
                ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


# ---------- 密钥加密存储（仅标准库，认证加密） ----------
def _master_key() -> bytes:
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes()
    key = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(key)
    try:
        os.chmod(SECRET_FILE, 0o600)
    except OSError:
        pass
    return key


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(key, nonce + counter.to_bytes(8, "big"),
                        hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def encrypt_str(plaintext: str) -> str:
    if not plaintext:
        return ""
    master = _master_key()
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", master, salt, 120_000, dklen=64)
    enc_key, mac_key = dk[:32], dk[32:]
    data = plaintext.encode("utf-8")
    ct = bytes(a ^ b for a, b in zip(data, _keystream(enc_key, nonce, len(data))))
    tag = hmac.new(mac_key, nonce + ct, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(salt + nonce + tag + ct).decode("ascii")


def decrypt_str(token: str) -> str:
    if not token:
        return ""
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
    except Exception:
        return ""
    if len(raw) < 64:
        return ""
    salt, nonce, tag, ct = raw[:16], raw[16:32], raw[32:64], raw[64:]
    master = _master_key()
    dk = hashlib.pbkdf2_hmac("sha256", master, salt, 120_000, dklen=64)
    enc_key, mac_key = dk[:32], dk[32:]
    expect = hmac.new(mac_key, nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(expect, tag):
        return ""
    data = bytes(a ^ b for a, b in zip(ct, _keystream(enc_key, nonce, len(ct))))
    return data.decode("utf-8")


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def _save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")


def _load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            return default
    return default


def _save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


def _spot_docs() -> dict:
    return _load_json(SPOT_DOCS_FILE, {})


def _save_spot_docs(d: dict) -> None:
    _save_json(SPOT_DOCS_FILE, d)


def _removed() -> dict:
    return _load_json(REMOVED_FILE, {"spots": [], "files": []})


def _save_removed(d: dict) -> None:
    _save_json(REMOVED_FILE, d)


def _custom_spots() -> list:
    v = _load_json(SPOTS_FILE, [])
    return v if isinstance(v, list) else []


def _save_custom_spots(lst: list) -> None:
    _save_json(SPOTS_FILE, lst)


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return key[0] + "*" * (len(key) - 1)
    return key[:6] + "••••••" + key[-4:]


# 运行时密钥：用户自定义优先，否则用内置默认值
_saved_providers = set()


def _init_runtime_keys() -> dict:
    keys = {"deepseek": DEEPSEEK_KEY, "zhipu": ZHIPU_KEY}
    for provider in keys:
        enc = _load_config().get(provider)
        if enc:
            val = decrypt_str(enc)
            if val:
                keys[provider] = val
                _saved_providers.add(provider)
    return keys


_runtime_keys = _init_runtime_keys()


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


def _md_files() -> list:
    """收集知识库目录（sample-data + uploads）下所有 .md 文件，去重，过滤已删除清单"""
    removed_files = set(_removed().get("files", []))
    seen, out = set(), []
    for d in (SAMPLE_DIR, UPLOAD_DIR):
        if d.exists():
            for f in sorted(d.glob("*.md")):
                if f.name in removed_files:
                    continue
                key = str(f)
                if key not in seen:
                    seen.add(key)
                    out.append(f)
    return out


_vectors_lock = threading.Lock()


def _persist_cache(entries=None):
    """将当前向量(或给定 entries)与 .md 文件清单写入磁盘缓存"""
    if entries is None:
        entries = list(_vectors)
    files_meta = {str(f): f.stat().st_mtime for f in _md_files()}
    CACHE_FILE.write_bytes(pickle.dumps({
        "model": EMBED_MODEL, "entries": entries, "files": files_meta}))


# ---------- 启动时构建/加载向量缓存（支持增量） ----------
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


_vectors = load_or_build_cache()


# ---------- 检索 ----------
def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(question: str, top_k: int = 4, threshold: float = 0.35):
    qv = embed(question)
    with _vectors_lock:
        snapshot = list(_vectors)
    scored = sorted(((cosine(qv, v), n, b) for n, b, v, _ in snapshot), reverse=True)
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


# ---------- 语义响应缓存 (LRU) ----------
_resp_cache = OrderedDict()
_cache_lock = threading.Lock()


def _norm(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?\"'“”‘’]", "", s.lower())


def cache_get(k):
    with _cache_lock:
        if k in _resp_cache:
            _resp_cache.move_to_end(k)
            return _resp_cache[k]
    return None


def cache_put(k, v):
    with _cache_lock:
        _resp_cache[k] = v
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


# ---------- HTTP handler ----------
def sse_event(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "ScenicServer/2.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            f = WEB_DIR / "index.html"
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/spots":
            removed_spots = set(_removed().get("spots", []))
            seen, spots = set(), []

            def add(name, intro):
                if name and name not in seen and name not in removed_spots:
                    seen.add(name)
                    spots.append({"name": name, "intro": intro})

            # 1) 手动创建的景点文件夹（优先展示）
            for name in _custom_spots():
                add(name, "")
            # 2) md 解析的景点
            for e in _vectors:
                intro = ""
                for line in e[1].split("\n"):
                    if line.startswith("简介："):
                        intro = line[len("简介："):].strip()
                        break
                add(e[0], intro)
            # 3) 非 md 文档显式归属的景区
            for spot in _spot_docs().values():
                add(spot, "")
            self._json({"spots": spots})
        elif path == "/api/files":
            self._handle_files()
        elif path == "/api/config":
            self._handle_config_status()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/chat":
            self._handle_chat()
        elif path == "/api/upload":
            self._handle_upload()
        elif path == "/api/config":
            self._handle_config_save()
        elif path == "/api/spots":
            self._handle_create_spot()
        else:
            self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/spots/"):
            self._handle_delete_spot(parsed.path)
        elif parsed.path.startswith("/api/files/"):
            self._handle_delete_file(parsed.path)
        elif parsed.path == "/api/config":
            self._handle_config_clear()
        else:
            self._json({"error": "not found"}, 404)

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        message = (body.get("message") or "").strip()
        if not message:
            self._json({"error": "empty message"}, 400)
            return

        # 密钥未配置：给出引导，不进演示也不报错
        if not _runtime_keys["zhipu"] or not _runtime_keys["deepseek"]:
            self._sse_start()
            self.wfile.write(sse_event({
                "textResponse": "尚未配置 API Key，无法生成真实讲解。请到「设置 → 账号」填入 DeepSeek 与智谱 Key（详见 README）。",
                "close": True, "sources": []}))
            self.wfile.flush()
            return

        # 命中语义缓存
        k = _norm(message)
        cached = cache_get(k)
        if cached:
            self._sse_start()
            self.wfile.write(sse_event({
                "textResponse": cached, "close": False, "cached": True}))
            self.wfile.write(sse_event({"textResponse": "", "close": True}))
            self.wfile.flush()
            return

        # 检索（带相似度阈值，过滤库外问题）
        try:
            passed, _ = retrieve(message, top_k=4, threshold=0.35)
        except Exception as e:
            self._json({"error": f"检索失败: {e}"}, 502)
            return

        if not passed:
            self._sse_start()
            self.wfile.write(sse_event({
                "textResponse": "抱歉，知识库中暂未收录与此相关的信息。您可以询问该景点的门票、开放时间、交通或看点，或到「设置 → 通用」上传相关文档。",
                "close": True, "sources": []}))
            self.wfile.flush()
            return

        context = "\n\n".join(f"【{n}】\n{b}" for n, b in passed)

        # 组装多轮消息：历史 + 当前问题（上下文注入到最后一轮用户消息）
        messages = [SYSTEM_MSG] + list(_safe_history(body.get("history")))
        if messages[-1]["role"] == "user":
            messages[-1]["content"] = (
                f"景点资料：\n{context}\n\n游客提问：{messages[-1]['content']}")
        else:
            messages.append({"role": "user",
                             "content": f"景点资料：\n{context}\n\n游客提问：{message}"})

        # 流式转发
        self._sse_start()
        collected = []
        try:
            for delta in deepseek_stream(messages):
                collected.append(delta)
                self.wfile.write(sse_event({"textResponse": delta, "close": False}))
                self.wfile.flush()
        except Exception as e:
            self.wfile.write(sse_event({"error": f"生成失败：{e}", "close": True}))
            self.wfile.flush()
            if collected:
                cache_put(k, "".join(collected))
            return
        self.wfile.write(sse_event({"textResponse": "", "close": True,
                                    "sources": [{"title": n} for n, _ in passed]}))
        self.wfile.flush()

        if collected:
            cache_put(k, "".join(collected))

    # ---------- 文档上传 / 列表 / 删除 ----------
    def _handle_files(self):
        removed_files = set(_removed().get("files", []))
        spot_docs = _spot_docs()
        items = []
        for source, d in (("upload", UPLOAD_DIR), ("sample", SAMPLE_DIR)):
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if not f.is_file() or f.name in removed_files:
                    continue
                st = f.stat()
                spots = []
                if f.suffix.lower() == ".md":
                    try:
                        spots = [n for n, _ in parse_spots_text(
                            f.read_text(encoding="utf-8", errors="ignore"),
                            spot_docs.get(f.name, f.stem))]
                    except Exception:
                        spots = []
                else:
                    sp = spot_docs.get(f.name)
                    if sp:
                        spots = [sp]
                items.append({
                    "name": f.name,
                    "size": st.st_size,
                    "type": f.suffix.lower().lstrip("."),
                    "uploadedAt": st.st_mtime,
                    "source": source,
                    "spots": spots,
                })
        self._json({"files": items})

    def _handle_upload(self):
        qs = parse_qs(urlparse(self.path).query)
        name = (qs.get("name") or [""])[0].strip()
        spot = (qs.get("spot") or [""])[0].strip()
        if not name:
            self._json({"error": "缺少文件名"}, 400)
            return
        name = Path(name).name
        name = re.sub(r'[\\/:*?"<>|`\x00-\x1f]', "_", name)
        if not name:
            self._json({"error": "文件名无效"}, 400)
            return
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            self._json({"error": "不支持的文件格式，仅支持 Word(.doc/.docx)、PDF 及常见图片"}, 400)
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._json({"error": "文件内容为空"}, 400)
            return
        if length > MAX_UPLOAD_SIZE:
            self._json({"error": f"文件大小超出限制（最大 {MAX_UPLOAD_SIZE // (1024 * 1024)} MB）"}, 413)
            return
        UPLOAD_DIR.mkdir(exist_ok=True)
        stem, suffix = Path(name).stem, Path(name).suffix
        dest = UPLOAD_DIR / name
        i = 1
        while dest.exists():
            dest = UPLOAD_DIR / f"{stem}_{i}{suffix}"
            i += 1
        remaining = length
        with dest.open("wb") as out:
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
        added, embed_err = 0, None
        if ext == ".md":
            text = dest.read_text(encoding="utf-8", errors="ignore")
            if not re.search(r"(?m)^##\s+", text):
                # 无标题 md：整文件作为一个景区，记录归属名（用户指定优先）
                sd = _spot_docs()
                sd[dest.name] = spot or dest.stem
                _save_spot_docs(sd)
            try:
                added = self._add_md_to_knowledge(dest, spot_name=spot)
            except Exception as e:
                embed_err = str(e)
        elif spot:
            # 非 md 文档记录归属景区
            sd = _spot_docs()
            sd[dest.name] = spot
            _save_spot_docs(sd)
        resp = {"ok": True, "name": dest.name, "size": length, "entries": added}
        if embed_err:
            resp["warning"] = f"文件已上传，但嵌入失败（{embed_err}），重启服务后将自动重试"
        self._json(resp)

    def _add_md_to_knowledge(self, path, spot_name=""):
        """解析 .md 并动态加入向量库，返回新增条目数（无标题时用 spot_name 兜底）"""
        text = path.read_text(encoding="utf-8", errors="ignore")
        entries = parse_spots_text(text, spot_name or path.stem)
        if not entries:
            return 0
        new = [(n, b, embed(b), str(path)) for n, b in entries]
        with _vectors_lock:
            _vectors.extend(new)
            _persist_cache()
        return len(new)

    def _remove_md_from_knowledge(self, path):
        """从向量库移除某 .md 文件的全部条目，返回移除数"""
        src = str(path)
        with _vectors_lock:
            before = len(_vectors)
            _vectors[:] = [e for e in _vectors if e[3] != src]
            if len(_vectors) != before:
                _persist_cache()
            return before - len(_vectors)

    def _handle_delete_file(self, path):
        parsed = urlparse(self.path)
        name = unquote(parsed.path[len("/api/files/"):])
        name = Path(name).name
        want_source = (parse_qs(parsed.query).get("source") or [""])[0].strip()
        target = None
        in_upload = False
        for d in (UPLOAD_DIR, SAMPLE_DIR):
            if d.exists():
                cand = d / name
                if cand.is_file():
                    src_label = "upload" if d == UPLOAD_DIR else "sample"
                    # 前端带 source 时，只删对应目录的那一份（解决重名歧义）
                    if want_source and want_source != src_label:
                        continue
                    target = cand
                    in_upload = (d == UPLOAD_DIR)
                    break
        if target is None:
            self._json({"error": "文件不存在"}, 404)
            return
        is_md = target.suffix.lower() == ".md"
        hard = False
        if in_upload:
            try:
                target.unlink()
                hard = True
            except OSError as e:
                if getattr(e, "winerror", None) == 32 or "另一个程序" in str(e):
                    msg = "删除失败：文件被其他程序占用，请关闭后再试"
                else:
                    msg = f"删除失败：无权限（{e}）"
                self._json({"error": msg}, 500)
                return
        else:
            # 内置 sample 文件：加入排除清单，不删源文件（可恢复）
            rem = _removed()
            if name not in rem.get("files", []):
                rem.setdefault("files", []).append(name)
                _save_removed(rem)
        if is_md:
            self._remove_md_from_knowledge(target)
        # 清理文档归属映射
        sd = _spot_docs()
        if name in sd:
            del sd[name]
            _save_spot_docs(sd)
        self._json({"ok": True, "hard": hard})

    def _handle_delete_spot(self, path):
        name = unquote(path[len("/api/spots/"):]).strip()
        if not name:
            self._json({"error": "景点名不能为空"}, 400)
            return
        with _vectors_lock:
            before = len(_vectors)
            _vectors[:] = [e for e in _vectors if e[0] != name]
            if len(_vectors) != before:
                _persist_cache()
        rem = _removed()
        if name not in rem.get("spots", []):
            rem.setdefault("spots", []).append(name)
            _save_removed(rem)
        # 清理归属该景点的非 md 文档映射
        sd = _spot_docs()
        changed = False
        for fname, sp in list(sd.items()):
            if sp == name:
                del sd[fname]
                changed = True
        if changed:
            _save_spot_docs(sd)
        # 从手动景点文件夹列表移除
        custom = _custom_spots()
        if name in custom:
            _save_custom_spots([n for n in custom if n != name])
        self._json({"ok": True})

    def _handle_create_spot(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        name = (body.get("name") or "").strip()
        if not name:
            self._json({"error": "景点名不能为空"}, 400)
            return
        removed = _removed()
        if name in removed.get("spots", []):
            removed.setdefault("spots", []).remove(name)
            _save_removed(removed)
        custom = _custom_spots()
        if name not in custom:
            custom.append(name)
            _save_custom_spots(custom)
        self._json({"ok": True, "name": name})

    # ---------- API Key 配置 ----------
    def _key_status(self, provider):
        val = _runtime_keys[provider]
        return {"hasKey": bool(val), "masked": mask_key(val),
                "configured": provider in _saved_providers}

    def _handle_config_status(self):
        self._json({"deepseek": self._key_status("deepseek"),
                    "zhipu": self._key_status("zhipu")})

    def _handle_config_save(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        provider = body.get("provider")
        key = (body.get("key") or "").strip()
        if provider not in ("deepseek", "zhipu"):
            self._json({"error": "无效的 provider"}, 400)
            return
        if not key:
            self._json({"error": "API Key 不能为空"}, 400)
            return
        cfg = _load_config()
        cfg[provider] = encrypt_str(key)
        _save_config(cfg)
        _runtime_keys[provider] = key
        _saved_providers.add(provider)
        self._json({"ok": True, "masked": mask_key(key), "configured": True})

    def _handle_config_clear(self):
        qs = parse_qs(urlparse(self.path).query)
        provider = (qs.get("provider") or [""])[0]
        if provider not in ("deepseek", "zhipu"):
            self._json({"error": "无效的 provider"}, 400)
            return
        cfg = _load_config()
        cfg.pop(provider, None)
        _save_config(cfg)
        _runtime_keys[provider] = DEEPSEEK_KEY if provider == "deepseek" else ZHIPU_KEY
        _saved_providers.discard(provider)
        self._json({"ok": True, "masked": mask_key(_runtime_keys[provider]), "configured": False})


def main():
    ap = argparse.ArgumentParser(description="景点讲解 AI 直连版后端")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址，默认 127.0.0.1（仅本机）。局域网共享用 0.0.0.0（需自行保障网络安全）")
    ap.add_argument("--port", type=int, default=PORT, help=f"监听端口，默认 {PORT}")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"景点讲解网页(直连版)已启动: http://{args.host}:{args.port}")
    print(f"  景点: {len(_vectors)} 条 | 嵌入: {EMBED_MODEL} | 聊天: deepseek-chat")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()

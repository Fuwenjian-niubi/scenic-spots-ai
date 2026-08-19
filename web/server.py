#!/usr/bin/env python3
"""
景点讲解 · Web 后端（直连版，零第三方依赖）
分层：storage.py（路径/JSON/元数据） + crypto.py（密钥加密） + rag.py（嵌入/检索/缓存/流式）
本文件仅保留：.env 加载、HTTP 路由 Handler、启动入口。

启动: python web/server.py    访问: http://127.0.0.1:8080
"""
import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import storage
from crypto import (
    _load_config,
    _runtime_keys,
    _save_config,
    _saved_providers,
    encrypt_str,
    init_runtime_keys,
    mask_key,
)
from extract import extract_text
from ocr import OCR_MAX_IMAGE_BYTES, describe_image
from rag import (
    EMBED_MODEL,
    SYSTEM_MSG,
    _norm,
    _safe_history,
    cache_get,
    cache_put,
    deepseek_stream,
    embed,
    ensure_vectors,
    parse_spots_text,
    reembed_spot,
    retrieve,
    split_chunk,
    sse_event,
    vector_add,
    vector_remove,
    vector_snapshot,
)
from storage import (
    ALLOWED_EXTS,
    DEFAULT_CHUNKING,
    MAX_UPLOAD_SIZE,
    SAMPLE_DIR,
    UPLOAD_DIR,
    UPLOAD_TOTAL_LIMIT,
    _custom_spots,
    _removed,
    _save_custom_spots,
    _save_removed,
    _save_spot_chunking,
    _save_spot_docs,
    _spot_chunking,
    _spot_docs,
    spot_chunking_for,
    upload_total_size,
)


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

# 运行时密钥：用户自定义优先，否则用环境变量默认值（须在加载向量库前完成）
init_runtime_keys({"deepseek": DEEPSEEK_KEY, "zhipu": ZHIPU_KEY})
# 向量库就绪（含新文档增量嵌入）
ensure_vectors()


# ---------- HTTP handler ----------
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
            f = storage.WEB_DIR / "index.html"
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
            for e in vector_snapshot():
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
        elif path == "/api/spot-config":
            self._handle_spot_config_get()
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
        elif path == "/api/spot-config":
            self._handle_spot_config_save()
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

        # 命中语义缓存（带来源）
        k = _norm(message)
        cached = cache_get(k)
        if cached:
            self._sse_start()
            self.wfile.write(sse_event({
                "textResponse": cached["text"], "close": False, "cached": True}))
            self.wfile.write(sse_event({
                "textResponse": "", "close": True, "sources": cached["sources"]}))
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
                cache_put(k, "".join(collected), [])
            return
        self.wfile.write(sse_event({"textResponse": "", "close": True,
                                    "sources": [{"title": n} for n, _ in passed]}))
        self.wfile.flush()

        if collected:
            cache_put(k, "".join(collected),
                      [{"title": n} for n, _ in passed])

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
        # 总量控制：uploads 目录总空间上限
        if upload_total_size() + length > UPLOAD_TOTAL_LIMIT:
            self._json({"error": f"上传空间已满（总上限 {UPLOAD_TOTAL_LIMIT // (1024 * 1024)} MB），请先删除部分文件再上传"}, 413)
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
        entry_name = spot or dest.stem
        if ext == ".md":
            text = dest.read_text(encoding="utf-8", errors="ignore")
            if not re.search(r"(?m)^##\s+", text):
                # 无标题 md：整文件作为一个景区，记录归属名（用户指定优先）
                sd = _spot_docs()
                sd[dest.name] = entry_name
                _save_spot_docs(sd)
            try:
                added = self._add_md_to_knowledge(dest, spot_name=spot)
            except Exception as e:
                embed_err = str(e)
        elif ext in (".docx", ".pdf"):
            # 记录归属景区，再提取文本入库
            sd = _spot_docs()
            sd[dest.name] = entry_name
            _save_spot_docs(sd)
            try:
                text = extract_text(dest)
                if text.strip():
                    added = self._embed_spot_text(dest, entry_name, text)
                else:
                    embed_err = "未能从文档提取文本（可能是扫描件或加密 PDF），文件已保存但不参与检索"
            except Exception as e:
                embed_err = str(e)
        else:
            # 图片：先尝试视觉模型识别内容（提取文字+描述入知识库），失败降级为文件名标签
            sd = _spot_docs()
            sd[dest.name] = entry_name
            _save_spot_docs(sd)
            added = 0
            try:
                desc = ""
                if dest.stat().st_size <= OCR_MAX_IMAGE_BYTES:
                    desc = describe_image(dest, _runtime_keys["zhipu"]) or ""
                if desc:
                    added = self._embed_spot_text(dest, entry_name, desc)
                    if not added:
                        embed_err = "图片内容识别结果为空，已降级为文件名标签"
                else:
                    embed_err = "图片超过 8MB 或内容识别失败，已降级为文件名标签"
            except Exception as e:
                embed_err = f"图片内容识别失败（{e}），已降级为文件名标签"
            if embed_err:
                # 降级：文件名标签条目，保证「该景点有这份资料」可检索
                try:
                    label = f"【资料文件】{dest.name}（{entry_name}）"
                    added = self._embed_spot_text(dest, entry_name, label)
                except Exception:  # noqa: S110  # 降级失败不影响文件已保存
                    pass
        resp = {"ok": True, "name": dest.name, "size": length, "entries": added}
        if embed_err:
            resp["warning"] = f"文件已保存，但{embed_err}"
        self._json(resp)

    def _embed_spot_text(self, path, name, body):
        """将一段文本作为单个景点条目，按该景点切分配置分块后入库，返回条目数"""
        cfg = spot_chunking_for(name)
        new = []
        for chunk in split_chunk(body, cfg["max_chars"], cfg["overlap"]):
            new.append((name, chunk, embed(chunk), str(path)))
        vector_add(new)
        return len(new)

    def _add_md_to_knowledge(self, path, spot_name=""):
        """解析 .md 并动态加入向量库（按景点切分配置分块），返回新增条目数"""
        text = path.read_text(encoding="utf-8", errors="ignore")
        entries = parse_spots_text(text, spot_name or path.stem)
        if not entries:
            return 0
        new = []
        for n, b in entries:
            cfg = spot_chunking_for(n)
            for chunk in split_chunk(b, cfg["max_chars"], cfg["overlap"]):
                new.append((n, chunk, embed(chunk), str(path)))
        vector_add(new)
        return len(new)

    def _remove_file_from_knowledge(self, path):
        """从向量库移除某文件的全部条目，返回移除数（线程安全）"""
        return vector_remove(lambda e: e[3] == str(path))

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
        # 从向量库移除该文件的全部条目（md / docx / pdf / 图片均可能已入库）
        self._remove_file_from_knowledge(target)
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
        vector_remove(lambda e: e[0] == name)
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

    # ---------- 文档切分配置（每景点独立） ----------
    def _handle_spot_config_get(self):
        qs = parse_qs(urlparse(self.path).query)
        spot = (qs.get("spot") or [""])[0].strip()
        all_cfg = _spot_chunking()
        if spot:
            cfg = spot_chunking_for(spot)
            self._json({"spot": spot, "maxChars": cfg["max_chars"],
                        "overlap": cfg["overlap"]})
            return
        self._json({
            "configs": {k: {"maxChars": v["max_chars"], "overlap": v["overlap"]}
                        for k, v in all_cfg.items()},
            "default": {"maxChars": DEFAULT_CHUNKING["max_chars"],
                        "overlap": DEFAULT_CHUNKING["overlap"]},
        })

    def _handle_spot_config_save(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        spot = (body.get("spot") or "").strip()
        if not spot:
            self._json({"error": "缺少景点名"}, 400)
            return
        try:
            max_chars = int(body.get("maxChars") or DEFAULT_CHUNKING["max_chars"])
            overlap = int(body.get("overlap") or DEFAULT_CHUNKING["overlap"])
        except (TypeError, ValueError):
            self._json({"error": "参数必须为整数"}, 400)
            return
        max_chars = max(100, min(max_chars, 5000))
        overlap = max(0, min(overlap, max_chars // 2))
        cfg = _spot_chunking()
        if max_chars == DEFAULT_CHUNKING["max_chars"] and overlap == DEFAULT_CHUNKING["overlap"]:
            cfg.pop(spot, None)  # 与默认一致则删除记录，后续跟随全局默认
        else:
            cfg[spot] = {"max_chars": max_chars, "overlap": overlap}
        _save_spot_chunking(cfg)
        # 立即按新配置重新嵌入该景点
        try:
            n = reembed_spot(spot)
        except Exception as e:
            self._json({"ok": True, "warning": f"配置已保存，但重新嵌入失败（{e}），重启服务后将按新配置生效"})
            return
        self._json({"ok": True, "entries": n, "maxChars": max_chars, "overlap": overlap})

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
    print(f"  景点: {len(ensure_vectors())} 条 | 嵌入: {EMBED_MODEL} | 聊天: deepseek-chat")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()

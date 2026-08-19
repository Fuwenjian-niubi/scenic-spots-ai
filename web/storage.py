#!/usr/bin/env python3
"""
存储层：路径常量 / JSON 持久化 / 景点元数据 / 上传空间统计
被 crypto.py、rag.py、server.py 共享。零第三方依赖。
"""
import json
from pathlib import Path

# ---------- 路径常量 ----------
WEB_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = WEB_DIR.parent / "sample-data"
UPLOAD_DIR = WEB_DIR / "uploads"
CACHE_FILE = WEB_DIR / ".vector_cache.pkl"
CONFIG_FILE = WEB_DIR / ".api_config.json"
SECRET_FILE = WEB_DIR / ".secret_key"
SPOT_DOCS_FILE = WEB_DIR / ".spot_docs.json"   # 文档 → 景区 映射
REMOVED_FILE = WEB_DIR / ".removed.json"       # {"spots": [...], "files": [...]} 删除/排除清单
SPOTS_FILE = WEB_DIR / ".spots.json"           # 手动创建的景点文件夹名列表
CHUNKING_FILE = WEB_DIR / ".spot_chunking.json"  # 景点 → 文档切分配置 {"max_chars","overlap"}

# ---------- 上传限制 ----------
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 单文件 20 MB
UPLOAD_TOTAL_LIMIT = 500 * 1024 * 1024  # 上传目录总空间 500 MB
ALLOWED_EXTS = {".doc", ".docx", ".pdf", ".md",
                ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

# ---------- 文档切分（每景点可独立配置） ----------
DEFAULT_CHUNKING = {"max_chars": 800, "overlap": 50}


def _spot_chunking() -> dict:
    v = _load_json(CHUNKING_FILE, {})
    return v if isinstance(v, dict) else {}


def _save_spot_chunking(d: dict) -> None:
    _save_json(CHUNKING_FILE, d)


def spot_chunking_for(spot: str) -> dict:
    """某景点的生效切分配置（自定义缺失时用默认值）"""
    cfg = _spot_chunking().get(spot)
    if isinstance(cfg, dict) and cfg.get("max_chars"):
        return {
            "max_chars": max(int(cfg["max_chars"]), 100),
            "overlap": max(int(cfg.get("overlap") or 0), 0),
        }
    return dict(DEFAULT_CHUNKING)


# ---------- JSON 持久化 ----------
def _load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            return default
    return default


def _save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


# ---------- 景点元数据 ----------
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


def upload_total_size() -> int:
    """统计 uploads 目录当前占用总字节数（用于总量控制）"""
    if not UPLOAD_DIR.exists():
        return 0
    total = 0
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                continue
    return total

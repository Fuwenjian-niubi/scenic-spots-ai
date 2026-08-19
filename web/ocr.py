#!/usr/bin/env python3
"""
图片内容识别层（OCR / 图片理解）：调用智谱免费视觉模型 glm-4v-flash。

- 保持零第三方依赖：base64 编码图片 + urllib 调用 chat/completions
- 复用智谱 API Key（与 embedding-3 同一把 Key）
- 输出：图片中的文字内容 + 内容概括（适合作为检索条目）
- 调用方需处理异常与降级（图片过大 / 无视觉模型 / 网络失败 → 回退文件名标签）

限制：glm-4v-flash 对图片有大小/分辨率上限，本模块对超大文件直接跳过。
"""
import base64
from pathlib import Path

from rag import http_json

# 智谱 chat/completions 端点（与 embedding 同源，同一把 Key）
ZHIPU_CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
VISION_MODEL = "glm-4v-flash"
# 超过该字节数不做 OCR（视觉模型对超大图不稳，且 base64 请求体膨胀）
OCR_MAX_IMAGE_BYTES = 8 * 1024 * 1024

_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
}

_PROMPT = (
    "你是景点资料整理助手。请识别这张图片中的所有文字内容（如票价、开放时间、"
    "景点介绍等），并用一两句话概括图片内容。只输出识别结果本身，不要任何客套或解释。"
)


def _to_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = _MIME.get(ext, "application/octet-stream")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def describe_image(path, api_key: str, timeout: int = 60) -> str:
    """识别图片文字并生成内容描述；失败抛异常由调用方降级。"""
    p = Path(path)
    if p.stat().st_size > OCR_MAX_IMAGE_BYTES:
        return ""
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url",
                 "image_url": {"url": _to_data_url(p)}},
            ],
        }],
        "temperature": 0.2,
    }
    d = http_json(ZHIPU_CHAT_URL, payload, api_key, timeout=timeout)
    return (d["choices"][0]["message"]["content"] or "").strip()

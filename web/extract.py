#!/usr/bin/env python3
"""
文档文本提取层（纯标准库，零第三方依赖）：

- .docx: 解析 zip 内 word/document.xml 的 <w:t> 文本节点（按段落拼接）
- .pdf : 启发式解析流对象中的文本操作符 (…) Tj / [(…)…] TJ；仅对「文本型 PDF」
         有效，扫描件（无文本层）提取结果为空
- .md/.txt: 原样返回
- 图片等: 无 OCR 能力，返回空串

调用方须自行处理空文本（如扫描件 PDF 提取失败）。
"""
import re
import zipfile
from pathlib import Path

# 提取后文本的最大长度（字符），防止异常文件撑爆内存
MAX_TEXT_CHARS = 2_000_000
# PDF 文本串中汉字占比低于该值的视为噪声（二进制/字体名等），丢弃
MIN_CN_RATIO = 0.05


def extract_text(path) -> str:
    """按扩展名提取文件文本；无法提取返回空串。"""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".docx":
        return _extract_docx(p)
    if ext == ".pdf":
        return _extract_pdf(p)
    if ext in (".md", ".txt", ".markdown"):
        return p.read_text(encoding="utf-8", errors="ignore")
    return ""


# ---------- docx ----------
def _extract_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError, OSError):
        return ""
    if not xml:
        return ""
    # 按段落(</w:p>)切分，提取每段内所有 <w:t …> 文本节点
    paras = re.split(r"</w:p>", xml)
    out = []
    for p in paras:
        texts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p)
        line = "".join(texts).strip()
        if line:
            out.append(line)
    return "\n".join(out)[:MAX_TEXT_CHARS]


# ---------- pdf ----------
_PDF_ESC = {
    b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b",
    b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\",
}


def _decode_pdf_string(raw: bytes) -> str:
    """解码 PDF 括号字符串 (…) ，处理 \\n \\t \\( \\) 等转义。"""
    s = raw[1:-1]
    s = re.sub(rb"\\(.)", lambda m: _PDF_ESC.get(m.group(1), m.group(1)), s)
    return s.decode("utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if not data.startswith(b"%PDF") or len(data) < 1024:
        return ""
    texts = []
    # 遍历所有 stream…endstream，提取 Tj / TJ 文本串
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.DOTALL):
        content = m.group(1)
        for op in re.finditer(
                rb"\(\s*(?:[^()\\]|\\.)*?\s*\)\s*(?:Tj|TJ)", content):
            raw = op.group(0)
            # 取最后一个括号字符串（Tj 单个 / TJ 数组取整体拼接）
            parts = re.findall(rb"\([^()\\]*(?:\\.[^()\\]*)*\)", raw)
            texts.append("".join(_decode_pdf_string(p) for p in parts))
    text = "".join(texts)
    # 过滤：保留含中文或可读英文的内容（丢弃字体表/编码映射等二进制噪声）
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？、：；（）《》\"'\-—…%￥元]",
                  " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cn / max(len(text), 1) < MIN_CN_RATIO:
        return ""
    return text[:MAX_TEXT_CHARS]


if __name__ == "__main__":
    import sys
    for fp in sys.argv[1:]:
        t = extract_text(fp)
        print(f"== {fp} == ({len(t)} 字符)")
        print(t[:500])

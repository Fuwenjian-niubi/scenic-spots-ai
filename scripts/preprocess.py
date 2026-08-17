#!/usr/bin/env python3
"""
景点知识库文档预处理脚本
功能:
  1. 清洗文本(全角标点统一、去空白噪声、去重复段落)
  2. 按景点条目结构化(每个景点一个 ## 块 + 固定字段)
  3. 超长条目按 500~800 字符语义分块(50 字符重叠)
  4. 输出规范化 Markdown,供 AnythingLLM 直接上传索引

用法:
  python preprocess.py <输入目录或文件> [-o 输出目录] [--chunk 600] [--overlap 50]
"""

import argparse
import re
import sys
from pathlib import Path

# 固定字段(景点知识标准结构)
FIELDS = ["名称", "简介", "历史", "文化", "看点", "交通", "开放时间", "门票", "贴士"]


def clean_text(text: str) -> str:
    """统一全角标点、去空白噪声、去重复段落"""
    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 全角空格 -> 半角, 连续空白 -> 单个
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去重复段落(连续两段相同只留一段)
    lines = [ln.strip() for ln in text.split("\n")]
    dedup = []
    for ln in lines:
        if not ln or not dedup or ln != dedup[-1]:
            dedup.append(ln)
    return "\n".join(dedup).strip()


def split_entry(entry: str, chunk: int, overlap: int) -> list:
    """超长条目按语义(句子)分块,重叠拼接"""
    if len(entry) <= chunk:
        return [entry]

    # 按句号/换行切句
    sentences = re.split(r"(?<=[。！？!?])|\n", entry)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) <= chunk:
            buf += s
        else:
            if buf:
                chunks.append(buf)
            # 重叠: 保留上一块尾部 overlap 字符
            buf = buf[-overlap:] + s if overlap else s
    if buf:
        chunks.append(buf)
    return chunks


def parse_entries(raw: str) -> list:
    """按 ## 标题或【景点】标记拆分条目; ### 以上标题降级为 ##"""
    # 统一景点标题标记为 ## 名称
    raw = re.sub(r"^【([^】]+)】", r"## \1", raw, flags=re.MULTILINE)
    raw = re.sub(r"^#{2,}\s*([^\n#]+)$", r"## \1", raw, flags=re.MULTILINE)

    entries, cur = [], []
    for line in raw.split("\n"):
        if line.startswith("## "):
            if cur:
                entries.append("\n".join(cur).strip())
            cur = [line]
        elif line.startswith("# "):  # 文档级标题(# ),跳过
            continue
        else:
            cur.append(line)
    if cur:
        entries.append("\n".join(cur).strip())
    return [e for e in entries if e]


def normalize_entry(entry: str) -> str:
    """为无结构化标题的条目补上 ## 名称"""
    if not entry.startswith("## "):
        entry = "## " + entry
    return entry


def process_file(path: Path, chunk: int, overlap: int) -> str:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    entries = parse_entries(text)
    out = []
    for entry in entries:
        entry = normalize_entry(entry)
        # 标题保留,正文分块
        head, _, body = entry.partition("\n")
        parts = split_entry(body, chunk, overlap) if body else [""]
        for i, p in enumerate(parts):
            if i == 0:
                out.append(f"{head}\n{p}".strip())
            else:
                out.append(f"{head}(续)\n{p}".strip())
    return "\n\n---\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="景点知识库预处理")
    ap.add_argument("input", help="输入目录或单个文件")
    ap.add_argument("-o", "--output", default="sample-data", help="输出目录")
    ap.add_argument("--chunk", type=int, default=600, help="分块大小(字符)")
    ap.add_argument("--overlap", type=int, default=50, help="分块重叠(字符)")
    args = ap.parse_args()

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(src.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in {".txt", ".md", ".markdown"}]
    if not files:
        print("[错误] 未找到 txt/md 文件", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        result = process_file(f, args.chunk, args.overlap)
        dst = out_dir / (f.stem + ".md")
        dst.write_text(result, encoding="utf-8")
        print(f"[完成] {f.name} -> {dst} ({len(result)} 字符)")

    print(f"\n共处理 {len(files)} 个文件, 输出至 {out_dir.resolve()}")


if __name__ == "__main__":
    main()

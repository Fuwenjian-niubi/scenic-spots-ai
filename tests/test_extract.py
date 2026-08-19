"""extract.py 纯标准库文本提取单元测试"""
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))

import extract


def _in_tmpdir(func):
    """在临时目录中运行测试（沙箱不允许手动 unlink 单个文件，交给 TemporaryDirectory 清理）"""
    def wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            return func(self, Path(td))
    return wrapper


class TestDocx(unittest.TestCase):
    @staticmethod
    def _make_docx(paras):
        body = "".join(
            f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>'
            for p in paras)
        xml = ('<?xml version="1.0"?>'
               '<w:document xmlns:w="http://schemas.openxmlformats.org/'
               'wordprocessingml/2006/main"><w:body>' + body +
               '</w:body></w:document>')
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("word/document.xml", xml)
        return buf.getvalue()

    @_in_tmpdir
    def test_extract_paragraphs(self, td):
        p = td / "t.docx"
        p.write_bytes(self._make_docx(["## 广州塔", "门票：成人 150 元起。"]))
        text = extract.extract_text(p)
        self.assertIn("广州塔", text)
        self.assertIn("150", text)
        self.assertIn("\n", text)  # 段落换行保留

    @_in_tmpdir
    def test_bad_zip_returns_empty(self, td):
        p = td / "bad.docx"
        p.write_bytes(b"not a zip")
        self.assertEqual(extract.extract_text(p), "")


class TestPdf(unittest.TestCase):
    @staticmethod
    def _make_pdf(text_lines):
        content = b"\n".join(
            f"BT /F1 12 Tf 72 720 Td ({ln}) Tj ET".encode() for ln in text_lines)
        # 填充注释使文件超过 extract 的 1024 字节启发式门槛（防垃圾文件误判）
        padding = b"% " + b"x" * 1200 + b"\n"
        return (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/"
                b"Contents 4 0 R>>endobj\n"
                b"4 0 obj<</Length " + str(len(content)).encode() + b">>stream\n"
                + content + b"\nendstream\nendobj\n" + padding +
                b"trailer<</Root 1 0 R>>\n%%EOF")

    @_in_tmpdir
    def test_extract_text_pdf(self, td):
        p = td / "t.pdf"
        p.write_bytes(self._make_pdf(["广州塔门票150元", "开放时间9点半"]))
        text = extract.extract_text(p)
        self.assertIn("广州塔", text)
        self.assertIn("150", text)

    @_in_tmpdir
    def test_garbage_pdf_returns_empty(self, td):
        p = td / "g.pdf"
        p.write_bytes(b"%PDF-1.4 garbage-no-stream")
        self.assertEqual(extract.extract_text(p), "")

    @_in_tmpdir
    def test_not_pdf_returns_empty(self, td):
        p = td / "x.pdf"
        p.write_bytes(b"just some text, not a pdf")
        self.assertEqual(extract.extract_text(p), "")


class TestOther(unittest.TestCase):
    @_in_tmpdir
    def test_md_returns_as_is(self, td):
        p = td / "t.md"
        p.write_text("## 广州塔\n门票：150", encoding="utf-8")
        self.assertEqual(extract.extract_text(p), "## 广州塔\n门票：150")

    @_in_tmpdir
    def test_image_returns_empty(self, td):
        p = td / "t.png"
        p.write_bytes(b"\x89PNG fake")
        self.assertEqual(extract.extract_text(p), "")

    @_in_tmpdir
    def test_unknown_ext_returns_empty(self, td):
        p = td / "t.xyz"
        p.write_bytes(b"whatever")
        self.assertEqual(extract.extract_text(p), "")


if __name__ == "__main__":
    unittest.main()

"""preprocess.py 纯函数单元测试"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import preprocess


class TestCleanText(unittest.TestCase):
    def test_unify_newlines(self):
        self.assertEqual(preprocess.clean_text("a\r\nb\rc"), "a\nb\nc")

    def test_fullwidth_space_to_halfwidth(self):
        self.assertEqual(preprocess.clean_text("a\u3000b"), "a b")

    def test_collapse_blank_lines(self):
        self.assertEqual(preprocess.clean_text("a\n\n\n\nb"), "a\n\nb")

    def test_dedup_consecutive_lines(self):
        self.assertEqual(preprocess.clean_text("a\na\nb"), "a\nb")


class TestSplitEntry(unittest.TestCase):
    def test_short_entry_unchanged(self):
        self.assertEqual(preprocess.split_entry("hello", 100, 50), ["hello"])

    def test_long_entry_split_by_sentence(self):
        entry = "句子甲。" * 10  # 每句 4 字符
        chunks = preprocess.split_entry(entry, 4, 0)
        self.assertEqual(len(chunks), 10)

    def test_long_entry_respects_chunk_size(self):
        entry = "一二三四五六七八九十。" * 20
        chunks = preprocess.split_entry(entry, 60, 20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 60 + 20 + 1 for c in chunks))


class TestParseEntries(unittest.TestCase):
    def test_split_by_heading(self):
        raw = "## 景点A\n正文A\n## 景点B\n正文B"
        entries = preprocess.parse_entries(raw)
        self.assertEqual(len(entries), 2)
        self.assertTrue(entries[0].startswith("## 景点A"))
        self.assertTrue(entries[1].startswith("## 景点B"))

    def test_convert_chinese_bracket(self):
        raw = "【景点A】\n正文"
        entries = preprocess.parse_entries(raw)
        self.assertTrue(entries[0].startswith("## 景点A"))

    def test_skip_document_title(self):
        raw = "# 文档标题\n## 景点A\n正文"
        entries = preprocess.parse_entries(raw)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].startswith("## 景点A"))


class TestNormalizeEntry(unittest.TestCase):
    def test_add_heading_when_missing(self):
        self.assertEqual(preprocess.normalize_entry("正文"), "## 正文")

    def test_keep_existing_heading(self):
        self.assertEqual(preprocess.normalize_entry("## 标题"), "## 标题")


if __name__ == "__main__":
    unittest.main()

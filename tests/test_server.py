#!/usr/bin/env python3
"""server 核心单测：crypto 加密 / rag 检索·缓存·归一化 / storage 总量统计。
通过 mock embed() 与临时目录隔离，不依赖真实 API Key 与磁盘缓存。"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

import crypto
import rag
import storage


# ---------- crypto ----------
class TestCrypto(unittest.TestCase):
    def test_encrypt_decrypt_roundtrip(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.object(crypto, "SECRET_FILE", Path(td) / "sk"):
            token = crypto.encrypt_str("sk-abcdef123456")
            self.assertNotIn("sk-abcdef123456", token)
            self.assertEqual(crypto.decrypt_str(token), "sk-abcdef123456")

    def test_decrypt_garbage(self):
        self.assertEqual(crypto.decrypt_str(""), "")
        self.assertEqual(crypto.decrypt_str("not-a-token"), "")
        self.assertEqual(crypto.decrypt_str("aGVsbG8="), "")

    def test_mask_key(self):
        self.assertEqual(crypto.mask_key(""), "")
        self.assertEqual(crypto.mask_key("abcd"), "a***")
        self.assertEqual(crypto.mask_key("sk-1234567890abcdef"),
                         "sk-123••••••cdef")


# ---------- rag：归一化 / 历史裁剪 / 缓存 ----------
class TestRagNorm(unittest.TestCase):
    def test_norm_removes_punct_space(self):
        self.assertEqual(rag._norm("广州塔，门票？"), "广州塔门票")
        self.assertEqual(rag._norm(" 越秀公园 开放时间 "), "越秀公园开放时间")

    def test_norm_removes_modal_particles(self):
        # 增强：去掉语气助词/虚词，让不同问法命中同一缓存
        self.assertEqual(rag._norm("广州塔的门票"), "广州塔门票")
        self.assertEqual(rag._norm("门票多少钱呢"), "门票多少钱")
        self.assertEqual(rag._norm("怎么去呀"), "怎么去")

    def test_norm_case_insensitive(self):
        self.assertEqual(rag._norm("VIP ticket"), "vipticket")


class TestSafeHistory(unittest.TestCase):
    def test_keeps_valid_roles(self):
        hist = [{"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}]
        out = rag._safe_history(hist)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "user")

    def test_drops_invalid_entries(self):
        hist = [{"role": "system", "content": "x"},
                {"role": "hacker", "content": "y"},
                "not-a-dict",
                None,
                {"role": "user", "content": "ok"}]
        out = rag._safe_history(hist)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["content"], "ok")

    def test_truncates_long_content(self):
        out = rag._safe_history([{"role": "user", "content": "x" * 9999}])
        self.assertEqual(len(out[0]["content"]), 4000)

    def test_caps_at_12_turns(self):
        hist = [{"role": "user", "content": str(i)} for i in range(20)]
        out = rag._safe_history(hist)
        self.assertEqual(len(out), 12)
        self.assertEqual(out[0]["content"], "8")


class TestResponseCache(unittest.TestCase):
    def setUp(self):
        rag._resp_cache.clear()

    def tearDown(self):
        rag._resp_cache.clear()

    def test_put_get_with_sources(self):
        rag.cache_put("k1", "答案", [{"title": "广州塔"}])
        got = rag.cache_get("k1")
        self.assertEqual(got["text"], "答案")
        self.assertEqual(got["sources"], [{"title": "广州塔"}])

    def test_put_get_default_sources(self):
        rag.cache_put("k2", "答案")
        self.assertEqual(rag.cache_get("k2")["sources"], [])

    def test_miss_returns_none(self):
        self.assertIsNone(rag.cache_get("no-such-key"))

    def test_lru_eviction(self):
        for i in range(rag.RESP_CACHE_SIZE + 5):
            rag.cache_put(f"key{i}", f"v{i}")
        self.assertIsNone(rag.cache_get("key0"))
        self.assertIsNotNone(rag.cache_get(f"key{rag.RESP_CACHE_SIZE + 4}"))


# ---------- rag：检索阈值 ----------
class TestRetrieve(unittest.TestCase):
    def setUp(self):
        # 注入假向量（广州塔→[1,0,0,0] 与查询同向；陈家祠→正交）
        rag._vectors = [
            ("广州塔", "body-gz", [1.0, 0.0, 0.0, 0.0], "src1"),
            ("陈家祠", "body-cjc", [0.0, 1.0, 0.0, 0.0], "src2"),
        ]

    def tearDown(self):
        rag._vectors = None

    @patch("rag.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_passed_above_threshold(self, _mock):
        passed, top = rag.retrieve("广州塔", top_k=2, threshold=0.9)
        self.assertEqual([n for n, _ in passed], ["广州塔"])
        self.assertEqual(len(top), 2)  # 排序结果含被过滤项

    @patch("rag.embed", return_value=[0.0, 1.0, 0.0, 0.0])
    def test_low_score_filtered(self, _mock):
        passed, _ = rag.retrieve("陈家祠", top_k=2, threshold=0.99)
        # 与广州塔相似度为 0，被阈值过滤；与陈家祠相似度 1.0 保留
        self.assertEqual([n for n, _ in passed], ["陈家祠"])

    @patch("rag.embed", return_value=[1.0, 1.0, 0.0, 0.0])
    def test_cosine_value(self, _mock):
        passed, top = rag.retrieve("两个都相关", top_k=2, threshold=0.7)
        self.assertEqual([n for n, _ in passed], ["广州塔", "陈家祠"])
        # 与两个向量相似度均为 1/√2 ≈ 0.707，恰好在阈值之上
        self.assertGreaterEqual(top[0][0], 0.7)


# ---------- rag：文档解析 ----------
class TestParseSpots(unittest.TestCase):
    def test_split_by_heading(self):
        text = "## 广州塔\n简介：塔高600米\n\n## 陈家祠\n简介：宗祠建筑"
        out = rag.parse_spots_text(text)
        self.assertEqual([n for n, _ in out], ["广州塔", "陈家祠"])
        self.assertIn("塔高600米", out[0][1])

    def test_fallback_name(self):
        out = rag.parse_spots_text("没有标题的内容", fallback_name="未知景点")
        self.assertEqual(out[0][0], "未知景点")

    def test_empty(self):
        self.assertEqual(rag.parse_spots_text("  "), [])


# ---------- storage：上传总量 ----------
class TestUploadTotal(unittest.TestCase):
    def test_total_size(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "a.md").write_bytes(b"x" * 100)
            (d / "b.pdf").write_bytes(b"y" * 250)
            (d / "sub").mkdir()
            (d / "sub" / "c.md").write_bytes(b"z" * 50)  # 子目录不计入（仅顶层文件）
            with patch.object(storage, "UPLOAD_DIR", d):
                self.assertEqual(storage.upload_total_size(), 350)

    def test_missing_dir(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.object(storage, "UPLOAD_DIR", Path(td) / "nope"):
            self.assertEqual(storage.upload_total_size(), 0)


if __name__ == "__main__":
    unittest.main()

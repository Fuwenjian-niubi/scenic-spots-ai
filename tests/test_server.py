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


# ---------- rag：文本切分 ----------
class TestSplitChunk(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(rag.split_chunk("广州塔简介", 800, 50), ["广州塔简介"])

    def test_long_text_split_within_limit(self):
        text = "这是第一段内容。这是第二段内容。这是第三段内容。" * 30
        chunks = rag.split_chunk(text, 50, 10)
        self.assertGreater(len(chunks), 1)
        # max_chars 下限 100：小于 100 的入参被抬到 100
        for c in chunks:
            self.assertLessEqual(len(c), 100)
            self.assertIn(c, text)  # 每块都是原文子串（边界自然）

    def test_overlap_keeps_context(self):
        text = "段落" * 200  # 400 字无标点，强制硬切
        chunks = rag.split_chunk(text, 100, 20)
        self.assertGreater(len(chunks), 1)
        # 相邻块应有重叠：前块末尾与后块开头重复 20 字符
        self.assertEqual(chunks[0][-20:], chunks[1][:20])

    def test_no_infinite_loop(self):
        text = "x" * 500
        chunks = rag.split_chunk(text, 100, 60)  # overlap 60 收窄到 50
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertGreater(len(chunks), 1)
        # overlap 收窄为 max_chars//2=50：相邻块重叠 50 字符（有重叠，拼接≠原文属预期）
        self.assertEqual(chunks[0][-50:], chunks[1][:50])


# ---------- storage：每景点切分配置 ----------
class TestSpotChunking(unittest.TestCase):
    def setUp(self):
        self._orig = storage.CHUNKING_FILE
        self._td = tempfile.TemporaryDirectory()
        storage.CHUNKING_FILE = Path(self._td.name) / "chunking.json"

    def tearDown(self):
        storage.CHUNKING_FILE = self._orig
        self._td.cleanup()

    def test_default_when_unset(self):
        self.assertEqual(storage.spot_chunking_for("广州塔"),
                         {"max_chars": 800, "overlap": 50})

    def test_custom_config(self):
        storage._save_spot_chunking({"广州塔": {"max_chars": 300, "overlap": 30}})
        self.assertEqual(storage.spot_chunking_for("广州塔"),
                         {"max_chars": 300, "overlap": 30})
        # 其他景点仍用默认
        self.assertEqual(storage.spot_chunking_for("陈家祠")["max_chars"], 800)

    def test_floor_max_chars(self):
        storage._save_spot_chunking({"A": {"max_chars": 10, "overlap": 5}})
        self.assertEqual(storage.spot_chunking_for("A")["max_chars"], 100)


# ---------- rag：向量库原子操作（并发安全） ----------
class TestVectorOps(unittest.TestCase):
    def setUp(self):
        # 隔离缓存文件：vector_add/remove 会 _persist_cache，防止污染真实磁盘缓存
        self._orig_cache = rag.CACHE_FILE
        self._td = tempfile.TemporaryDirectory()
        rag.CACHE_FILE = Path(self._td.name) / "cache.pkl"
        rag._vectors = [
            ("广州塔", "body-1", [1.0, 0.0], "src1"),
            ("陈家祠", "body-2", [0.0, 1.0], "src2"),
        ]

    def tearDown(self):
        rag._vectors = None
        rag.CACHE_FILE = self._orig_cache
        self._td.cleanup()

    def test_add(self):
        rag.vector_add([("白云山", "body-3", [1.0, 1.0], "src3")])
        self.assertEqual(len(rag.vector_snapshot()), 3)

    def test_remove_by_name(self):
        n = rag.vector_remove(lambda e: e[0] == "广州塔")
        self.assertEqual(n, 1)
        self.assertEqual([e[0] for e in rag.vector_snapshot()], ["陈家祠"])

    def test_remove_missing_returns_zero(self):
        n = rag.vector_remove(lambda e: e[0] == "不存在")
        self.assertEqual(n, 0)
        self.assertEqual(len(rag.vector_snapshot()), 2)

    def test_snapshot_is_copy(self):
        snap = rag.vector_snapshot()
        snap.append(("花城广场", "x", [0.0, 0.0], "src"))
        self.assertEqual(len(rag.vector_snapshot()), 2)  # 原库不受影响


# ---------- rag：名称命中加权 ----------
class TestNameWeighting(unittest.TestCase):
    def setUp(self):
        # 单条目库：查询 qv 与条目余弦=0.30（低于阈值 0.40）
        rag._vectors = [
            ("广州塔", "广州塔 门票 开放时间 交通 看点", [1.0, 0.0, 0.0, 0.0], "src1"),
        ]

    def tearDown(self):
        rag._vectors = None

    @patch("rag.embed", return_value=[0.3, 0.954, 0.0, 0.0])
    def test_name_hit_crosses_threshold(self, _mock):
        # 查询含「广州塔」→ +0.15 → 0.45 ≥ 阈值 0.40，通过
        passed, top = rag.retrieve("广州塔的门票多少钱", top_k=1, threshold=0.40)
        self.assertEqual([n for n, _ in passed], ["广州塔"])
        self.assertGreater(top[0][0], 0.40)

    @patch("rag.embed", return_value=[0.3, 0.954, 0.0, 0.0])
    def test_no_name_hit_stays_rejected(self, _mock):
        # 查询不含景点名：不加权，0.30 < 0.40，拒绝
        passed, _ = rag.retrieve("一般门票多少钱", top_k=1, threshold=0.40)
        self.assertEqual(passed, [])


# ---------- rag：按配置重嵌某景点 ----------
class TestReembedSpot(unittest.TestCase):
    def test_reembed_respects_config(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.object(rag, "embed", side_effect=lambda t: [1.0, 0.0, 0.0, 0.0]), \
                patch.object(rag, "CACHE_FILE", Path(td) / "cache.pkl"), \
                patch.object(storage, "CHUNKING_FILE", Path(td) / "chunking.json"), \
                patch.object(storage, "SAMPLE_DIR", Path(td)), \
                patch.object(storage, "UPLOAD_DIR", Path(td)):
            # 一篇超长文档：默认配置(800)不分块；自定义 200 字符切分
            long_body = "## 长景点\n" + "内容。" * 300  # 约 1000+ 字
            (Path(td) / "long.md").write_text(long_body, encoding="utf-8")
            storage._save_spot_chunking({"长景点": {"max_chars": 200, "overlap": 20}})
            rag._vectors = None
            try:
                n = rag.reembed_spot("长景点")
                self.assertGreater(n, 1)  # 按 200 字符切成了多块
                vs = rag.ensure_vectors()
                self.assertEqual(len(vs), n)
                self.assertTrue(all(e[0] == "长景点" and len(e[1]) <= 200 for e in vs))
            finally:
                rag._vectors = None


if __name__ == "__main__":
    unittest.main()

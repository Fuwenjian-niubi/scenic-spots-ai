"""run_e2e.py 纯函数单元测试"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import run_e2e


class TestCosine(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(run_e2e.cosine([1, 0], [1, 0]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(run_e2e.cosine([1, 0], [0, 1]), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(run_e2e.cosine([0, 0], [1, 1]), 0.0)


class TestParseEntries(unittest.TestCase):
    def test_split_by_heading(self):
        raw = "## A\n正文A\n## B\n正文B"
        entries = run_e2e.parse_entries(raw)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0][0], "A")
        self.assertEqual(entries[1][0], "B")

    def test_body_includes_name(self):
        entries = run_e2e.parse_entries("## A\n正文")
        self.assertTrue(entries[0][1].startswith("A"))


if __name__ == "__main__":
    unittest.main()

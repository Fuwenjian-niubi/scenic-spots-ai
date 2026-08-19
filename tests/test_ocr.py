"""ocr.py 图片识别单元测试（mock 视觉 API，不真实调用）"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))

import ocr


def _tmp_img(ext=".png", size=100):
    td = tempfile.TemporaryDirectory()
    p = Path(td.name) / ("img" + ext)
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * size)
    return p, td


class TestDescribeImage(unittest.TestCase):
    @patch("ocr.http_json")
    def test_ok_returns_content(self, mock):
        mock.return_value = {
            "choices": [{"message": {"content": "门票150元，开放时间9:30"}}]}
        p, td = _tmp_img()
        try:
            out = ocr.describe_image(p, "sk-test")
        finally:
            td.cleanup()
        self.assertEqual(out, "门票150元，开放时间9:30")
        # 校验 payload：模型、data URL 前缀、文本提示
        url = mock.call_args.args[0]
        self.assertEqual(url, ocr.ZHIPU_CHAT_URL)
        payload = mock.call_args.args[1]
        self.assertEqual(payload["model"], "glm-4v-flash")
        content = payload["messages"][0]["content"]
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    @patch("ocr.http_json")
    def test_empty_content_stripped(self, mock):
        mock.return_value = {"choices": [{"message": {"content": "   "}}]}
        p, td = _tmp_img()
        try:
            self.assertEqual(ocr.describe_image(p, "sk-test"), "")
        finally:
            td.cleanup()

    @patch("ocr.http_json")
    def test_missing_choices_raises(self, mock):
        mock.return_value = {"error": {"message": "model not found"}}
        p, td = _tmp_img()
        try:
            with self.assertRaises(KeyError):
                ocr.describe_image(p, "sk-test")
        finally:
            td.cleanup()

    def test_oversize_returns_empty(self):
        p, td = _tmp_img(size=ocr.OCR_MAX_IMAGE_BYTES + 10)
        try:
            self.assertEqual(ocr.describe_image(p, "sk-test"), "")
        finally:
            td.cleanup()

    def test_mime_mapping(self):
        p, td = _tmp_img(ext=".jpg")
        try:
            url = ocr._to_data_url(p)
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main()

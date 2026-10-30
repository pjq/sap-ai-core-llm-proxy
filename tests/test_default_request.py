"""Regression tests for modern GPT token-limit normalization."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxy_server import handle_default_request


class TestDefaultRequestTokenLimits(unittest.TestCase):
    def convert(self, payload, model):
        with patch("proxy_server.load_balance_url", return_value=(
            "https://example.test/deployments/test", "SFMobileSH", "default", model
        )):
            return handle_default_request(payload, model)[1]

    def test_legacy_limit_is_translated_for_modern_gpt(self):
        for model in ("gpt-5.6-sol", "gpt-6-luna", "gpt-6.1-sol",
                      "gpt-6.1-sol-2026-09-29"):
            with self.subTest(model=model):
                payload = {"messages": [], "max_tokens": 256}
                out = self.convert(payload, model)
                self.assertEqual(out["max_completion_tokens"], 256)
                self.assertNotIn("max_tokens", out)
                self.assertEqual(payload, {"messages": [], "max_tokens": 256})

    def test_explicit_completion_limit_takes_precedence(self):
        for model in ("gpt-5.6-luna", "gpt-6-luna", "gpt-6.1-sol"):
            with self.subTest(model=model):
                out = self.convert({"max_tokens": 256, "max_completion_tokens": 512}, model)
                self.assertEqual(out["max_completion_tokens"], 512)
                self.assertNotIn("max_tokens", out)

    def test_completion_limit_is_preserved(self):
        self.assertEqual(self.convert({"max_completion_tokens": 256}, "gpt-6.1-sol"),
                         {"max_completion_tokens": 256})

    def test_missing_limit_is_not_invented(self):
        self.assertEqual(self.convert({"messages": []}, "gpt-6.1-sol"), {"messages": []})

    def test_older_models_keep_legacy_limit(self):
        for model in ("gpt-4o", "gpt-5.5"):
            with self.subTest(model=model):
                self.assertEqual(self.convert({"max_tokens": 256}, model), {"max_tokens": 256})


if __name__ == "__main__":
    unittest.main()

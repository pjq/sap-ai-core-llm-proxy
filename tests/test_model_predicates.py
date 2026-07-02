"""Characterization tests for the model-routing predicates.

These pin the CURRENT behavior of is_claude_model / is_claude_37_or_4 /
is_gemini_model exactly as it is today, so the planned optimizations
(which include tightening these predicates) can be verified not to change
behavior silently. Cases marked "# QUIRK" capture arguably-wrong behavior
that the optimization phase is expected to fix — when it does, flip that
specific assertion in the same commit.
"""

import unittest

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from proxy_server import is_claude_model, is_claude_37_or_4, is_gemini_model


class TestIsClaudeModel(unittest.TestCase):
    def test_matches_claude_names(self):
        self.assertTrue(is_claude_model("anthropic--claude-4.6-opus"))
        self.assertTrue(is_claude_model("claude-3.5-sonnet"))
        self.assertTrue(is_claude_model("anthropic--claude-3.7-sonnet"))

    def test_matches_sonnet_names(self):
        self.assertTrue(is_claude_model("some-sonnet-model"))

    def test_matches_uppercase(self):
        self.assertTrue(is_claude_model("CLAUDE-X"))
        self.assertTrue(is_claude_model("SONNET-X"))

    def test_non_claude_models(self):
        self.assertFalse(is_claude_model("gpt-5.4"))
        self.assertFalse(is_claude_model("gpt-4o"))
        self.assertFalse(is_claude_model("gemini-2.5-pro"))


class TestIsClaude37Or4(unittest.TestCase):
    def test_real_claude_37_and_4(self):
        self.assertTrue(is_claude_37_or_4("anthropic--claude-4.6-opus"))
        self.assertTrue(is_claude_37_or_4("anthropic--claude-3.7-sonnet"))
        self.assertTrue(is_claude_37_or_4("anthropic--claude-4-sonnet"))

    def test_claude_35_is_false(self):
        self.assertFalse(is_claude_37_or_4("anthropic--claude-3.5-sonnet"))
        self.assertFalse(is_claude_37_or_4("claude-3.5"))

    def test_quirk_non_claude_names_return_true(self):
        # QUIRK: the `or "3.5" not in model` clause makes ANY string without
        # "3.5" return True — including non-Claude models and empty string.
        # Optimization phase should replace this with an explicit Claude check.
        self.assertTrue(is_claude_37_or_4("gpt-5.4"))       # QUIRK
        self.assertTrue(is_claude_37_or_4(""))               # QUIRK
        self.assertTrue(is_claude_37_or_4("gemini-2.5-pro")) # QUIRK

    def test_substring_4_triggers_true(self):
        # "4" appears in the string -> True regardless of "3.5" presence.
        self.assertTrue(is_claude_37_or_4("claude-3.5-v4"))


class TestIsGeminiModel(unittest.TestCase):
    def test_matches_gemini(self):
        self.assertTrue(is_gemini_model("gemini-2.5-pro"))
        self.assertTrue(is_gemini_model("gemini-1.5-flash"))
        self.assertTrue(is_gemini_model("GEMINI-PRO"))  # lower() applied internally

    def test_non_gemini(self):
        self.assertFalse(is_gemini_model("gpt-5.4"))
        self.assertFalse(is_gemini_model("anthropic--claude-4.6-opus"))


if __name__ == "__main__":
    unittest.main()

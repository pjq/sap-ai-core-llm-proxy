"""Characterization tests for response-format converters (backend -> client).

Covers convert_claude_to_openai, convert_claude37_to_openai,
convert_gemini_to_openai, convert_gemini_response_to_claude,
convert_openai_response_to_claude. IDs use random/time; assertions target
stable fields only.
"""

import unittest

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from proxy_server import (
    convert_claude_to_openai,
    convert_claude37_to_openai,
    convert_gemini_to_openai,
    convert_gemini_response_to_claude,
    convert_openai_response_to_claude,
)


class TestConvertClaudeToOpenAI(unittest.TestCase):
    def test_standard_claude_path(self):
        # A non-3.7/4 model name (contains "3.5") uses the standard branch.
        response = {
            "content": [{"text": "hello"}],
            "stop_reason": "end_turn",
            "role": "assistant",
            "id": "abc",
            "model": "claude-3.5",
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        out = convert_claude_to_openai(response, "anthropic--claude-3.5-sonnet")
        self.assertEqual(out["choices"][0]["message"]["content"], "hello")
        self.assertEqual(out["choices"][0]["finish_reason"], "end_turn")
        self.assertEqual(out["usage"]["prompt_tokens"], 3)
        self.assertEqual(out["usage"]["completion_tokens"], 5)
        self.assertEqual(out["usage"]["total_tokens"], 8)

    def test_delegates_to_claude37_for_4x(self):
        # A 4.x model routes through the /converse converter, which expects
        # the output.message.content shape.
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
        }
        out = convert_claude_to_openai(response, "anthropic--claude-4.6-opus")
        self.assertEqual(out["choices"][0]["message"]["content"], "hi")
        self.assertEqual(out["usage"]["total_tokens"], 3)

    def test_malformed_returns_error_struct(self):
        out = convert_claude_to_openai({"bad": "shape"}, "anthropic--claude-3.5-sonnet")
        self.assertIn("error", out)


class TestConvertClaude37ToOpenAI(unittest.TestCase):
    def _resp(self, **over):
        base = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
        }
        base.update(over)
        return base

    def test_basic_extraction(self):
        out = convert_claude37_to_openai(self._resp(), "m")
        self.assertEqual(out["choices"][0]["message"]["content"], "hi")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["model"], "m")
        self.assertEqual(out["usage"]["total_tokens"], 30)

    def test_stop_reason_map(self):
        self.assertEqual(
            convert_claude37_to_openai(self._resp(stopReason="max_tokens"), "m")["choices"][0]["finish_reason"],
            "length",
        )
        self.assertEqual(
            convert_claude37_to_openai(self._resp(stopReason="tool_use"), "m")["choices"][0]["finish_reason"],
            "tool_calls",
        )
        self.assertEqual(
            convert_claude37_to_openai(self._resp(stopReason="unknown_xyz"), "m")["choices"][0]["finish_reason"],
            "stop",  # default
        )

    def test_first_text_block_fallback(self):
        # Block 0 is not text; converter scans for the first text block.
        resp = self._resp()
        resp["output"]["message"]["content"] = [
            {"type": "tool_use", "id": "x"},
            {"type": "text", "text": "found"},
        ]
        out = convert_claude37_to_openai(resp, "m")
        self.assertEqual(out["choices"][0]["message"]["content"], "found")

    def test_total_tokens_fallback_when_missing(self):
        resp = self._resp(usage={"inputTokens": 4, "outputTokens": 6})
        out = convert_claude37_to_openai(resp, "m")
        self.assertEqual(out["usage"]["total_tokens"], 10)

    def test_malformed_returns_error_object(self):
        out = convert_claude37_to_openai({"bad": True}, "m")
        self.assertEqual(out["object"], "error")
        self.assertEqual(out["type"], "proxy_conversion_error")


class TestConvertGeminiToOpenAI(unittest.TestCase):
    def _resp(self, **over):
        base = {
            "candidates": [
                {"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
        }
        base.update(over)
        return base

    def test_basic(self):
        out = convert_gemini_to_openai(self._resp(), "gemini-2.5-pro")
        self.assertEqual(out["choices"][0]["message"]["content"], "hi")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"]["total_tokens"], 5)
        self.assertEqual(out["model"], "gemini-2.5-pro")

    def test_safety_finish_reason_maps_to_content_filter(self):
        resp = self._resp()
        resp["candidates"][0]["finishReason"] = "SAFETY"
        out = convert_gemini_to_openai(resp, "m")
        self.assertEqual(out["choices"][0]["finish_reason"], "content_filter")

    def test_no_candidates_returns_error(self):
        out = convert_gemini_to_openai({"candidates": []}, "m")
        self.assertEqual(out["object"], "error")


class TestConvertGeminiResponseToClaude(unittest.TestCase):
    def test_basic(self):
        resp = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3},
        }
        out = convert_gemini_response_to_claude(resp, "gemini-2.5-pro")
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["content"][0]["text"], "hi")
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["usage"]["input_tokens"], 2)

    def test_error_struct(self):
        out = convert_gemini_response_to_claude({"candidates": []}, "m")
        self.assertEqual(out["type"], "error")


class TestConvertOpenAIResponseToClaude(unittest.TestCase):
    def test_text_content(self):
        resp = {
            "id": "cid",
            "model": "gpt-5.4",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        out = convert_openai_response_to_claude(resp)
        self.assertEqual(out["content"][0], {"type": "text", "text": "hello"})
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["id"], "cid")
        self.assertEqual(out["usage"]["input_tokens"], 1)

    def test_tool_calls_converted(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call_1",
                                "function": {"name": "f", "arguments": '{"a": 1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        out = convert_openai_response_to_claude(resp)
        tool_use = out["content"][0]
        self.assertEqual(tool_use["type"], "tool_use")
        self.assertEqual(tool_use["name"], "f")
        self.assertEqual(tool_use["input"], {"a": 1})
        self.assertEqual(out["stop_reason"], "tool_use")

    def test_empty_choices_returns_error(self):
        out = convert_openai_response_to_claude({"choices": []})
        self.assertEqual(out["type"], "error")


if __name__ == "__main__":
    unittest.main()

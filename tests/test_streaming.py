"""Characterization tests for streaming: chunk converters + generators.

Chunk converters are pure and cheap. The generator functions
(generate_streaming_response, generate_claude_streaming_response) read the
Flask `request` object and call `_http_session.post(...)`, so they run inside
an `app.test_request_context` with `_http_session.post` mocked to return a
fake streaming response. No real network I/O occurs.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import proxy_server
from proxy_server import (
    app,
    convert_claude_chunk_to_openai,
    convert_claude37_chunk_to_openai,
    convert_gemini_chunk_to_openai,
    convert_gemini_chunk_to_claude_delta,
    get_claude_stop_reason_from_gemini_chunk,
    convert_openai_chunk_to_claude_delta,
    get_claude_stop_reason_from_openai_chunk,
    generate_streaming_response,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class _FakeStreamResponse:
    """Mimics the context-manager streaming response from requests."""

    def __init__(self, lines=None, content_chunks=None, status=200):
        self._lines = lines or []
        self._content_chunks = content_chunks or []
        self.status_code = status
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for ln in self._lines:
            yield ln if isinstance(ln, bytes) else ln.encode("utf-8")

    def iter_content(self, chunk_size=128):
        for c in self._content_chunks:
            yield c if isinstance(c, bytes) else c.encode("utf-8")


def _drain(gen):
    """Collect a generator's output as a list of str (decoding bytes)."""
    out = []
    for item in gen:
        out.append(item.decode("utf-8") if isinstance(item, bytes) else item)
    return out


# --------------------------------------------------------------------------
# Chunk converters (pure)
# --------------------------------------------------------------------------
class TestConvertClaudeChunkToOpenAI(unittest.TestCase):
    def test_content_block_delta(self):
        chunk = 'data: {"type": "content_block_delta", "delta": {"text": "hi"}}'
        out = convert_claude_chunk_to_openai(chunk, "claude-3.5")
        self.assertTrue(out.startswith("data: "))
        payload = json.loads(out[len("data: "):].strip())
        self.assertEqual(payload["choices"][0]["delta"]["content"], "hi")

    def test_message_delta_end_turn_sets_finish(self):
        chunk = 'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}'
        payload = json.loads(convert_claude_chunk_to_openai(chunk, "claude-3.5")[6:].strip())
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_invalid_json_returns_error_sse(self):
        out = convert_claude_chunk_to_openai("data: not-json", "claude-3.5")
        self.assertIn("error", out)


class TestConvertClaude37ChunkToOpenAI(unittest.TestCase):
    def test_content_block_delta(self):
        out = convert_claude37_chunk_to_openai(
            {"contentBlockDelta": {"delta": {"text": "hi"}}}, "m"
        )
        payload = json.loads(out[6:].strip())
        self.assertEqual(payload["choices"][0]["delta"]["content"], "hi")

    def test_message_start_role(self):
        out = convert_claude37_chunk_to_openai({"messageStart": {"role": "assistant"}}, "m")
        payload = json.loads(out[6:].strip())
        self.assertEqual(payload["choices"][0]["delta"]["role"], "assistant")

    def test_message_stop_maps_finish_reason(self):
        out = convert_claude37_chunk_to_openai({"messageStop": {"stopReason": "max_tokens"}}, "m")
        payload = json.loads(out[6:].strip())
        self.assertEqual(payload["choices"][0]["finish_reason"], "length")

    def test_metadata_chunk_returns_none(self):
        self.assertIsNone(convert_claude37_chunk_to_openai({"metadata": {}}, "m"))

    def test_content_block_delta_without_text_returns_none(self):
        self.assertIsNone(
            convert_claude37_chunk_to_openai({"contentBlockDelta": {"delta": {}}}, "m")
        )

    def test_parses_json_string_input(self):
        out = convert_claude37_chunk_to_openai(
            '{"contentBlockDelta": {"delta": {"text": "x"}}}', "m"
        )
        payload = json.loads(out[6:].strip())
        self.assertEqual(payload["choices"][0]["delta"]["content"], "x")


class TestConvertGeminiChunkToOpenAI(unittest.TestCase):
    def test_text_delta(self):
        chunk = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        out = convert_gemini_chunk_to_openai(chunk, "gemini-2.5-pro")
        payload = json.loads(out[6:].strip())
        self.assertEqual(payload["choices"][0]["delta"]["content"], "hi")

    def test_finish_reason(self):
        chunk = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "x"}]}}]}
        payload = json.loads(convert_gemini_chunk_to_openai(chunk, "m")[6:].strip())
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_no_candidates_returns_none(self):
        self.assertIsNone(convert_gemini_chunk_to_openai({"candidates": []}, "m"))


class TestClaudeDeltaHelpers(unittest.TestCase):
    def test_gemini_chunk_to_claude_delta(self):
        chunk = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        out = convert_gemini_chunk_to_claude_delta(chunk)
        self.assertEqual(out["delta"]["text"], "hi")
        self.assertEqual(out["type"], "content_block_delta")

    def test_gemini_stop_reason(self):
        chunk = {"candidates": [{"finishReason": "MAX_TOKENS"}]}
        self.assertEqual(get_claude_stop_reason_from_gemini_chunk(chunk), "max_tokens")

    def test_openai_chunk_to_claude_delta(self):
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        out = convert_openai_chunk_to_claude_delta(chunk)
        self.assertEqual(out["delta"]["text"], "hi")

    def test_openai_stop_reason(self):
        chunk = {"choices": [{"finish_reason": "stop"}]}
        self.assertEqual(get_claude_stop_reason_from_openai_chunk(chunk), "end_turn")


# --------------------------------------------------------------------------
# generate_streaming_response (mocked upstream, real request context)
# --------------------------------------------------------------------------
class TestGenerateStreamingResponse(unittest.TestCase):
    def _run(self, model, fake_response, body=None):
        body = body or {"model": model, "messages": [{"role": "user", "content": "hi"}]}
        with app.test_request_context(
            "/v1/chat/completions", method="POST", json=body
        ):
            with patch.object(proxy_server._http_session, "post", return_value=fake_response):
                with patch.object(proxy_server, "log_token_usage"):
                    return _drain(
                        generate_streaming_response(
                            "https://backend/x", {}, body, model, "SUB"
                        )
                    )

    def test_claude_4x_branch_yields_converted_and_done(self):
        # Claude 3.7/4 branch uses iter_lines and ast.literal_eval on each chunk.
        lines = [
            "data: {'contentBlockDelta': {'delta': {'text': 'hi'}}}",
            "",
        ]
        out = self._run("anthropic--claude-4.6-opus", _FakeStreamResponse(lines=lines))
        self.assertTrue(any('"content": "hi"' in chunk for chunk in out))
        self.assertEqual(out[-1], "data: [DONE]\n\n")

    def test_gemini_branch_yields_converted_and_done(self):
        gemini_line = json.dumps({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
        out = self._run("gemini-2.5-pro", _FakeStreamResponse(lines=[f"data: {gemini_line}"]))
        self.assertTrue(any('"content": "hi"' in chunk for chunk in out))
        self.assertEqual(out[-1], "data: [DONE]\n\n")

    def test_gpt_branch_passthrough_and_done(self):
        # Default (GPT) branch uses iter_content and passes bytes through.
        sse = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        out = self._run("gpt-5.4", _FakeStreamResponse(content_chunks=[sse]))
        self.assertTrue(any("hi" in chunk for chunk in out))
        self.assertEqual(out[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()

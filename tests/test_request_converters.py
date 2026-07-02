"""Characterization tests for request-format converters (client -> backend).

Covers convert_openai_to_claude, convert_openai_to_claude37,
convert_claude_request_to_openai, convert_claude_request_to_gemini.
Behavior is pinned as-is; "# QUIRK" marks cases the optimization phase
may revisit.
"""

import unittest

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from proxy_server import (
    convert_openai_to_claude,
    convert_openai_to_claude37,
    convert_claude_request_to_openai,
    convert_claude_request_to_gemini,
)


class TestConvertOpenAIToClaude(unittest.TestCase):
    def test_extracts_leading_system_message(self):
        payload = {
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ]
        }
        out = convert_openai_to_claude(payload)
        self.assertEqual(out["system"], "be terse")
        self.assertEqual(out["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(out["anthropic_version"], "bedrock-2023-05-31")

    def test_defaults_applied(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out = convert_openai_to_claude(payload)
        self.assertEqual(out["system"], "")
        self.assertEqual(out["max_tokens"], 4096000)  # QUIRK: very large default
        self.assertEqual(out["temperature"], 1.0)

    def test_respects_provided_values(self):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": 0.2,
        }
        out = convert_openai_to_claude(payload)
        self.assertEqual(out["max_tokens"], 100)
        self.assertEqual(out["temperature"], 0.2)


class TestConvertOpenAIToClaude37(unittest.TestCase):
    def test_string_content_becomes_text_block(self):
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        out = convert_openai_to_claude37(payload)
        self.assertEqual(
            out["messages"], [{"role": "user", "content": [{"text": "hello"}]}]
        )

    def test_inference_config_mapping(self):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 256,
            "temperature": 0.5,
            "stop": "STOP",
        }
        out = convert_openai_to_claude37(payload)
        cfg = out["inferenceConfig"]
        self.assertEqual(cfg["maxTokens"], 256)
        self.assertEqual(cfg["temperature"], 0.5)
        self.assertEqual(cfg["stopSequences"], ["STOP"])  # str -> list

    def test_stop_list_passthrough(self):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["a", "b"],
        }
        out = convert_openai_to_claude37(payload)
        self.assertEqual(out["inferenceConfig"]["stopSequences"], ["a", "b"])

    def test_system_message_inserted_as_first_user_block(self):
        payload = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]
        }
        out = convert_openai_to_claude37(payload)
        self.assertEqual(out["messages"][0], {"role": "user", "content": [{"text": "sys"}]})
        self.assertEqual(out["messages"][1], {"role": "user", "content": [{"text": "hi"}]})

    def test_image_url_data_url_to_bedrock_converse(self):
        b64 = "aGVsbG8="
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ]
        }
        out = convert_openai_to_claude37(payload)
        block = out["messages"][0]["content"][0]
        self.assertEqual(block["image"]["format"], "png")
        self.assertEqual(block["image"]["source"]["bytes"], b64)

    def test_unsupported_role_skipped(self):
        payload = {
            "messages": [
                {"role": "tool", "content": "ignored"},
                {"role": "user", "content": "kept"},
            ]
        }
        out = convert_openai_to_claude37(payload)
        # Only the user message survives.
        self.assertEqual(len(out["messages"]), 1)
        self.assertEqual(out["messages"][0]["role"], "user")

    def test_no_inference_config_when_empty(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out = convert_openai_to_claude37(payload)
        self.assertNotIn("inferenceConfig", out)


class TestConvertClaudeRequestToOpenAI(unittest.TestCase):
    def test_system_and_token_mapping(self):
        payload = {
            "model": "claude-x",
            "system": "sys",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50,
            "temperature": 0.3,
            "stream": True,
        }
        out = convert_claude_request_to_openai(payload)
        self.assertEqual(out["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(out["messages"][1], {"role": "user", "content": "hi"})
        self.assertEqual(out["max_completion_tokens"], 50)
        self.assertEqual(out["temperature"], 0.3)
        self.assertTrue(out["stream"])
        self.assertEqual(out["model"], "claude-x")

    def test_tools_mapped_to_function_format(self):
        payload = {
            "model": "claude-x",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"name": "t", "description": "d", "input_schema": {"type": "object"}}
            ],
        }
        out = convert_claude_request_to_openai(payload)
        tool = out["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "t")
        self.assertEqual(tool["function"]["parameters"], {"type": "object"})

    def test_no_system_when_absent(self):
        payload = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]}
        out = convert_claude_request_to_openai(payload)
        self.assertEqual(out["messages"][0]["role"], "user")


class TestConvertClaudeRequestToGemini(unittest.TestCase):
    def test_system_prepended_to_first_user(self):
        payload = {
            "system": "SYS",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = convert_claude_request_to_gemini(payload)
        text = out["contents"][0]["parts"]["text"]
        self.assertIn("SYS", text)
        self.assertIn("hi", text)

    def test_role_mapping_and_merge(self):
        payload = {
            "messages": [
                {"role": "assistant", "content": "a1"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "u1"},
            ]
        }
        out = convert_claude_request_to_gemini(payload)
        # Two consecutive assistant messages merge into one 'model' entry.
        self.assertEqual(out["contents"][0]["role"], "model")
        self.assertIn("a1", out["contents"][0]["parts"]["text"])
        self.assertIn("a2", out["contents"][0]["parts"]["text"])
        self.assertEqual(out["contents"][1]["role"], "user")

    def test_list_content_text_join(self):
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}
            ]
        }
        out = convert_claude_request_to_gemini(payload)
        self.assertEqual(out["contents"][0]["parts"]["text"], "x y")

    def test_generation_config_and_tools(self):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 42,
            "temperature": 0.7,
            "tools": [{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
        }
        out = convert_claude_request_to_gemini(payload)
        self.assertEqual(out["generation_config"]["maxOutputTokens"], 42)
        self.assertEqual(out["generation_config"]["temperature"], 0.7)
        self.assertEqual(out["tools"][0]["function_declarations"][0]["name"], "t")


if __name__ == "__main__":
    unittest.main()

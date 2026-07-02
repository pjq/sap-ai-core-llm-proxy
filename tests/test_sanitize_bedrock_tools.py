import unittest


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from proxy_server import sanitize_bedrock_tools
from proxy_server import sanitize_bedrock_tool_result_content


class TestSanitizeBedrockTools(unittest.TestCase):
    def test_removes_custom_defer_loading(self):
        body = {
            "tools": [
                {
                    "name": "t1",
                    "description": "d",
                    "input_schema": {"type": "object", "properties": {}},
                    "custom": {"defer_loading": True},
                }
            ]
        }

        sanitize_bedrock_tools(body, request_id="test")

        self.assertIn("tools", body)
        self.assertEqual(1, len(body["tools"]))
        self.assertNotIn("custom", body["tools"][0])

    def test_strips_unsupported_tool_types_and_tool_choice(self):
        body = {
            "tools": [
                {"type": "web_search_20250305", "name": "search"},
                {"type": "computer_20241022", "name": "computer"},
            ],
            "tool_choice": {"type": "auto"},
        }

        sanitize_bedrock_tools(body, request_id="test")

        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)


class TestSanitizeBedrockToolResultContent(unittest.TestCase):
    def test_converts_tool_reference_block_to_text(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "1",
                            "content": [
                                {"type": "tool_reference", "tool_name": "Read"},
                                {"type": "text", "text": "ok"},
                            ],
                        }
                    ],
                }
            ]
        }

        sanitize_bedrock_tool_result_content(body, request_id="test")

        tool_result = body["messages"][0]["content"][0]
        self.assertEqual("tool_result", tool_result["type"])
        self.assertIsInstance(tool_result["content"], list)
        self.assertEqual("text", tool_result["content"][0]["type"])
        self.assertIn("tool_ref", tool_result["content"][0]["text"])
        self.assertEqual("text", tool_result["content"][1]["type"])
        self.assertEqual("ok", tool_result["content"][1]["text"])

    def test_normalizes_string_tool_result_content_to_list(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "1",
                            "content": "hello",
                        }
                    ],
                }
            ]
        }

        sanitize_bedrock_tool_result_content(body, request_id="test")

        tool_result = body["messages"][0]["content"][0]
        self.assertEqual([{"type": "text", "text": "hello"}], tool_result["content"])


if __name__ == "__main__":
    unittest.main()

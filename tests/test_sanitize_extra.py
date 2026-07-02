"""Extra branch coverage for the Bedrock sanitizers.

test_sanitize_bedrock_tools.py already covers the happy paths; this file
pins the remaining edge branches (invalid inputs, count return values,
passthrough) so the sanitizers can be refactored safely.
"""

import unittest

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from proxy_server import sanitize_bedrock_tools, sanitize_bedrock_tool_result_content


class TestSanitizeBedrockToolsEdges(unittest.TestCase):
    def test_non_list_tools_is_noop(self):
        body = {"tools": "not-a-list"}
        result = sanitize_bedrock_tools(body)
        self.assertEqual(result, {"removed_fields": 0, "stripped_tools": 0})
        self.assertEqual(body["tools"], "not-a-list")  # untouched

    def test_missing_tools_key_is_noop(self):
        body = {}
        result = sanitize_bedrock_tools(body)
        self.assertEqual(result["stripped_tools"], 0)

    def test_non_dict_tool_is_stripped(self):
        body = {"tools": ["oops", 123]}
        sanitize_bedrock_tools(body)
        self.assertNotIn("tools", body)  # all stripped -> key removed

    def test_missing_or_empty_name_stripped(self):
        body = {
            "tools": [
                {"description": "d", "input_schema": {"type": "object"}},  # no name
                {"name": "", "description": "d", "input_schema": {"type": "object"}},  # empty
            ]
        }
        sanitize_bedrock_tools(body)
        self.assertNotIn("tools", body)

    def test_non_dict_input_schema_stripped(self):
        body = {"tools": [{"name": "t", "description": "d", "input_schema": "bad"}]}
        sanitize_bedrock_tools(body)
        self.assertNotIn("tools", body)

    def test_non_str_description_coerced_to_empty(self):
        body = {"tools": [{"name": "t", "description": 123, "input_schema": {"type": "object"}}]}
        sanitize_bedrock_tools(body)
        self.assertEqual(body["tools"][0]["description"], "")

    def test_removed_fields_counted(self):
        body = {
            "tools": [
                {
                    "name": "t",
                    "description": "d",
                    "input_schema": {"type": "object"},
                    "extra1": 1,
                    "extra2": 2,
                }
            ]
        }
        result = sanitize_bedrock_tools(body)
        self.assertEqual(result["removed_fields"], 2)
        self.assertEqual(set(body["tools"][0].keys()), {"name", "description", "input_schema"})

    def test_valid_tool_preserved(self):
        body = {"tools": [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]}
        sanitize_bedrock_tools(body)
        self.assertEqual(len(body["tools"]), 1)


class TestSanitizeToolResultContentEdges(unittest.TestCase):
    def test_dict_content_dumped_to_text(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "1", "content": {"k": "v"}}
                    ],
                }
            ]
        }
        result = sanitize_bedrock_tool_result_content(body)
        block = body["messages"][0]["content"][0]["content"][0]
        self.assertEqual(block["type"], "text")
        self.assertIn("k", block["text"])
        self.assertEqual(result["normalized_tool_results"], 1)

    def test_allowed_content_types_passthrough(self):
        original = [{"type": "text", "text": "keep"}, {"type": "image", "source": {}}]
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "1", "content": list(original)}
                    ],
                }
            ]
        }
        sanitize_bedrock_tool_result_content(body)
        self.assertEqual(body["messages"][0]["content"][0]["content"], original)

    def test_unsupported_block_converted(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "1",
                            "content": [{"type": "weird_block", "x": 1}],
                        }
                    ],
                }
            ]
        }
        result = sanitize_bedrock_tool_result_content(body)
        block = body["messages"][0]["content"][0]["content"][0]
        self.assertEqual(block["type"], "text")
        self.assertIn("[unsupported_tool_result_content]", block["text"])
        self.assertEqual(result["converted_blocks"], 1)

    def test_non_list_messages_noop(self):
        body = {"messages": "nope"}
        result = sanitize_bedrock_tool_result_content(body)
        self.assertEqual(result, {"converted_blocks": 0, "normalized_tool_results": 0})

    def test_non_tool_result_untouched(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}
            ]
        }
        sanitize_bedrock_tool_result_content(body)
        self.assertEqual(body["messages"][0]["content"][0], {"type": "text", "text": "hi"})


if __name__ == "__main__":
    unittest.main()

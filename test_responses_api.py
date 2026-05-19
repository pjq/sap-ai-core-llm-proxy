import unittest
import json
import time
from unittest.mock import patch, MagicMock

from proxy_server import (
    convert_responses_input_to_messages,
    convert_responses_tools,
    build_responses_api_response,
    _gen_resp_id,
    _gen_msg_id,
)


class TestConvertResponsesInputToMessages(unittest.TestCase):

    def test_string_input(self):
        payload = {"input": "Hello world"}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(messages, [{"role": "user", "content": "Hello world"}])

    def test_string_input_with_instructions(self):
        payload = {"input": "Hi", "instructions": "You are helpful."}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": "You are helpful."})
        self.assertEqual(messages[1], {"role": "user", "content": "Hi"})

    def test_none_input(self):
        payload = {"instructions": "System msg"}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(messages, [{"role": "system", "content": "System msg"}])

    def test_message_with_input_text(self):
        payload = {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        ]}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "hello"}])

    def test_output_text_converted_to_text(self):
        payload = {"input": [
            {"role": "assistant", "content": [{"type": "output_text", "text": "I said hi"}]}
        ]}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "I said hi"}])

    def test_function_call_uses_call_id(self):
        payload = {"input": [
            {"type": "function_call", "id": "fc_1", "call_id": "call_abc", "name": "my_func", "arguments": "{}"}
        ]}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_abc")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "my_func")

    def test_function_call_output(self):
        payload = {"input": [
            {"type": "function_call", "call_id": "call_abc", "name": "my_func", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_abc", "output": "result123"}
        ]}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(len(messages), 2)
        # Assistant message with tool_calls
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_abc")
        # Tool response
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "call_abc")
        self.assertEqual(messages[1]["content"], "result123")

    def test_consecutive_function_calls_merged(self):
        payload = {"input": [
            {"type": "function_call", "call_id": "call_1", "name": "func_a", "arguments": "{\"x\":1}"},
            {"type": "function_call", "call_id": "call_2", "name": "func_b", "arguments": "{\"y\":2}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "out1"},
            {"type": "function_call_output", "call_id": "call_2", "output": "out2"},
        ]}
        messages = convert_responses_input_to_messages(payload)
        # Should be: 1 assistant msg with 2 tool_calls, then 2 tool responses
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(len(messages[0]["tool_calls"]), 2)
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[0]["tool_calls"][1]["id"], "call_2")
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "call_1")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call_2")

    def test_non_consecutive_function_calls_separate(self):
        payload = {"input": [
            {"type": "function_call", "call_id": "call_1", "name": "func_a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "out1"},
            {"type": "function_call", "call_id": "call_2", "name": "func_b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_2", "output": "out2"},
        ]}
        messages = convert_responses_input_to_messages(payload)
        # Should be: assistant(1 tool_call), tool, assistant(1 tool_call), tool
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(len(messages[0]["tool_calls"]), 1)
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(len(messages[2]["tool_calls"]), 1)
        self.assertEqual(messages[3]["role"], "tool")

    def test_full_conversation_with_codex_pattern(self):
        """Simulate a real codex conversation: user → assistant text → function_calls → outputs → user"""
        payload = {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "explore the code"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Let me look at the files."}]},
            {"type": "function_call", "call_id": "call_rg", "name": "exec_command", "arguments": "{\"cmd\":\"rg --files\"}"},
            {"type": "function_call", "call_id": "call_ls", "name": "exec_command", "arguments": "{\"cmd\":\"ls -la\"}"},
            {"type": "function_call_output", "call_id": "call_rg", "output": "file1.py\nfile2.py"},
            {"type": "function_call_output", "call_id": "call_ls", "output": "total 16\n-rw-r--r-- 1 user file1.py"},
            {"role": "user", "content": [{"type": "input_text", "text": "what did you find?"}]}
        ]}
        messages = convert_responses_input_to_messages(payload)
        # user, assistant text, assistant tool_calls, tool, tool, user = 6
        self.assertEqual(len(messages), 6)
        # user
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "explore the code"}])
        # assistant text (output_text → text)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "Let me look at the files."}])
        # merged tool calls (separate assistant msg because previous assistant had content not tool_calls)
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(len(messages[2]["tool_calls"]), 2)
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call_rg")
        self.assertEqual(messages[2]["tool_calls"][1]["id"], "call_ls")
        # tool responses
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call_rg")
        self.assertEqual(messages[4]["role"], "tool")
        self.assertEqual(messages[4]["tool_call_id"], "call_ls")
        # follow-up user
        self.assertEqual(messages[5]["role"], "user")

    def test_input_image_conversion(self):
        payload = {"input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": "describe this"},
                {"type": "input_image", "image_url": "https://example.com/img.png"}
            ]}
        ]}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"][0], {"type": "text", "text": "describe this"})
        self.assertEqual(messages[0]["content"][1], {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}})

    def test_string_items_in_list(self):
        payload = {"input": ["hello", "world"]}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "user", "content": "hello"})
        self.assertEqual(messages[1], {"role": "user", "content": "world"})


class TestConvertResponsesTools(unittest.TestCase):

    def test_function_tools_converted(self):
        tools = [
            {"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}}
        ]
        result = convert_responses_tools(tools)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_non_function_tools_skipped(self):
        tools = [
            {"type": "web_search_preview"},
            {"type": "container_exec", "container": {}},
            {"type": "function", "name": "my_func", "description": "d", "parameters": {}}
        ]
        result = convert_responses_tools(tools)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "my_func")

    def test_no_tools_returns_none(self):
        self.assertIsNone(convert_responses_tools(None))
        self.assertIsNone(convert_responses_tools([]))

    def test_only_non_function_tools_returns_none(self):
        tools = [{"type": "web_search_preview"}, {"type": "shell"}]
        result = convert_responses_tools(tools)
        self.assertIsNone(result)


class TestBuildResponsesApiResponse(unittest.TestCase):

    def test_text_response(self):
        completion = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        }
        result = build_responses_api_response(completion, "gpt-5.4", "resp_test123")
        self.assertEqual(result["id"], "resp_test123")
        self.assertEqual(result["object"], "response")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model"], "gpt-5.4")
        self.assertEqual(len(result["output"]), 1)
        self.assertEqual(result["output"][0]["type"], "message")
        self.assertEqual(result["output"][0]["content"][0]["type"], "output_text")
        self.assertEqual(result["output"][0]["content"][0]["text"], "Hello!")
        self.assertEqual(result["usage"]["input_tokens"], 5)
        self.assertEqual(result["usage"]["output_tokens"], 3)

    def test_tool_call_response(self):
        completion = {
            "choices": [{"message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
                }]
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        result = build_responses_api_response(completion, "gpt-5.4", "resp_test456")
        self.assertEqual(len(result["output"]), 1)
        self.assertEqual(result["output"][0]["type"], "function_call")
        self.assertEqual(result["output"][0]["name"], "get_weather")
        self.assertEqual(result["output"][0]["arguments"], "{\"city\":\"NYC\"}")
        self.assertEqual(result["output"][0]["call_id"], "call_xyz")

    def test_text_and_tool_call_response(self):
        completion = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"}
                }]
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        }
        result = build_responses_api_response(completion, "gpt-5.4", "resp_test789")
        self.assertEqual(len(result["output"]), 2)
        # Function call first
        self.assertEqual(result["output"][0]["type"], "function_call")
        # Then message
        self.assertEqual(result["output"][1]["type"], "message")
        self.assertEqual(result["output"][1]["content"][0]["text"], "Let me check that.")

    def test_empty_choices(self):
        completion = {"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        result = build_responses_api_response(completion, "gpt-5.4", "resp_empty")
        self.assertEqual(result["output"], [])


class TestIdGeneration(unittest.TestCase):

    def test_resp_id_format(self):
        rid = _gen_resp_id()
        self.assertTrue(rid.startswith("resp_"))
        self.assertEqual(len(rid), 37)  # "resp_" + 32 hex chars

    def test_msg_id_format(self):
        mid = _gen_msg_id()
        self.assertTrue(mid.startswith("msg_"))
        self.assertEqual(len(mid), 36)  # "msg_" + 32 hex chars

    def test_ids_unique(self):
        ids = {_gen_resp_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()

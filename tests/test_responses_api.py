import unittest
import json
import time
from unittest.mock import patch, MagicMock

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
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
        """A function_call with matching output uses call_id as the tool_call id."""
        payload = {"input": [
            {"type": "function_call", "id": "fc_1", "call_id": "call_abc", "name": "my_func", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_abc", "output": "done"}
        ]}
        messages = convert_responses_input_to_messages(payload)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_abc")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "my_func")
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "call_abc")

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

    def test_orphaned_tool_calls_stripped(self):
        """Tool calls without matching function_call_output should be removed."""
        payload = {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "do something"}]},
            {"type": "function_call", "call_id": "call_good", "name": "func_a", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_orphan", "name": "func_b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_good", "output": "result"},
            # No output for call_orphan
            {"role": "user", "content": [{"type": "input_text", "text": "continue"}]}
        ]}
        messages = convert_responses_input_to_messages(payload)
        # The orphaned tool_call should be stripped
        tool_call_ids = []
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tool_call_ids.append(tc["id"])
        self.assertIn("call_good", tool_call_ids)
        self.assertNotIn("call_orphan", tool_call_ids)

    def test_all_tool_calls_orphaned_removes_assistant_msg(self):
        """If ALL tool calls in a mid-conversation assistant message are orphaned, the message is removed."""
        payload = {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "call_orphan1", "name": "func_a", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_orphan2", "name": "func_b", "arguments": "{}"},
            # No function_call_output for either, but more messages follow
            {"role": "user", "content": [{"type": "input_text", "text": "ok"}]}
        ]}
        messages = convert_responses_input_to_messages(payload)
        # The assistant message with orphaned tool_calls should be completely removed
        roles = [msg["role"] for msg in messages]
        self.assertEqual(roles, ["user", "user"])

    def test_trailing_tool_calls_without_output_are_stripped(self):
        """Tool calls at the end without output are also stripped (backend rejects them)."""
        payload = {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "do it"}]},
            {"type": "function_call", "call_id": "call_pending", "name": "exec_cmd", "arguments": "{}"},
        ]}
        messages = convert_responses_input_to_messages(payload)
        # Only the user message remains
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    def test_interleaved_developer_message_reordered(self):
        """Developer messages between tool_call and tool response are moved after the tool response."""
        payload = {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "run git checkout"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Creating branch..."}]},
            {"type": "function_call", "call_id": "call_git", "name": "exec_command", "arguments": "{\"cmd\":\"git checkout -b new\"}"},
            {"role": "developer", "content": [{"type": "input_text", "text": "Approved command prefix saved: git checkout"}]},
            {"type": "function_call_output", "call_id": "call_git", "output": "Switched to a new branch 'new'"},
            {"role": "user", "content": [{"type": "input_text", "text": "continue"}]}
        ]}
        messages = convert_responses_input_to_messages(payload)
        # Find the assistant with tool_calls
        tc_idx = next(i for i, m in enumerate(messages) if m.get("role") == "assistant" and "tool_calls" in m)
        # Tool response must immediately follow the assistant tool_calls
        self.assertEqual(messages[tc_idx + 1]["role"], "tool")
        self.assertEqual(messages[tc_idx + 1]["tool_call_id"], "call_git")
        # Developer message comes after the tool response
        dev_idx = next(i for i, m in enumerate(messages) if m.get("role") == "developer")
        self.assertGreater(dev_idx, tc_idx + 1)


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
        self.assertEqual(result["output_text"], "Hello!")
        self.assertIsNone(result["error"])
        self.assertTrue(result["parallel_tool_calls"])
        self.assertEqual(result["text"]["format"]["type"], "text")
        self.assertEqual(result["usage"]["input_tokens"], 5)
        self.assertEqual(result["usage"]["output_tokens"], 3)
        self.assertEqual(result["usage"]["input_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(result["usage"]["output_tokens_details"]["reasoning_tokens"], 0)

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
        self.assertEqual(result["output_text"], "")

    def test_list_content_response(self):
        completion = {
            "choices": [{"message": {"role": "assistant", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "output_text", "text": " world", "annotations": [{"type": "citation"}]}
            ]}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        }
        result = build_responses_api_response(completion, "gpt-5.4", "resp_list")
        self.assertEqual(result["output_text"], "Hello world")
        self.assertEqual(result["output"][0]["content"][1]["annotations"], [{"type": "citation"}])


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


class TestGenerateResponsesStreaming(unittest.TestCase):
    """Test SSE event format for streaming responses."""

    def _collect_events(self, generator):
        """Parse SSE events from generator output."""
        events = []
        for chunk in generator:
            lines = chunk.strip().split("\n")
            event_type = None
            data = None
            for line in lines:
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
            if event_type and data:
                events.append((event_type, data))
        return events

    @patch('proxy_server._http_session')
    @patch('proxy_server.fetch_token', return_value='test-token')
    def test_text_streaming_event_format(self, mock_token, mock_session):
        """Verify SSE events match the OpenAI Responses API spec."""
        from proxy_server import generate_responses_streaming

        # Mock backend streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter([
            'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}',
            'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}',
            'data: {"choices":[{"delta":{"content":" world"},"index":0}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
            'data: [DONE]',
        ])
        mock_session.post.return_value = mock_response

        events = self._collect_events(
            generate_responses_streaming("http://test", {}, {"messages": []}, "gpt-5.4", "test")
        )

        # Check response.created has nested response object
        self.assertEqual(events[0][0], "response.created")
        self.assertIn("response", events[0][1])
        self.assertEqual(events[0][1]["type"], "response.created")
        self.assertEqual(events[0][1]["response"]["status"], "in_progress")
        self.assertEqual(events[0][1]["response"]["output_text"], "")

        # Check response.in_progress
        self.assertEqual(events[1][0], "response.in_progress")
        self.assertIn("response", events[1][1])
        self.assertEqual(events[1][1]["type"], "response.in_progress")

        # Check output_item.added has response_id
        self.assertEqual(events[2][0], "response.output_item.added")
        self.assertIn("response_id", events[2][1])
        self.assertEqual(events[2][1]["item"]["type"], "message")

        # Check content_part.added
        self.assertEqual(events[3][0], "response.content_part.added")
        self.assertIn("response_id", events[3][1])

        # Check text deltas have response_id
        text_deltas = [(t, d) for t, d in events if t == "response.output_text.delta"]
        self.assertEqual(len(text_deltas), 2)
        self.assertEqual(text_deltas[0][1]["delta"], "Hello")
        self.assertEqual(text_deltas[1][1]["delta"], " world")
        self.assertIn("response_id", text_deltas[0][1])

        # Check response.completed has nested response object
        completed = [(t, d) for t, d in events if t == "response.completed"]
        self.assertEqual(len(completed), 1)
        self.assertIn("response", completed[0][1])
        self.assertEqual(completed[0][1]["type"], "response.completed")
        self.assertEqual(completed[0][1]["response"]["status"], "completed")
        self.assertEqual(completed[0][1]["response"]["output_text"], "Hello world")
        self.assertEqual(completed[0][1]["response"]["usage"]["input_tokens"], 5)
        self.assertEqual(completed[0][1]["response"]["usage"]["output_tokens"], 2)
        self.assertEqual(completed[0][1]["response"]["usage"]["output_tokens_details"]["reasoning_tokens"], 0)

    @patch('proxy_server._http_session')
    @patch('proxy_server.fetch_token', return_value='test-token')
    def test_function_call_streaming_event_format(self, mock_token, mock_session):
        """Verify function call SSE events match OpenAI spec."""
        from proxy_server import generate_responses_streaming

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter([
            'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"index":0}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\""}}]},"index":0}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"NYC\\"}"}}]},"index":0}]}',
            'data: [DONE]',
        ])
        mock_session.post.return_value = mock_response

        events = self._collect_events(
            generate_responses_streaming("http://test", {}, {"messages": []}, "gpt-5.4", "test")
        )

        # Find function call events
        fc_added = [(t, d) for t, d in events if t == "response.output_item.added" and d.get("item", {}).get("type") == "function_call"]
        self.assertEqual(len(fc_added), 1)
        self.assertEqual(fc_added[0][1]["item"]["name"], "get_weather")
        self.assertEqual(fc_added[0][1]["item"]["call_id"], "call_abc")
        self.assertIn("response_id", fc_added[0][1])

        # Check arguments deltas
        arg_deltas = [(t, d) for t, d in events if t == "response.function_call_arguments.delta"]
        self.assertEqual(len(arg_deltas), 2)
        self.assertIn("response_id", arg_deltas[0][1])

        # Check done events
        arg_done = [(t, d) for t, d in events if t == "response.function_call_arguments.done"]
        self.assertEqual(len(arg_done), 1)
        self.assertEqual(arg_done[0][1]["arguments"], '{"city": "NYC"}')
        self.assertIn("response_id", arg_done[0][1])

        # Check response.completed
        completed = [(t, d) for t, d in events if t == "response.completed"]
        self.assertEqual(len(completed), 1)
        self.assertIn("response", completed[0][1])
        fc_output = completed[0][1]["response"]["output"]
        self.assertEqual(len(fc_output), 1)
        self.assertEqual(fc_output[0]["type"], "function_call")
        self.assertEqual(fc_output[0]["name"], "get_weather")

    @patch('proxy_server._http_session')
    @patch('proxy_server.fetch_token', return_value='test-token')
    def test_all_events_have_sequence_number(self, mock_token, mock_session):
        """Every SSE event must have a sequence_number field."""
        from proxy_server import generate_responses_streaming

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter([
            'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}',
            'data: [DONE]',
        ])
        mock_session.post.return_value = mock_response

        events = self._collect_events(
            generate_responses_streaming("http://test", {}, {"messages": []}, "gpt-5.4", "test")
        )

        for event_type, data in events:
            self.assertIn("sequence_number", data,
                          f"Event {event_type} missing sequence_number")
            self.assertIsInstance(data["sequence_number"], int)

        # Verify sequence numbers are monotonically increasing
        seq_nums = [d["sequence_number"] for _, d in events]
        self.assertEqual(seq_nums, sorted(seq_nums))
        self.assertEqual(len(set(seq_nums)), len(seq_nums))


if __name__ == "__main__":
    unittest.main()

"""Native GPT-6 Responses relay regressions, including function-tool state."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import proxy_server as p


class TestNativeResponses(unittest.TestCase):
    def setUp(self):
        cfg = p.ProxyConfig(subaccounts={"SFMobileSH": SimpleNamespace(
            service_key=SimpleNamespace(identityzoneid="tenant"))},
            model_to_subaccounts={"gpt-6.1-sol": ["SFMobileSH"], "gpt-6-luna": ["SFMobileSH"]})
        self.patches = [
            patch.object(p, "proxy_config", cfg),
            patch.object(p, "verify_request_token", return_value=True),
            patch.object(p, "fetch_token", return_value="oauth"),
            patch.object(p, "load_balance_url", side_effect=lambda model: (
                "https://example.test/deployments/test", "SFMobileSH", "default", model)),
            patch.object(p, "_http_session"),
        ]
        mocks = [x.start() for x in self.patches]
        for x in self.patches:
            self.addCleanup(x.stop)
        self.session = mocks[-1]
        self.backend = self.session.post.return_value
        self.backend.status_code = 200
        self.client = p.app.test_client()

    def test_tools_reasoning_and_output_limit_pass_through(self):
        payload = {"model": "gpt-6.1-sol", "input": "Hi", "reasoning": {"effort": "medium"},
                   "max_output_tokens": 1024, "tools": [{"type": "function", "name": "read"}]}
        self.backend.json.return_value = {"id": "resp_test", "output": [{"type": "function_call", "call_id": "call_1"}]}
        response = self.client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, self.backend.json.return_value)
        self.assertEqual(self.session.post.call_args.args[0],
                         "https://example.test/deployments/test/responses?api-version=2025-04-01-preview")
        self.assertEqual(self.session.post.call_args.kwargs["json"], payload)
        self.assertEqual(self.session.post.call_args.kwargs["headers"]["Authorization"], "Bearer oauth")
        self.backend.close.assert_called_once()

    def test_follow_up_input_items_are_not_converted(self):
        payload = {"model": "gpt-6-luna", "input": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"}]}
        self.backend.json.return_value = {"output": []}
        self.assertEqual(self.client.post("/v1/responses", json=payload).status_code, 200)
        self.assertEqual(self.session.post.call_args.kwargs["json"], payload)

    def test_stream_is_relayed_verbatim_and_closed(self):
        chunks = [b'event: response.output_item.added\ndata: {"type":"function_call"}\n\n',
                  b'', b'event: response.completed\ndata: {"type":"response.completed"}\n\n']
        self.backend.iter_content.return_value = iter(chunks)
        response = self.client.post("/v1/responses", json={"model": "gpt-6-luna", "input": "Hi", "stream": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"".join(chunks))
        self.assertTrue(self.session.post.call_args.kwargs["stream"])
        self.backend.close.assert_called()
        response.close()

    def test_backend_error_preserves_status_and_details(self):
        self.backend.status_code = 400
        error = {"error": {"code": "unsupported_parameter", "param": "bad", "message": "Bad field"}}
        self.backend.json.return_value = error
        response = self.client.post("/v1/responses", json={"model": "gpt-6.1-sol", "input": "Hi", "stream": True})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, error)
        self.backend.close.assert_called_once()

    def test_non_json_error_is_returned_and_closed(self):
        self.backend.status_code = 502
        self.backend.json.side_effect = ValueError("Not JSON")
        self.backend.text = "Gateway unavailable"
        response = self.client.post("/v1/responses", json={"model": "gpt-6.1-sol", "input": "Hi"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json["error"]["message"], "Gateway unavailable")
        self.backend.close.assert_called_once()

    def test_stream_exception_closes_backend(self):
        def broken():
            yield b"data: {}\n\n"
            raise RuntimeError("Disconnected")
        self.backend.iter_content.return_value = broken()
        response = self.client.post("/v1/responses", json={"model": "gpt-6.1-sol", "input": "Hi", "stream": True})
        with self.assertRaises(RuntimeError):
            _ = response.data
        self.backend.close.assert_called()
        response.close()


if __name__ == "__main__":
    unittest.main()

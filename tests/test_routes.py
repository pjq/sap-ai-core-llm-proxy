"""Characterization tests for the Flask route handlers.

Uses app.test_client(). All upstream I/O is mocked:
- proxy_config is replaced with a fake 3-model config (GPT/Claude/Gemini).
- fetch_token is stubbed (no OAuth call).
- _http_session.post is stubbed for GPT/embeddings paths.
- get_sapaicore_sdk_client is stubbed for the Claude /v1/messages path.

"# QUIRK" marks behavior the optimization phase may revisit (e.g. /v1/models
performing no auth check).
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import proxy_server
from proxy_server import (
    app,
    ProxyConfig,
    SubAccountConfig,
    ServiceKey,
    load_balance_url,
)

TOKEN = "test-secret-token"


def _make_config():
    sub = SubAccountConfig(
        name="S",
        resource_group="rg",
        service_key_json="unused.json",
        deployment_models={
            "gpt-5.4": ["https://backend/gpt"],
            "anthropic--claude-4.6-opus": ["https://backend/claude"],
            "gemini-2.5-pro": ["https://backend/gemini"],
        },
    )
    sub.service_key = ServiceKey("id", "secret", "https://auth", "zone")
    sub.normalized_models = dict(sub.deployment_models)
    cfg = ProxyConfig(
        subaccounts={"S": sub},
        secret_authentication_tokens=[TOKEN],
        model_to_subaccounts={},
    )
    cfg.build_model_mapping()
    return cfg


def _fake_post(json_body, status=200):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_body
    resp.status_code = status
    resp.close.return_value = None
    return resp


class _RouteTestBase(unittest.TestCase):
    def setUp(self):
        load_balance_url.counters = {}
        self.cfg = _make_config()
        self._p = patch.object(proxy_server, "proxy_config", self.cfg)
        self._p.start()
        self.client = app.test_client()
        self._ft = patch.object(proxy_server, "fetch_token", return_value="backend-tok")
        self._ft.start()

    def tearDown(self):
        self._ft.stop()
        self._p.stop()
        load_balance_url.counters = {}

    def _auth(self):
        return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


class TestHealthAndModels(_RouteTestBase):
    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_models_lists_configured(self):
        # QUIRK: /v1/models performs no token verification (auth check commented out).
        r = self.client.get("/v1/models")
        self.assertEqual(r.status_code, 200)
        ids = {m["id"] for m in r.get_json()["data"]}
        self.assertEqual(ids, {"gpt-5.4", "anthropic--claude-4.6-opus", "gemini-2.5-pro"})


class TestAuth(_RouteTestBase):
    def test_missing_token_401(self):
        r = self.client.post("/v1/chat/completions", json={"model": "gpt-5.4", "messages": []})
        self.assertEqual(r.status_code, 401)

    def test_invalid_token_401(self):
        r = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "gpt-5.4", "messages": []},
        )
        self.assertEqual(r.status_code, 401)

    def test_messages_missing_token_401(self):
        r = self.client.post("/v1/messages", json={"model": "anthropic--claude-4.6-opus", "messages": []})
        self.assertEqual(r.status_code, 401)

    def test_auth_disabled_when_no_tokens_configured(self):
        # Pin behavior: empty secret list => auth disabled (verify_request_token True).
        self.cfg.secret_authentication_tokens = []
        with patch.object(proxy_server, "_http_session") as sess:
            sess.post.return_value = _fake_post({"usage": {}, "choices": []})
            r = self.client.post(
                "/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
            )
        self.assertEqual(r.status_code, 200)


class TestChatCompletionsRouting(_RouteTestBase):
    def test_gpt_non_streaming(self):
        backend_json = {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 3}}
        with patch.object(proxy_server, "_http_session") as sess:
            sess.post.return_value = _fake_post(backend_json)
            r = self.client.post(
                "/v1/chat/completions",
                headers=self._auth(),
                json={"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["choices"][0]["message"]["content"], "hi")

    def test_unknown_model_falls_back_to_gpt(self):
        backend_json = {"choices": [{"message": {"content": "x"}}], "usage": {}}
        with patch.object(proxy_server, "_http_session") as sess:
            sess.post.return_value = _fake_post(backend_json)
            r = self.client.post(
                "/v1/chat/completions",
                headers=self._auth(),
                json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        # Falls back to gpt-5.4 (present) -> 200.
        self.assertEqual(r.status_code, 200)

    def test_unknown_model_and_no_fallback_404(self):
        # Remove gpt-5.4 so the fallback also fails.
        del self.cfg.subaccounts["S"].normalized_models["gpt-5.4"]
        self.cfg.build_model_mapping()
        r = self.client.post(
            "/v1/chat/completions",
            headers=self._auth(),
            json={"model": "no-such-model", "messages": []},
        )
        self.assertEqual(r.status_code, 404)


class TestMessagesRoute(_RouteTestBase):
    def _sdk_returning(self, body_dict):
        """Fake SDK client whose invoke_model yields a JSON body of body_dict."""
        client = MagicMock()
        payload_bytes = json.dumps(body_dict).encode("utf-8")
        client.invoke_model.return_value = {"body": [payload_bytes]}
        return client

    def test_claude_non_streaming_returns_body(self):
        sdk = self._sdk_returning(
            {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 2}}
        )
        with patch.object(proxy_server, "get_sapaicore_sdk_client", return_value=sdk):
            r = self.client.post(
                "/v1/messages",
                headers={"x-api-key": TOKEN, "Content-Type": "application/json"},
                json={
                    "model": "anthropic--claude-4.6-opus",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": False,
                },
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["content"][0]["text"], "hi")

    def test_missing_model_400(self):
        r = self.client.post(
            "/v1/messages",
            headers={"x-api-key": TOKEN, "Content-Type": "application/json"},
            json={"messages": []},
        )
        self.assertEqual(r.status_code, 400)

    def test_system_message_extracted_into_top_level(self):
        # Capture the body passed to invoke_model and assert system extraction.
        sdk = self._sdk_returning({"content": [{"type": "text", "text": "x"}], "usage": {}})
        with patch.object(proxy_server, "get_sapaicore_sdk_client", return_value=sdk):
            self.client.post(
                "/v1/messages",
                headers={"x-api-key": TOKEN, "Content-Type": "application/json"},
                json={
                    "model": "anthropic--claude-4.6-opus",
                    "system": None,
                    "messages": [
                        {"role": "system", "content": "be nice"},
                        {"role": "user", "content": "hi"},
                    ],
                    "max_tokens": 50,
                    "stream": False,
                },
            )
        sent = json.loads(sdk.invoke_model.call_args.kwargs["body"])
        self.assertIn("system", sent)
        self.assertEqual(sent["system"], [{"type": "text", "text": "be nice"}])
        self.assertTrue(all(m["role"] != "system" for m in sent["messages"]))
        self.assertEqual(sent["anthropic_version"], "bedrock-2023-05-31")

    def test_context_management_and_output_config_removed(self):
        sdk = self._sdk_returning({"content": [{"type": "text", "text": "x"}], "usage": {}})
        with patch.object(proxy_server, "get_sapaicore_sdk_client", return_value=sdk):
            self.client.post(
                "/v1/messages",
                headers={"x-api-key": TOKEN, "Content-Type": "application/json"},
                json={
                    "model": "anthropic--claude-4.6-opus",
                    "messages": [{"role": "user", "content": "hi"}],
                    "context_management": {"foo": 1},
                    "output_config": {"bar": 2},
                    "max_tokens": 50,
                    "stream": False,
                },
            )
        sent = json.loads(sdk.invoke_model.call_args.kwargs["body"])
        self.assertNotIn("context_management", sent)
        self.assertNotIn("output_config", sent)

    def test_thinking_budget_adjusts_max_tokens(self):
        sdk = self._sdk_returning({"content": [{"type": "text", "text": "x"}], "usage": {}})
        with patch.object(proxy_server, "get_sapaicore_sdk_client", return_value=sdk):
            self.client.post(
                "/v1/messages",
                headers={"x-api-key": TOKEN, "Content-Type": "application/json"},
                json={
                    "model": "anthropic--claude-4.6-opus",
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking": {"type": "enabled", "budget_tokens": 1000},
                    "max_tokens": 500,  # below budget -> should be bumped to budget+1
                    "stream": False,
                },
            )
        sent = json.loads(sdk.invoke_model.call_args.kwargs["body"])
        self.assertEqual(sent["max_tokens"], 1001)


class TestEmbeddingsRoute(_RouteTestBase):
    def test_requires_input(self):
        r = self.client.post(
            "/v1/embeddings",
            headers=self._auth(),
            json={"model": "gpt-5.4"},
        )
        self.assertEqual(r.status_code, 400)

    def test_embedding_formatting(self):
        # embeddings uses load_balance_url on the model; add an embedding model.
        self.cfg.subaccounts["S"].normalized_models["text-embedding-3-large"] = ["https://backend/emb"]
        self.cfg.build_model_mapping()
        backend_json = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}
        with patch.object(proxy_server, "_http_session") as sess:
            sess.post.return_value = _fake_post(backend_json)
            r = self.client.post(
                "/v1/embeddings",
                headers=self._auth(),
                json={"model": "text-embedding-3-large", "input": "hello"},
            )
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["embedding"], [0.1, 0.2])
        self.assertEqual(body["usage"]["total_tokens"], 2)


class TestResponsesRoute(_RouteTestBase):
    def test_non_streaming_string_input(self):
        backend_json = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        with patch.object(proxy_server, "_http_session") as sess:
            sess.post.return_value = _fake_post(backend_json)
            r = self.client.post(
                "/v1/responses",
                headers=self._auth(),
                json={"model": "gpt-5.4", "input": "hi", "stream": False},
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["object"], "response")


if __name__ == "__main__":
    unittest.main()

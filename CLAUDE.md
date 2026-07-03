# SAP AI Core LLM Proxy

## Overview
A Python/Flask proxy server that translates OpenAI-compatible and Anthropic Claude API requests to SAP AI Core backend services. Supports GPT, Claude, and Gemini models through a unified interface.

## Architecture

### Entry Points (proxy_server.py)
- `/v1/chat/completions` — OpenAI Chat Completions API (streaming + non-streaming)
- `/v1/responses` — OpenAI Responses API (used by Codex CLI, modern OpenAI clients)
- `/v1/messages` — Anthropic Claude Messages API (used by Claude Code)
- `/v1/messages/count_tokens` — Anthropic token-counting endpoint (returns a tiktoken-based estimate; SAP AI Core Bedrock has no native count_tokens)
- `/v1/embeddings` — OpenAI Embeddings API
- `/v1/models` — List available models
- `/health` — Health check

### Request Flow
1. Auth: `verify_request_token()` checks `Authorization` or `x-api-key` header against `secret_authentication_tokens` in config
2. Model resolution: `load_balance_url(model)` → round-robin across subaccounts/deployments
3. Route by model type: `is_claude_model()` / `is_gemini_model()` / default (GPT)
4. Token fetch: `fetch_token(subaccount_name)` — per-subaccount OAuth with caching
5. Forward to SAP AI Core backend with appropriate transformation

### Model Routing
- `handle_default_request()` — GPT models → `/chat/completions?api-version=...`
- `handle_claude_request()` — Claude models → SAP AI SDK (bedrock)
- `handle_gemini_request()` — Gemini models → Gemini-specific endpoint

### Streaming
- `generate_streaming_response()` — Produces OpenAI SSE format (`data: {...}\n\n`)
- `generate_responses_streaming()` — Produces Responses API SSE format (`event: type\ndata: {...}\n\n`)
- `generate_claude_streaming_response()` — Produces Anthropic SSE format (`event: type\ndata: {...}\n\n`)

### Config (config.json)
```json
{
  "subAccounts": {
    "Name": {
      "resource_group": "default",
      "service_key_json": "key_file.json",
      "deployment_models": {
        "model-name": ["https://...deployment_url..."]
      }
    }
  },
  "secret_authentication_tokens": ["token1", "token2"],
  "port": 3001,
  "host": "0.0.0.0"
}
```

**IMPORTANT:** `config.json` is in `.gitignore` — never commit it.

## Key Patterns
- Global `_http_session` (requests.Session with retry/pooling) for non-SDK requests
- Global `_bedrock_clients` dict for SAP AI SDK clients (Claude models)
- `proxy_config` global holds all runtime state (subaccounts, model→subaccount mapping)
- Default fallback: GPT → `gpt-5.4`, Claude → `anthropic--claude-4.6-opus`
- `load_balance_url()` round-robin counters are guarded by `_load_balance_lock` (Waitress runs ~100 threads)
- Verbose debug logging uses the `_LazyJSON` wrapper so `json.dumps` only runs when DEBUG is enabled
- Streaming reads the GPT/other branch with `iter_content(chunk_size=8192)`; model-type predicates are hoisted out of per-chunk loops

## Running
```shell
python proxy_server.py --config config.json
python proxy_server.py --config config.json --debug
```

## Testing

### Unit tests
All tests live in `tests/` (a package with `__init__.py` + `conftest.py` that put the project
root on `sys.path`). No pytest in `.venv` — use the stdlib `unittest` runner:
```shell
python -m unittest discover -p 'test_*.py'   # run from the project root
```
The suite is a characterization net (~130 tests) covering the model predicates, load balancer,
request/response/chunk converters, sanitizers, streaming generators, and Flask routes. All upstream
I/O is mocked — no real SAP AI Core calls. Run it before and after any change to `proxy_server.py`;
green means behavior preserved. Tests that pin an intentional current quirk are marked
`# QUIRK: revisit in optimization`.

### Load testing
`tests/load_testing.py` is a standalone load-test script (not part of the unit suite — it hits a
live proxy). It resolves `config.json` from the project root, so run it from anywhere:
```shell
python tests/load_testing.py            # uses config.json at project root
CONFIG=configs/other.json python tests/load_testing.py
```

### Manual smoke tests (against a running server)
```shell
# Chat completions
curl http://127.0.0.1:3001/v1/chat/completions \
  -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}]}'

# Responses API
curl http://127.0.0.1:3001/v1/responses \
  -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","input":"hello","stream":true}'

# Claude Messages
curl http://127.0.0.1:3001/v1/messages \
  -H "x-api-key: TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"anthropic--claude-4.6-opus","messages":[{"role":"user","content":"hello"}],"max_tokens":100}'
```

## Dependencies
- Flask, requests, urllib3 — HTTP server and client
- `gen_ai_hub.proxy` (sap-ai-sdk) — SAP AI Core SDK for Claude/bedrock integration
- `sanitize_bedrock_tools()` — strips unsupported tool parameters for bedrock

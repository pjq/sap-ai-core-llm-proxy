# sap-ai-core LLM Proxy Server

A lightweight Python proxy that exposes SAP AI Core hosted Large Language Models (LLMs) through an OpenAI‑compatible REST interface. It transforms (adapts) SAP AI Core LLM chat / embeddings / streaming APIs (OpenAI, Claude, Gemini variants) into OpenAI Chat Completions + Embeddings formats so that existing OpenAI SDKs, tools and IDE integrations can be used without changes.

Supported client types include (examples):
- [Cursor IDE](https://www.cursor.com/) (currently only accepts gpt-4o via custom endpoint)
- Cline
- Cherry Studio
- Lobe Chat
- Claude Code (via Claude Code Router)
- ChatWise
- ChitChat / ChatCat
- Any OpenAI compatible library or script

> IMPORTANT: You must first deploy the corresponding model(s) in SAP AI Core (see official guide: https://developers.sap.com/tutorials/ai-core-generative-ai.html). Collect each deployment inference URL and place it in `deployment_models` in the configuration.

---
## Contents
1. Features
2. Supported Models & Conversion Logic
3. Architecture & Load Balancing (Multi‑subAccount)
4. Installation & Requirements
5. Configuration (Multi & Legacy Formats)
6. Running the Proxy (HTTP / HTTPS, --config, --debug)
7. Embeddings Endpoint Usage
8. Local Chat CLI (`chat.py`)
9. Load Testing (`load_testing.py`)
10. Integrations (Cursor, Cline, Claude Code Router, Cherry Studio, etc.)
11. Security & Operational Guidance
12. Token Usage Logging (logs/token_usage.log)
13. Troubleshooting
14. Example cURL Requests
15. Roadmap / Planned Enhancements
16. License & Contact

---
## 1. Features
- OpenAI Chat Completions compatible endpoint: `/v1/chat/completions` (streaming & non‑streaming)
- OpenAI Embeddings compatible endpoint: `/v1/embeddings`
- Model listing endpoint: `/v1/models`
- Multi‑subAccount model pooling & round‑robin load balancing (cross‑subAccount + within‑subAccount)
- Automatic model name normalization (removes `anthropic--` prefixes)
- Separate conversion logic pipelines:
  - Claude 3.5 vs Claude 3.7 / 4 (different upstream endpoints & streaming formats)
  - Gemini (request & streaming chunk adaptation)
  - OpenAI family (gpt-* & o* models) with API version handling (`2023-05-15`, `2024-12-01-preview` for o3/o4/o3-mini)
- Streaming translation of upstream Claude/Gemini events into OpenAI SSE chunks
- Token acquisition & caching per subAccount with independent expiry
- Automatic token fetch retry & error handling
- Token usage logging to `logs/token_usage.log` (prompt/completion/total, model, subAccount, IP)
- Debug logging mode (`--debug`) for deep inspection

## 2. Supported Models & Conversion Logic
Categories currently handled (extend by adding to config):
- OpenAI family via SAP AI Core: `gpt-4o`, `gpt-4.1`, `gpt-o3`, `gpt-o3-mini`, `gpt-o4-mini`
- Anthropic Claude: `3.5-sonnet`, `3.7-sonnet`, `4-sonnet` (you may alias names for tools that block the word `claude`)
- Google Gemini: `gemini-2.5-pro`
- Embeddings: (any embedding deployment reachable via the same base inference URLs; exposed through unified `/v1/embeddings`)

Conversion specifics:
- Claude 3.5: Uses legacy `/invoke` and `/invoke-with-response-stream` patterns; payload built with `anthropic_version` & simple message array.
- Claude 3.7 / 4: Uses `/converse` and `/converse-stream`; converts OpenAI messages into block-based structure; streaming events (`messageStart`, `contentBlockDelta`, `messageStop`, etc.) translated to OpenAI `chat.completion.chunk` SSE frames.
- Gemini: Converts OpenAI messages to `generateContent` or `streamGenerateContent` payloads (`contents` + `generation_config` + `safety_settings`); finish reasons mapped to OpenAI; usage metadata converted.
- OpenAI (gpt/o*): For `o3`, `o3-mini`, `o4-mini` the proxy switches to preview API version and removes unsupported params (e.g. `temperature` for some reasoning models).

## 3. Architecture & Load Balancing
- Each subAccount block supplies:
  - `service_key_json` (OAuth credentials)
  - `resource_group`
  - `deployment_models` (model name -> list of deployment inference URLs)
- Normalization: keys like `anthropic--3.7-sonnet` become `3.7-sonnet` so clients use clean model IDs.
- Load Balancing:
  1. Model to subAccount selection (round‑robin across subAccounts that contain the model)
  2. URL selection inside chosen subAccount (round‑robin across listed deployment URLs)
- Failover: If no subAccount has a model, 404 (or fallback to default if configured). Add additional URLs for passive resiliency.
- Independent token caches per subAccount with 5‑minute safety margin before expiry.

## 4. Installation & Requirements
Recommended:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Key dependencies (see `requirements.txt`):
- flask
- requests
- ai_core_sdk (for SAP AI Core interactions if extended)
- openai (used by local CLI)
- litellm (optional future/alternate usage)

## 5. Configuration
Two formats are supported.

### 5.1 Multi‑subAccount (Preferred)
`config.json` (commented pseudo with explanations – not strictly valid JSON):
```jsonc
{
  "subAccounts": {
    "prod-eu": {                           // Logical name
      "resource_group": "default",        // SAP AI Core resource group
      "service_key_json": "prod_key.json",// File containing OAuth credentials
      "deployment_models": {               // Model name -> one or more deployment URLs
        "gpt-4o": ["https://.../deployments/<id1>"],
        "gpt-o3": ["https://.../deployments/<id2>"],
        "3.7-sonnet": ["https://.../deployments/<id3>"]
      }
    },
    "backup-us": {
      "resource_group": "default",
      "service_key_json": "backup_key.json",
      "deployment_models": {
        "gpt-4o": ["https://.../deployments/<id4>", "https://.../deployments/<id5>"],
        "4-sonnet": ["https://.../deployments/<id6>"],
        "gemini-2.5-pro": ["https://.../deployments/<id7>"]
      }
    }
  },
  "secret_authentication_tokens": ["local-dev-key-1", "local-dev-key-2"], // Client bearer tokens accepted by proxy
  "host": "127.0.0.1",                 // Bind host
  "port": 3001                          // Bind port
}
```

Valid JSON example (copy/adapt):
```json
{
  "subAccounts": {
    "subAccount1": {
      "resource_group": "default",
      "service_key_json": "demokey1.json",
      "deployment_models": {
        "gpt-4o": ["https://api.ai.../deployments/<hidden_id_1>"],
        "gpt-4.1": ["https://api.ai.../deployments/<hidden_id_1b>"],
        "3.5-sonnet": ["https://api.ai.../deployments/<hidden_id_2>"]
      }
    },
    "subAccount2": {
      "resource_group": "default",
      "service_key_json": "demokey2.json",
      "deployment_models": {
        "gpt-4o": ["https://api.ai.../deployments/<hidden_id_3>"],
        "3.7-sonnet": ["https://api.ai.../deployments/<hidden_id_4>"],
        "4-sonnet": ["https://api.ai.../deployments/<hidden_id_5>"],
        "gemini-2.5-pro": ["https://api.ai.../deployments/<hidden_id_6>"]
      }
    }
  },
  "secret_authentication_tokens": ["<hidden_key_1>", "<hidden_key_2>"],
  "port": 3001,
  "host": "127.0.0.1"
}
```

### 5.2 Legacy (Single SubAccount)
Still accepted for backward compatibility (fields: `service_key_json`, `deployment_models`, `resource_group`, `secret_authentication_tokens`, `host`, `port`). Model name normalization (removing `anthropic--`) is applied automatically.

### 5.3 Service Key File
Example (`prod_key.json`):
```json
{
  "serviceurls": { "AI_API_URL": "https://api.ai.<region>.<provider>.ml.hana.ondemand.com" },
  "clientid": "<client id>",
  "clientsecret": "<client secret>",
  "identityzoneid": "<zone id>",
  "url": "https://<uaa-domain>/oauth/token"
}
```
Only the subset actually used: `clientid`, `clientsecret`, `url`, `identityzoneid`.

### 5.4 Model Naming Guidance
If a tool blocks `claude` explicitly (e.g., some Cursor contexts), you can alias the model in `deployment_models` as `3.7-sonnet` or `sonnet` without the substring.

## 6. Running the Proxy

### 6.1 Basic
```bash
python proxy_server.py --config config.json
```
Outputs base URL & available models.

### 6.2 Debug Mode
```bash
python proxy_server.py --config config.json --debug
```
Enables verbose logging of conversions and streaming chunk processing.

### 6.3 HTTPS (Self‑Signed Example)
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
# Adjust proxy_server.py or run behind nginx/caddy for TLS termination.
```
Run the app (optionally modify run stanza to include `ssl_context=('cert.pem','key.pem')`). For production prefer a reverse proxy (nginx, Traefik, Caddy) terminating HTTPS and forwarding to the Flask app over localhost.

### 6.4 Configuration Flags
- `--config <path>` specify non‑default JSON
- `--debug` sets global log level DEBUG

### 6.5 Planned (Environment Overrides)
Future enhancement: allow `PROXY_HOST` / `PROXY_PORT` environment variables to override config values (not yet implemented at code time of this README).

## 7. Embeddings Endpoint
Endpoint: `POST /v1/embeddings`
OpenAI‑compatible request:
```bash
curl -X POST http://127.0.0.1:3001/v1/embeddings \
  -H "Authorization: Bearer <your-proxy-token>" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-3-large","input":"Hello embeddings"}'
```
Response format (OpenAI style):
```json
{
  "object": "list",
  "data": [ { "object": "embedding", "embedding": [0.01, ...], "index": 0 } ],
  "model": "text-embedding-3-large",
  "usage": { "prompt_tokens": <n>, "total_tokens": <n> }
}
```
Multiple inputs: provide an array in `input` (the proxy forwards appropriately). Encoding format parameter is accepted and passed/ignored based on backend capability.

## 8. Local Chat CLI (`chat.py`)
Simple REPL using the OpenAI Python SDK pointed at the proxy.
```bash
python chat.py --model gpt-4o
```
Requirements:
- `config.json` must exist and contain `secret_authentication_tokens`; the first token is used automatically.
- Adjust `openai.base_url` in `chat.py` if running remote.
Exit with `exit`.

## 9. Load Testing (`load_testing.py`)
Script issues concurrent requests to measure latency & success rates.
Example:
```bash
python load_testing.py            # uses defaults at end of file
# or modify call:
# load_test(num_threads=20, total_requests=200, endpoint="chat/completions", model="gpt-4o")
```
Metrics logged:
- Total time & requests per second
- Success rate (% 200 responses)
- Status code distribution
- Min / Max / Avg / Median response time
- Sample content validation (verifies expected short reply if configured)
Interpretation:
- Low success rate: check token errors or model availability.
- High max vs median: indicates tail latency (possible cold starts or throttling).
You can comment/uncomment the last line in the script to target different models.

## 10. Integrations
### 10.1 Model Discovery
Tools often query `GET /v1/models`. This returns normalized IDs (without `anthropic--`). Use these IDs in tool configuration. Example browse: http://127.0.0.1:3001/v1/models

### 10.2 Common Settings
- Base URL: `http://<host>:<port>/v1`
- API Key: one value from `secret_authentication_tokens`
- Model ID: as returned by `/v1/models`

### 10.3 Cursor IDE
- Only `gpt-4o` currently accepted for custom endpoints per latest forum guidance.
- Set Custom Base URL to the proxy; use a proxy token as API key.
Reference: https://forum.cursor.com/t/custom-api-keys-fail-with-the-model-does-not-work-with-your-current-plan-or-api-key/97422

### 10.4 Cline
Provider: OpenAI API Compatible
- Base URL: `http://127.0.0.1:3001/v1`
- API Key: proxy token
- Model: e.g. `4-sonnet` or `gpt-4o`
Cline now has official SAP AI Core support; proxy remains useful for unified multi‑model routing.

### 10.5 Claude Code Router
Install & configure:
```bash
npm install -g @anthropic-ai/claude-code
npm install -g @musistudio/claude-code-router
ccr code
```
Sample `~/.claude-code-router/config.json`:
```json
{
  "OPENAI_API_KEY": "<proxy-token>",
  "OPENAI_BASE_URL": "http://127.0.0.1:3001/v1",
  "OPENAI_MODEL": "3.7-sonnet",
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "http://127.0.0.1:3001/v1",
      "api_key": "<proxy-token>",
      "models": ["gpt-4o", "3.7-sonnet", "4-sonnet"]
    }
  ],
  "Router": {
    "background": "gpt-4o",
    "think": "deepseek,deepseek-reasoner",
    "longContext": "openrouter,3.7-sonnet"
  }
}
```

### 10.6 Cherry Studio
Add Provider -> Type: OpenAI
- API Host: `http://127.0.0.1:3001`
- API Key: proxy token
- Add models manually (`/v1/models` list)
If `claude` is blocked, use `3.7-sonnet` alias.

### 10.7 Additional Tools
- Lobe Chat, ChatWise, other OpenAI SDK based tools: configure like standard OpenAI; supply base URL & token.

## 11. Security & Operational Guidance
- Authentication: Proxy enforces simple bearer token match against `secret_authentication_tokens`. Use long, random tokens. Consider storing outside source control.
- Upstream OAuth: Each subAccount’s SAP AI Core token is fetched via client credentials and cached until near expiry.
- HTTPS: Prefer terminating TLS at a reverse proxy. Self‑signed dev cert generation example shown above.
- Logging: Debug logs may include partial payloads. Avoid enabling `--debug` in production unless diagnosing issues.
- Token Leakage: The proxy truncates Authorization header when logging usage. Do not log full secrets.
- Hardening (recommended if exposed):
  - IP allowlist or auth layer in front
  - Rate limiting (e.g., nginx `limit_req` or a WAF)
  - Rotate `secret_authentication_tokens` periodically
- Least Privilege: Service key files should have restricted filesystem permissions.

## 12. Token Usage Logging
File: `logs/token_usage.log`
Format (example):
```
2025-08-08T10:00:00Z User: Bearer abcdef... IP: 127.0.0.1 Model: gpt-4o SubAccount: prod-eu PromptTokens: 12 CompletionTokens: 25 TotalTokens: 37
```
Streaming entries show `(Streaming)` suffix. Use this log for metering or cost attribution. (Estimation for some streaming cases if upstream does not send counts in-chunk.)

## 13. Troubleshooting
| Symptom | Cause | Resolution |
|---------|-------|-----------|
| 401 Unauthorized | Missing/invalid proxy token | Send `Authorization: Bearer <one-of-secret_authentication_tokens>` |
| 404 Model not found | Model key not in any subAccount mapping | Check `/v1/models`; add deployment; ensure normalized name (remove `anthropic--`) |
| Token fetch timeout | Network / auth endpoint slow | Retry; verify service key `url`; increase network reliability |
| Empty or malformed streaming chunks | Upstream event format changed or parsing error | Run with `--debug`; inspect logs; open issue with raw chunk sample |
| Conversion error (Claude 3.7/4) | Unexpected structure (e.g., tool use blocks) | Update converter to handle new block types; temporary fallback: disable streaming (`"stream": false`) |
| Gemini response missing text | Non-text part returned first | Extend converter to search additional parts (already attempts search) |
| Proxy starts but no models | Mis-typed JSON or missing `deployment_models` | Validate JSON with lint; ensure lists of URLs |
| Embeddings returns 500 | No `input` field or backend error | Provide `input`; verify embedding deployment; check logs |

If unresolved, enable `--debug` and review stack traces.

## 14. Example cURL Requests
Non‑streaming chat:
```bash
curl -X POST http://127.0.0.1:3001/v1/chat/completions \
  -H "Authorization: Bearer <proxy-token>" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o",
        "messages": [{"role":"user","content":"Say hi in one word"}],
        "max_tokens": 20,
        "stream": false
      }'
```
Streaming chat (watch incremental deltas):
```bash
curl -N -X POST http://127.0.0.1:3001/v1/chat/completions \
  -H "Authorization: Bearer <proxy-token>" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "3.7-sonnet",
        "messages": [{"role":"user","content":"Explain SSE in one sentence"}],
        "stream": true
      }'
```
List models:
```bash
curl -H "Authorization: Bearer <proxy-token>" http://127.0.0.1:3001/v1/models
```
Embeddings (see Section 7).

## 15. Roadmap / Planned Enhancements
- Environment variable overrides for host/port (PROXY_HOST / PROXY_PORT)
- Optional rate limiting & API quota middleware
- Extended tool / function calling translation for Claude & Gemini
- Enhanced multi‑tenant analytics dashboard reading `token_usage.log`

## 16. License & Contact
### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contact
Questions / issues: [pengjianqing@gmail.com]

---
Happy building! If you encounter new upstream response shapes (especially for Claude 3.7/4 or Gemini streaming), please contribute conversion improvements.

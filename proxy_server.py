import logging
from flask import Flask, request, jsonify, Response, stream_with_context
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import threading
import json
import base64
import random
import os
import uuid
from datetime import datetime, timedelta
import shutil
import argparse
import re
import ast
import atexit
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from werkzeug.serving import make_server
from concurrent.futures import ThreadPoolExecutor

# SAP AI SDK imports
from gen_ai_hub.proxy.core.utils import kwargs_if_set
from gen_ai_hub.proxy.native.amazon.clients import Session

# Monkey patch to fix client_type compatibility issue between gen_ai_hub and ai_core_sdk
# The gen_ai_hub library passes 'client_type' parameter that ai_core_sdk 2.6.2 doesn't accept
from ai_core_sdk.ai_core_v2_client import AICoreV2Client
_original_from_env = AICoreV2Client.from_env

@staticmethod
def _patched_from_env(profile_name: str = None, **kwargs):
    """Patched version of from_env that filters out unsupported 'client_type' parameter."""
    # Remove client_type from kwargs as it's not supported in ai_core_sdk 2.6.2
    kwargs.pop('client_type', None)
    return _original_from_env(profile_name=profile_name, **kwargs)

AICoreV2Client.from_env = _patched_from_env



@dataclass
class ServiceKey:
    clientid: str
    clientsecret: str
    url: str
    identityzoneid: str

@dataclass
class TokenInfo:
    token: Optional[str] = None
    expiry: float = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

@dataclass
class SubAccountConfig:
    name: str
    resource_group: str
    service_key_json: str
    deployment_models: Dict[str, List[str]]
    service_key: Optional[ServiceKey] = None
    token_info: TokenInfo = field(default_factory=TokenInfo)
    normalized_models: Dict[str, List[str]] = field(default_factory=dict)
    
    def load_service_key(self):
        """Load service key from file"""
        key_data = load_config(self.service_key_json)
        self.service_key = ServiceKey(
            clientid=key_data.get('clientid'),
            clientsecret=key_data.get('clientsecret'),
            url=key_data.get('url'),
            identityzoneid=key_data.get('identityzoneid')
        )
        
    def normalize_model_names(self):
        """Keep model names as-is without removing any prefixes"""
        self.normalized_models = self.deployment_models.copy()

@dataclass
class ProxyConfig:
    subaccounts: Dict[str, SubAccountConfig] = field(default_factory=dict)
    secret_authentication_tokens: List[str] = field(default_factory=list)
    port: int = 3001
    host: str = "127.0.0.1"
    # Global model to subaccount mapping for load balancing
    model_to_subaccounts: Dict[str, List[str]] = field(default_factory=dict)
    
    def initialize(self):
        """Initialize all subaccounts and build model mappings"""
        for subaccount in self.subaccounts.values():
            subaccount.load_service_key()
            subaccount.normalize_model_names()
            
        # Build model to subaccounts mapping for load balancing
        self.build_model_mapping()
    
    def build_model_mapping(self):
        """Build a mapping of models to the subaccounts that have them"""
        self.model_to_subaccounts = {}
        for subaccount_name, subaccount in self.subaccounts.items():
            for model in subaccount.normalized_models.keys():
                if model not in self.model_to_subaccounts:
                    self.model_to_subaccounts[model] = []
                self.model_to_subaccounts[model].append(subaccount_name)


# Global configuration
proxy_config = ProxyConfig()


class _LazyJSON:
    """Defer json.dumps until the log record is actually formatted.

    Passed as a %-style logging arg, its __str__ runs only when the record
    passes the level filter and is emitted. This keeps expensive pretty-print
    serialization of request/response payloads out of the hot path when DEBUG
    logging is disabled (the common production case).
    """
    __slots__ = ("obj", "kwargs")

    def __init__(self, obj, **kwargs):
        self.obj = obj
        self.kwargs = kwargs

    def __str__(self):
        try:
            return json.dumps(self.obj, **self.kwargs)
        except Exception:
            return repr(self.obj)


def _format_bytes(num_bytes):
    """Human-readable byte size, e.g. 512 B, 3.4 KB, 1.2 MB."""
    if num_bytes is None:
        return "unknown"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def request_payload_size(req):
    """Return the raw request payload size in bytes.

    Prefers the Content-Length header (bytes on the wire, no extra work). Falls
    back to measuring the buffered body if the header is absent (e.g. chunked
    transfer). Returns None if the size can't be determined.
    """
    length = req.content_length
    if length is not None:
        return length
    try:
        return len(req.get_data(cache=True))
    except Exception:
        return None


# ------------------------
# Token counting (estimate)
# ------------------------
# SAP AI Core's Bedrock backend exposes no native count_tokens endpoint, so the
# /v1/messages/count_tokens route returns an estimate. We use tiktoken (already a
# dependency) as the counter. Note: tiktoken is OpenAI's tokenizer and undercounts
# Claude by ~15-20% (more on code); this is a best-effort estimate, not exact.
_token_encoder = None
_token_encoder_lock = threading.Lock()


def _get_token_encoder():
    """Lazily load and cache a tiktoken encoder (cl100k_base)."""
    global _token_encoder
    if _token_encoder is None:
        with _token_encoder_lock:
            if _token_encoder is None:
                import tiktoken
                _token_encoder = tiktoken.get_encoding("cl100k_base")
    return _token_encoder


def _flatten_content_to_text(content):
    """Flatten Anthropic message content (str or list of blocks) into plain text
    for token estimation. Pulls text out of text/tool_result blocks and stringifies
    the rest so structural tokens are still roughly accounted for."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    parts.append(_flatten_content_to_text(block.get("content")))
                elif block.get("type") == "tool_use":
                    # name + serialized input approximates the tokens the model sees
                    parts.append(str(block.get("name", "")))
                    try:
                        parts.append(json.dumps(block.get("input", {})))
                    except Exception:
                        parts.append(str(block.get("input", "")))
                else:
                    try:
                        parts.append(json.dumps(block))
                    except Exception:
                        parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def estimate_input_tokens(request_json):
    """Estimate input tokens for a request body, best-effort and schema-tolerant.

    Handles the fields used across all the proxy's inbound shapes:
      - Anthropic Messages: system, messages, tools (input_schema)
      - OpenAI chat/completions: messages, tools (function.parameters)
      - OpenAI Responses: instructions, input (string or list), tools
      - Embeddings: input (string or list)
    Counts the concatenation with tiktoken. Returns an int (approximation —
    tiktoken is OpenAI's tokenizer, so Claude counts are undercounted ~15-20%)."""
    if not isinstance(request_json, dict):
        return 0

    encoder = _get_token_encoder()
    segments = []

    system = request_json.get("system")
    if system is not None:
        segments.append(_flatten_content_to_text(system))

    instructions = request_json.get("instructions")
    if isinstance(instructions, str):
        segments.append(instructions)

    for message in request_json.get("messages", []) or []:
        if isinstance(message, dict):
            segments.append(str(message.get("role", "")))
            segments.append(_flatten_content_to_text(message.get("content")))

    # Responses API `input` / embeddings `input` (string, or list of items/blocks).
    input_field = request_json.get("input")
    if isinstance(input_field, str):
        segments.append(input_field)
    elif isinstance(input_field, list):
        for item in input_field:
            if isinstance(item, str):
                segments.append(item)
            elif isinstance(item, dict):
                # Responses input items carry text under "content"; fall back to whole dict.
                if "content" in item or "text" in item:
                    segments.append(_flatten_content_to_text(item.get("content", item.get("text"))))
                else:
                    segments.append(_flatten_content_to_text(item))

    tools = request_json.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            # Anthropic tool shape
            segments.append(str(tool.get("name", "")))
            segments.append(str(tool.get("description", "")))
            schema = tool.get("input_schema")
            # OpenAI tool shape: {"type":"function","function":{name,description,parameters}}
            fn = tool.get("function")
            if isinstance(fn, dict):
                segments.append(str(fn.get("name", "")))
                segments.append(str(fn.get("description", "")))
                schema = schema if schema is not None else fn.get("parameters")
            if schema is not None:
                try:
                    segments.append(json.dumps(schema))
                except Exception:
                    segments.append(str(schema))

    text = "\n".join(s for s in segments if s)
    return len(encoder.encode(text))


def request_token_estimate(req):
    """Best-effort input-token estimate for the current request, for logging.

    Never raises — returns None if the body can't be parsed or estimated, so it
    can be dropped straight into a log line alongside the byte size."""
    try:
        body = req.get_json(silent=True, cache=True)
        if not body:
            return None
        return estimate_input_tokens(body)
    except Exception:
        return None


# ------------------------
# HTTP Session with Connection Pool Management
# ------------------------
def create_http_session():
    """Create a requests Session with connection pooling and proper cleanup"""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    # Configure adapter with increased connection pool limits to handle concurrent requests
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=50,      # Increased: Number of connection pools (was 10)
        pool_maxsize=100,         # Increased: Max connections per pool (was 20)
        pool_block=False          # Don't block when pool is full
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# Create global session (reuse across requests)
_http_session = create_http_session()

app = Flask(__name__)

# Configure Flask to close connections properly
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['JSON_SORT_KEYS'] = False

@app.after_request
def after_request_cleanup(response):
    """Cleanup after each request"""
    # Note: Don't set Connection header - it's a "hop-by-hop" header
    # that WSGI servers (like Waitress) manage themselves per PEP 3333
    # Waitress will handle connection lifecycle properly

    # Force close connection if response is large or if client requests it
    if request.headers.get('Connection', '').lower() == 'close':
        response.headers['Connection'] = 'close'

    return response

# ------------------------
# SAP AI SDK session/client cache for performance
# ------------------------
# Creating a new SDK Session()/client per request is expensive. Reuse a process-wide
# Session and cache clients per model in a thread-safe manner.
_sdk_session = None
_sdk_session_lock = threading.Lock()
_bedrock_clients: Dict[str, Any] = {}
_bedrock_clients_lock = threading.Lock()

def get_sapaicore_sdk_session() -> Session:
    """Lazily initialize and return a global SAP AI Core SDK Session."""
    global _sdk_session
    if _sdk_session is None:
        with _sdk_session_lock:
            if _sdk_session is None:
                logging.info("Initializing global SAP AI SDK Session")
                _sdk_session = Session()
    return _sdk_session

def get_sapaicore_sdk_client(model_name: str):
    """Get or create a cached SAP AI Core (Bedrock) client for the given model."""
    client = _bedrock_clients.get(model_name)
    if client is not None:
        return client
    with _bedrock_clients_lock:
        client = _bedrock_clients.get(model_name)
        if client is None:
            logging.info(f"Creating SAP AI SDK client for model '{model_name}'")
            client = get_sapaicore_sdk_session().client(model_name=model_name)
            _bedrock_clients[model_name] = client
    return client

@app.route('/v1/embeddings', methods=['POST'])
def handle_embedding_request():
    _size = request_payload_size(request)
    logging.info("Received request to /v1/embeddings (payload size: %s / %s bytes, ~%s tokens)",
                 _format_bytes(_size), _size, request_token_estimate(request))
    if not verify_request_token(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    payload = request.json
    input_text = payload.get("input")
    model = payload.get("model", "text-embedding-3-large")
    encoding_format = payload.get("encoding_format")

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    try:
        endpoint_url, modified_payload, subaccount_name = handle_embedding_service_call(input_text, model, encoding_format)
        subaccount_token = fetch_token(subaccount_name)
        subaccount = proxy_config.subaccounts[subaccount_name]
        resource_group = subaccount.resource_group
        service_key = subaccount.service_key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {subaccount_token}",
            "AI-Resource-Group": resource_group,
            "AI-Tenant-Id": service_key.identityzoneid
        }
        response = _http_session.post(endpoint_url, headers=headers, json=modified_payload)
        response.raise_for_status()
        result = jsonify(format_embedding_response(response.json(), model)), 200
        response.close()
        return result
    except Exception as e:
        logging.error(f"Error handling embedding request: {e}")
        return jsonify({"error": str(e)}), 500

def handle_embedding_service_call(input_text, model, encoding_format):
    # Logic to prepare the request to SAP AI Core
    selected_url, subaccount_name, _, model = load_balance_url(model)
    
    # Construct the URL based on the official SAP AI Core documentation
    api_version = "2023-05-15"
    endpoint_url = f"{selected_url.rstrip('/')}/embeddings?api-version={api_version}"
    
    # The payload for the embeddings endpoint only requires the input.
    modified_payload = {"input": input_text}
        
    return endpoint_url, modified_payload, subaccount_name

def format_embedding_response(response, model):
    # Logic to convert the response to OpenAI format
    # Process all embeddings in the response data list
    data_list = []
    total_embedding_length = 0

    for item in response.get("data", []):
        embedding_data = item.get("embedding", [])
        index = item.get("index", len(data_list))

        data_list.append(
            {"object": "embedding", "embedding": embedding_data, "index": index}
        )

        total_embedding_length += len(embedding_data)

    return {
        "object": "list",
        "data": data_list,
        "model": model,
        "usage": {
            "prompt_tokens": total_embedding_length,
            "total_tokens": total_embedding_length,
        },
    }

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a new logger for token usage
token_logger = logging.getLogger('token_usage')
token_logger.setLevel(logging.INFO)

# Create a file handler for token usage logging
log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_file = os.path.join(log_directory, 'token_usage.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter for token usage logging
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the token usage logger
token_logger.addHandler(file_handler)

# Prevent token_logger from propagating to root logger (avoid duplicate logs)
token_logger.propagate = False

# --- Request/Response File Logger ---
# Enabled via --log-requests CLI flag. When enabled, writes JSON files capturing
# the full 4-stage lifecycle of each request to logs/requests/YYYY-MM-DD/.
_log_requests_enabled = False


class RequestFileLogger:
    """Thread-safe file-based request/response lifecycle logger.

    Writes JSON payloads to per-day subdirectories under logs/requests/.
    Zero-cost when disabled: the first check in log() is a simple boolean test.
    Thread-safe by design: each file has a unique name (timestamp + request_id),
    so concurrent writes never collide. Only directory creation needs a lock.
    """

    BASE_DIR = os.path.join("logs", "requests")
    _dir_cache = set()
    _dir_cache_lock = threading.Lock()

    @classmethod
    def _ensure_dir(cls, date_str):
        """Create date subdirectory if not yet created this session (thread-safe, cached)."""
        if date_str in cls._dir_cache:
            return
        with cls._dir_cache_lock:
            if date_str not in cls._dir_cache:
                dir_path = os.path.join(cls.BASE_DIR, date_str)
                os.makedirs(dir_path, exist_ok=True)
                cls._dir_cache.add(date_str)

    @classmethod
    def log(cls, request_id, model, stage_num, stage_name, payload, agent=None):
        """Write a single stage file for a request lifecycle.

        Args:
            request_id: Unique request identifier (e.g. "1721623425482-3847")
            model: Model name string
            stage_num: 1-4 integer indicating lifecycle stage
            stage_name: One of "client_request", "transformed_request",
                        "backend_response", "client_response"
            payload: dict or string to serialize as JSON
            agent: Optional agent/client identifier (e.g. "ClaudeCode", "Pi", "Codex")
        """
        if not _log_requests_enabled:
            return

        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H%M%S")
            ms_str = f"{now.microsecond // 1000:03d}"

            # Sanitize model name for filesystem — strip common prefixes and keep the useful part
            raw_model = model or "unknown"
            # Strip "anthropic--" prefix (SAP AI Core naming convention)
            if raw_model.startswith("anthropic--"):
                raw_model = raw_model[len("anthropic--"):]
            model_short = re.sub(r'[^\w\-.]', '_', raw_model)[:25]

            # Use last 8 chars of request_id as short correlation key
            req_id_short = (request_id or "unknown")[-8:]

            # Sanitize agent name for filesystem
            agent_part = re.sub(r'[^\w\-.]', '_', agent or "unknown")[:15]

            cls._ensure_dir(date_str)

            filename = f"{time_str}_{ms_str}_{req_id_short}_{agent_part}_{model_short}_{stage_num}_{stage_name}.json"
            filepath = os.path.join(cls.BASE_DIR, date_str, filename)

            # Serialize and write
            if isinstance(payload, str):
                content = payload
            else:
                content = json.dumps(payload, indent=2, ensure_ascii=False, default=str)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        except Exception as e:
            # Never let logging failures crash the proxy
            logging.warning("RequestFileLogger.log failed: %s", e)


def _cleanup_request_logs():
    """Delete request log directories older than 7 days. Runs every 6 hours as a daemon thread."""
    while True:
        time.sleep(6 * 3600)
        if not _log_requests_enabled:
            continue
        try:
            base = RequestFileLogger.BASE_DIR
            if not os.path.exists(base):
                continue
            cutoff = datetime.now() - timedelta(days=7)
            for entry in os.listdir(base):
                try:
                    dir_date = datetime.strptime(entry, "%Y-%m-%d")
                    if dir_date < cutoff:
                        shutil.rmtree(os.path.join(base, entry))
                        logging.info("Cleaned up old request logs: %s", entry)
                except ValueError:
                    pass  # Not a date-formatted directory name
        except Exception as e:
            logging.warning("Request log cleanup error: %s", e)


def _assemble_stream_response(chunks, model, prompt_tokens=0, completion_tokens=0, total_tokens=0):
    """Assemble streaming chunks into a single response object matching non-streaming format.

    Extracts content from OpenAI-style or Claude-style chunks and builds a clean
    response dict (same shape as a non-streaming response).

    Args:
        chunks: List of chunk dicts or raw SSE strings
        model: Model name
        prompt_tokens: Prompt token count
        completion_tokens: Completion token count
        total_tokens: Total token count

    Returns:
        A dict resembling a non-streaming response with assembled content
    """
    content_parts = []
    role = "assistant"
    finish_reason = None
    response_id = None

    for chunk in chunks:
        # Handle raw SSE string chunks (OpenAI pass-through)
        if isinstance(chunk, str):
            # Parse SSE lines like "data: {...}\n\n"
            for line in chunk.strip().split('\n'):
                line = line.strip()
                if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                    try:
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        if not response_id:
                            response_id = data.get("id")
                        fr = data.get("choices", [{}])[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
            continue

        if not isinstance(chunk, dict):
            continue

        # OpenAI-format chunks (from Claude/Gemini converted to OpenAI)
        if "choices" in chunk:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            if not response_id:
                response_id = chunk.get("id")
            fr = chunk.get("choices", [{}])[0].get("finish_reason")
            if fr:
                finish_reason = fr

        # Claude-native chunks (from /v1/messages Bedrock path)
        chunk_type = chunk.get("type")
        if chunk_type == "content_block_delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                content_parts.append(delta.get("text", ""))
            elif delta.get("type") == "thinking_delta":
                content_parts.append(delta.get("thinking", ""))
        elif chunk_type == "message_start":
            msg = chunk.get("message", {})
            if not response_id:
                response_id = msg.get("id")
            role = msg.get("role", role)
        elif chunk_type == "message_delta":
            stop = chunk.get("delta", {}).get("stop_reason")
            if stop:
                finish_reason = stop

    assembled_content = "".join(content_parts)

    return {
        "id": response_id or "stream-assembled",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": role,
                "content": assembled_content
            },
            "finish_reason": finish_reason or "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        },
        "_streaming": True,
        "_chunk_count": len(chunks)
    }


def _detect_agent(flask_request):
    """Detect the coding agent/client from request headers or metadata.

    Checks User-Agent header, known client-specific headers, and request body
    hints to identify the source tool (Claude Code, Pi, Codex, Cursor, Aider, etc.).

    Args:
        flask_request: The Flask request object

    Returns:
        Short agent identifier string (e.g. "ClaudeCode", "Pi", "Codex")
    """
    user_agent = flask_request.headers.get("User-Agent", "").lower()
    # x-client-name is used by some OpenAI SDK clients
    x_client = flask_request.headers.get("x-client-name", "").lower()

    # Claude Code
    if "claude-code" in user_agent or "claudecode" in user_agent:
        return "ClaudeCode"

    # Pi (coding agent TUI) — checks User-Agent and x-client-name
    if "pi/" in user_agent or "pi-agent" in user_agent or "pi-coding" in user_agent:
        return "Pi"

    # Codex CLI (OpenAI)
    if "codex" in user_agent or "codex" in x_client:
        return "Codex"

    # Cursor
    if "cursor" in user_agent or "cursor" in x_client:
        return "Cursor"

    # Windsurf / Codeium
    if "windsurf" in user_agent or "codeium" in user_agent:
        return "Windsurf"

    # Continue.dev
    if "continue" in user_agent:
        return "Continue"

    # Aider
    if "aider" in user_agent:
        return "Aider"

    # Cline / Roo Code
    if "cline" in user_agent or "roo" in user_agent:
        return "Cline"

    # Copilot
    if "copilot" in user_agent or "github-copilot" in user_agent:
        return "Copilot"

    # --- Fallback: inspect request body for agent-specific tool/system patterns ---
    try:
        body = flask_request.get_json(silent=True, cache=True)
        if body:
            # Check system message or first message content for agent signatures
            messages = body.get("messages", [])
            # Look at the system message (first message or system field)
            system_text = ""
            if isinstance(body.get("system"), str):
                system_text = body["system"][:500].lower()
            elif isinstance(body.get("system"), list):
                system_text = str(body["system"][:2])[:500].lower()
            if messages and isinstance(messages, list):
                first_msg = messages[0] if messages else {}
                if first_msg.get("role") == "system":
                    content = first_msg.get("content", "")
                    if isinstance(content, str):
                        system_text += content[:500].lower()

            # Pi signatures
            if "pi, a coding agent" in system_text or "pi-coding-agent" in system_text:
                return "Pi"

            # Claude Code signatures
            if "claude code" in system_text or "anthropic's official cli" in system_text:
                return "ClaudeCode"

            # Codex CLI signatures
            if "codex" in system_text and "openai" in system_text:
                return "Codex"

            # Check tool names for agent-specific patterns
            tools = body.get("tools", [])
            if tools and isinstance(tools, list):
                tool_names = set()
                for t in tools[:20]:  # Only check first 20 tools
                    fn = t.get("function", {}) if isinstance(t, dict) else {}
                    name = fn.get("name", "").lower()
                    if name:
                        tool_names.add(name)

                # Pi tool signatures
                if "observe_ui" in tool_names or "act_ui" in tool_names or "expand_ui" in tool_names:
                    return "Pi"

                # Claude Code tool signatures
                if "todowrite" in tool_names or "todoread" in tool_names:
                    return "ClaudeCode"

                # Codex CLI tool signatures
                if "shell" in tool_names and "apply_patch" in tool_names:
                    return "Codex"

                # Cursor tool signatures
                if "codebase_search" in tool_names or "run_terminal_cmd" in tool_names:
                    return "Cursor"

    except Exception:
        pass

    # Generic OpenAI SDK clients
    if "openai" in user_agent:
        return "OpenAI-SDK"

    # Anthropic SDK
    if "anthropic" in user_agent:
        return "Anthropic-SDK"

    # Check x-client-name as fallback
    if x_client:
        return x_client[:15]

    return "unknown"


# Helper function to log token usage in plain text format
def log_token_usage(model, subaccount_name, prompt_tokens, completion_tokens,
                    total_tokens, user_id=None, ip_address=None, is_streaming=False,
                    duration_ms=None, request_id=None, error=None):
    """Log token usage in plain text format.

    Args:
        model: Model name used
        subaccount_name: SubAccount that handled the request
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
        user_id: User/auth token (already truncated)
        ip_address: Client IP address
        is_streaming: Whether this was a streaming request
        duration_ms: Request duration in milliseconds
        request_id: Unique request identifier (not logged, kept for compatibility)
        error: Error message if request failed
    """
    # Use provided user_id and ip_address, or defaults
    if user_id is None:
        user_id = "unknown"
    if ip_address is None:
        ip_address = "unknown"

    # Build log message in original format with optional enhancements
    log_parts = [
        f"User: {user_id}",
        f"IP: {ip_address}",
        f"Model: {model}",
        f"SubAccount: {subaccount_name}",
        f"PromptTokens: {prompt_tokens}",
        f"CompletionTokens: {completion_tokens}",
        f"TotalTokens: {total_tokens}"
    ]

    # Add optional duration
    if duration_ms is not None:
        log_parts.append(f"Duration: {duration_ms}ms")

    # Add streaming indicator
    if is_streaming:
        log_parts.append("(Streaming)")

    # Add error if present
    if error:
        log_parts.append(f"Error: {error}")

    # Log in original comma-separated format
    token_logger.info(", ".join(log_parts))

# Global variables for token management
token = None
token_expiry = 0
lock = threading.Lock()

def load_config(file_path):
    """Loads configuration from a JSON file with support for multiple subAccounts."""
    with open(file_path, 'r') as file:
        config_json = json.load(file)
    
    # Check if this is the new format with subAccounts
    if 'subAccounts' in config_json:
        # Create a proper ProxyConfig instance
        proxy_conf = ProxyConfig(
            secret_authentication_tokens=config_json.get('secret_authentication_tokens', []),
            port=config_json.get('port', 3001),
            host=config_json.get('host', '127.0.0.1')
        )
        
        # Parse each subAccount
        for sub_name, sub_config in config_json.get('subAccounts', {}).items():
            proxy_conf.subaccounts[sub_name] = SubAccountConfig(
                name=sub_name,
                resource_group=sub_config.get('resource_group', 'default'),
                service_key_json=sub_config.get('service_key_json', ''),
                deployment_models=sub_config.get('deployment_models', {})
            )
        
        return proxy_conf
    else:
        # For backward compatibility - return the raw JSON
        return config_json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Proxy server for AI models")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-requests", action="store_true",
                        help="Enable file-based logging of full request/response payloads to logs/requests/")
    return parser.parse_args()

def fetch_token(subaccount_name: str) -> str:
    """Fetches or retrieves a cached SAP AI Core authentication token for a specific subAccount.
    
    Args:
        subaccount_name: Name of the subAccount to fetch token for
        
    Returns:
        The authentication token
        
    Raises:
        ValueError: If subaccount is not found or service key is missing
        ConnectionError: If there's a network issue during token fetch
    """
    if subaccount_name not in proxy_config.subaccounts:
        raise ValueError(f"SubAccount '{subaccount_name}' not found in configuration")
    
    subaccount = proxy_config.subaccounts[subaccount_name]
    if not subaccount.service_key:
        raise ValueError(f"Service key not loaded for subAccount '{subaccount_name}'")
    
    with subaccount.token_info.lock:
        now = time.time()
        # Return cached token if still valid
        if subaccount.token_info.token and now < subaccount.token_info.expiry:
            logging.info(f"Using cached token for subAccount '{subaccount_name}'.")
            return subaccount.token_info.token

        logging.info(f"Fetching new token for subAccount '{subaccount_name}'.")

        # Build auth header with Base64 encoded clientid:clientsecret
        service_key = subaccount.service_key
        auth_string = f"{service_key.clientid}:{service_key.clientsecret}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        # Build token endpoint URL and headers
        token_url = f"{service_key.url}/oauth/token?grant_type=client_credentials"
        headers = {"Authorization": f"Basic {encoded_auth}"}

        try:
            response = _http_session.post(token_url, headers=headers, timeout=15)
            response.raise_for_status()

            token_data = response.json()
            new_token = token_data.get('access_token')

            # Explicitly close the response to release connection
            response.close()
            
            # Check for empty token
            if not new_token:
                raise ValueError("Fetched token is empty")
            
            # Calculate expiry (use expires_in from response, default to 4 hours, with 5-minute buffer)
            expires_in = int(token_data.get('expires_in', 14400))
            subaccount.token_info.token = new_token
            subaccount.token_info.expiry = now + expires_in - 300  # 5-minute buffer
            
            logging.info(f"Token fetched successfully for subAccount '{subaccount_name}'.")
            return new_token
        
        except requests.exceptions.Timeout as err:
            logging.error(f"Timeout fetching token for '{subaccount_name}': {err}")
            subaccount.token_info.token = None
            subaccount.token_info.expiry = 0
            raise TimeoutError(f"Timeout connecting token endpoint for '{subaccount_name}'") from err
            
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP error fetching token for '{subaccount_name}': {err.response.status_code}-{err.response.text}")
            subaccount.token_info.token = None
            subaccount.token_info.expiry = 0
            raise ConnectionError(f"HTTP Error {err.response.status_code} fetching token for '{subaccount_name}'") from err
            
        except requests.exceptions.RequestException as err:
            logging.error(f"Network/Request error fetching token for '{subaccount_name}': {err}")
            subaccount.token_info.token = None
            subaccount.token_info.expiry = 0
            raise ConnectionError(f"Network error fetching token for '{subaccount_name}': {err}") from err
            
        except Exception as err:
            logging.error(f"Unexpected token fetch error for '{subaccount_name}': {err}", exc_info=True)
            subaccount.token_info.token = None
            subaccount.token_info.expiry = 0
            raise RuntimeError(f"Unexpected error processing token response for '{subaccount_name}': {err}") from err

def verify_request_token(request):
    """Verifies the Authorization or x-api-key header from the incoming client request."""
    token = request.headers.get("Authorization") or request.headers.get("x-api-key")
    logging.info(f"verify_request_token, Token received in request: {token[:15]}..." if token and len(token) > 15 else token)

    if not proxy_config.secret_authentication_tokens:
        logging.warning("Client authentication disabled - no tokens configured.")
        return True

    if not token:
        logging.error("Missing token in request. Checked Authorization and x-api-key headers.")
        return False

    # The check `secret_key in token` handles both "Bearer <token>" and just "<token>"
    if not any(secret_key in token for secret_key in proxy_config.secret_authentication_tokens):
        logging.error("Invalid token - no matching token found.")
        return False

    logging.debug("Client token verified successfully.")
    return True

def convert_openai_to_claude(payload):
    # Extract system message if present
    system_message = ""
    messages = payload["messages"]
    if messages and messages[0]["role"] == "system":
        system_message = messages.pop(0)["content"]
    # Conversion logic from OpenAI to Claude API format
    claude_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": payload.get("max_tokens", 4096000),
        "temperature": payload.get("temperature", 1.0),
        "system": system_message,
        "messages": messages,
        # "thinking": {
        #     "type": "enabled",
        #     "budget_tokens": 16000
        # },
    }
    return claude_payload

def convert_openai_to_claude37(payload):
    """
    Converts an OpenAI API request payload to the format expected by the
    Claude 3.7 /converse endpoint.
    """
    logging.debug("Original OpenAI payload for Claude 3.7 conversion: %s", _LazyJSON(payload, indent=2))

    # Extract system message if present
    system_message = ""
    messages = payload.get("messages", [])
    if messages and messages[0].get("role") == "system":
        system_message = messages.pop(0).get("content", "")

    # Extract inference configuration parameters
    inference_config = {}
    if "max_tokens" in payload:
        # Ensure max_tokens is an integer
        try:
            inference_config["maxTokens"] = int(payload["max_tokens"])
        except (ValueError, TypeError):
             logging.warning(f"Invalid value for max_tokens: {payload['max_tokens']}. Using default or omitting.")
    if "temperature" in payload:
         # Ensure temperature is a float
        try:
            inference_config["temperature"] = float(payload["temperature"])
        except (ValueError, TypeError):
            logging.warning(f"Invalid value for temperature: {payload['temperature']}. Using default or omitting.")
    if "stop" in payload:
        stop_sequences = payload["stop"]
        if isinstance(stop_sequences, str):
            inference_config["stopSequences"] = [stop_sequences]
        elif isinstance(stop_sequences, list) and all(isinstance(s, str) for s in stop_sequences):
            inference_config["stopSequences"] = stop_sequences
        else:
            logging.warning(f"Unsupported type or content for 'stop' parameter: {stop_sequences}. Ignoring.")

    # Convert messages format
    converted_messages = []
    # The loop now iterates through the original messages list,
    # potentially including the system message if it wasn't removed earlier.
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # Handle system, user, or assistant roles for inclusion in the messages list
        # Note: While the top-level 'system' parameter is standard for Claude /converse,
        # this modification includes the system message in the 'messages' array as requested.
        # This might deviate from the expected API usage.
        if role in ["user", "assistant"]:
            if content:
                if isinstance(content, str):
                    # Convert string content to the required list of blocks format
                    converted_messages.append({
                        "role": role,
                        "content": [{"text": content}]
                    })
                elif isinstance(content, list):
                    # Validate that each item in the list is a correctly structured block
                    validated_content = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            validated_content.append(item)
                        elif isinstance(item, str):
                             # Convert string item to block format
                            validated_content.append({"text": item})
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            # Convert OpenAI image_url format to AWS Bedrock Converse image format
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                # Parse data URL format: data:image/png;base64,<data>
                                try:
                                    header, data = image_url.split(",", 1)
                                    media_type = header.split(";")[0].split(":")[1]  # e.g., "image/png"
                                    # Extract format (png, jpeg, gif, webp) from media type
                                    image_format = media_type.split("/")[1] if "/" in media_type else "png"

                                    # AWS Bedrock Converse API format
                                    converse_image = {
                                        "image": {
                                            "format": image_format,
                                            "source": {
                                                "bytes": data
                                            }
                                        }
                                    }
                                    validated_content.append(converse_image)
                                    logging.info(f"Converted image_url to AWS Bedrock Converse format with format: {image_format}")
                                except Exception as e:
                                    logging.error(f"Failed to parse image data URL: {e}")
                            else:
                                logging.warning(f"Unsupported image_url format (only data URLs supported): {image_url[:100]}")
                        else:
                            logging.warning(f"Skipping invalid content block for role {role}: {item}")
                    
                    if validated_content:
                        converted_messages.append({
                            "role": role,
                            "content": validated_content
                        })
                    else:
                        logging.warning(f"Skipping message for role {role} due to all content blocks being invalid: {content}")
                else:
                    logging.warning(f"Skipping message for role {role} due to unsupported content type: {type(content)}")
            else:
                logging.warning(f"Skipping message for role {role} due to missing content: {msg}")
        else:
             # Skip any other unsupported roles
             logging.warning(f"Skipping message with unsupported role for Claude /converse: {role}")
             continue
    
    # add the system_message to the converted_messages as the first element
    if system_message:
        converted_messages.insert(0, {
            "role": "user",
            "content": [{"text": system_message}]
        })

    # Construct the final Claude 3.7 payload
    claude_payload = {
        "messages": converted_messages
    }

    # Add inferenceConfig only if it's not empty
    if inference_config:
        claude_payload["inferenceConfig"] = inference_config

    # Add system message if it exists
    # Claude 3.7 doesn't support the system_message as a top-level parameter
    # if system_message:
        # Claude /converse API supports a top-level system prompt as a list of blocks
        # claude_payload["system"] = [{"text": system_message}]

    logging.debug("Converted Claude 3.7 payload: %s", _LazyJSON(claude_payload, indent=2))
    return claude_payload


def convert_claude_request_to_openai(payload):
    """Converts a Claude Messages API request to an OpenAI Chat Completion request."""
    logging.debug("Original Claude payload for OpenAI conversion: %s", _LazyJSON(payload, indent=2))

    openai_messages = []
    if "system" in payload and payload["system"]:
        openai_messages.append({"role": "system", "content": payload["system"]})

    openai_messages.extend(payload.get("messages", []))

    openai_payload = {
        "model": payload.get("model"),
        "messages": openai_messages,
    }

    if "max_tokens" in payload:
        openai_payload["max_completion_tokens"] = payload["max_tokens"]
    if "temperature" in payload:
        openai_payload["temperature"] = payload["temperature"]
    if "stream" in payload:
        openai_payload["stream"] = payload["stream"]
    if "tools" in payload and payload["tools"]:
        # Convert Claude tools format to OpenAI tools format
        openai_tools = []
        for tool in payload["tools"]:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
        openai_payload["tools"] = openai_tools
        logging.debug(f"Converted {len(openai_tools)} tools for OpenAI format")

    logging.debug("Converted OpenAI payload: %s", _LazyJSON(openai_payload, indent=2))
    return openai_payload


def convert_claude_request_to_gemini(payload):
    """Converts a Claude Messages API request to a Google Gemini request."""
    logging.debug("Original Claude payload for Gemini conversion: %s", _LazyJSON(payload, indent=2))

    gemini_contents = []
    system_prompt = payload.get("system", "")

    claude_messages = payload.get("messages", [])

    if system_prompt and claude_messages and claude_messages[0]["role"] == "user":
        first_user_content = claude_messages[0]["content"]
        if isinstance(first_user_content, list):
            first_user_content_text = " ".join(c.get("text", "") for c in first_user_content if c.get("type") == "text")
        else:
            first_user_content_text = first_user_content

        claude_messages[0]["content"] = f"{system_prompt}\\n\\n{first_user_content_text}"

    for message in claude_messages:
        role = "user" if message["role"] == "user" else "model"

        if isinstance(message["content"], list):
            content_text = " ".join(c.get("text", "") for c in message["content"] if c.get("type") == "text")
        else:
            content_text = message["content"]

        if gemini_contents and gemini_contents[-1]["role"] == role:
            gemini_contents[-1]["parts"]["text"] += f"\\n\\n{content_text}"
        else:
            gemini_contents.append({
                "role": role,
                "parts": {"text": content_text}
            })

    gemini_payload = {
        "contents": gemini_contents,
        "generation_config": {},
        "safety_settings": {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        }
    }

    if "max_tokens" in payload:
        gemini_payload["generation_config"]["maxOutputTokens"] = payload["max_tokens"]
    if "temperature" in payload:
        gemini_payload["generation_config"]["temperature"] = payload["temperature"]
    if "tools" in payload and payload["tools"]:
        # Convert Claude tools format to Gemini tools format
        gemini_tools = []
        for tool in payload["tools"]:
            gemini_tool = {
                "function_declarations": [{
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }]
            }
            gemini_tools.append(gemini_tool)
        gemini_payload["tools"] = gemini_tools
        logging.debug(f"Converted {len(gemini_tools)} tools for Gemini format")

    logging.debug("Converted Gemini payload: %s", _LazyJSON(gemini_payload, indent=2))
    return gemini_payload


def convert_claude_request_for_bedrock(payload):
    """
    Converts a Claude Messages API request to Bedrock Claude format.
    Handles tools conversion for Bedrock compatibility.
    """
    logging.debug("Original Claude payload for Bedrock conversion: %s", _LazyJSON(payload, indent=2))
    
    bedrock_payload = {}
    
    # Copy basic fields
    for field in ["model", "max_tokens", "temperature", "top_p", "top_k", "stop_sequences"]:
        if field in payload:
            bedrock_payload[field] = payload[field]
    
    # Handle system field - keep as array format
    if "system" in payload:
        bedrock_payload["system"] = payload["system"]
    
    # Handle messages - remove cache_control fields
    if "messages" in payload:
        cleaned_messages = []
        for message in payload["messages"]:
            cleaned_message = {"role": message["role"]}
            
            # Handle content
            if isinstance(message["content"], list):
                cleaned_content = []
                for content_item in message["content"]:
                    if isinstance(content_item, dict):
                        # Remove cache_control field
                        cleaned_item = {k: v for k, v in content_item.items() if k != "cache_control"}
                        cleaned_content.append(cleaned_item)
                    else:
                        cleaned_content.append(content_item)
                cleaned_message["content"] = cleaned_content
            else:
                cleaned_message["content"] = [{"type": "text", "text": message["content"]}]
            
            cleaned_messages.append(cleaned_message)
        bedrock_payload["messages"] = cleaned_messages

    sanitize_bedrock_tool_result_content(bedrock_payload, request_id="convert_claude_request_for_bedrock")
    
    # Handle tools conversion if present
    if "tools" in payload and payload["tools"]:
        bedrock_payload["tools"] = payload["tools"]
        sanitize_bedrock_tools(bedrock_payload, request_id="convert_claude_request_for_bedrock")
        logging.debug(f"Tools present in request: {len(bedrock_payload.get('tools', []))} tools")
    
    # Handle anthropic_beta if present (but not in payload, should be in headers)
    # Remove it from payload as it should be in headers only
    
    # Set anthropic_version if not present
    if "anthropic_version" not in bedrock_payload:
        bedrock_payload["anthropic_version"] = "bedrock-2023-05-31"
    
    logging.debug("Converted Bedrock Claude payload: %s", _LazyJSON(bedrock_payload, indent=2))
    return bedrock_payload


def sanitize_bedrock_tools(body: Dict[str, Any], request_id: str = "unknown") -> Dict[str, int]:
    tools_list = body.get("tools")
    removed_fields = 0
    stripped_tools = 0

    if not isinstance(tools_list, list):
        return {"removed_fields": 0, "stripped_tools": 0}

    BEDROCK_UNSUPPORTED_TOOL_TYPES = {
        "web_search_20250305",
        "computer_20241022",
        "text_editor_20241022",
        "bash_20241022",
    }

    original_tool_count = len(tools_list)
    sanitized_tools: List[Dict[str, Any]] = []

    for tool in tools_list:
        if not isinstance(tool, dict):
            stripped_tools += 1
            continue

        if tool.get("type") in BEDROCK_UNSUPPORTED_TOOL_TYPES:
            stripped_tools += 1
            continue

        name = tool.get("name")
        input_schema = tool.get("input_schema")
        description = tool.get("description")

        if not isinstance(name, str) or not name:
            stripped_tools += 1
            continue
        if not isinstance(input_schema, dict):
            stripped_tools += 1
            continue
        if not isinstance(description, str):
            description = ""

        sanitized_tool = {"name": name, "description": description, "input_schema": input_schema}
        removed_fields += max(0, len(tool) - len(sanitized_tool))
        sanitized_tools.append(sanitized_tool)

    if stripped_tools:
        logging.info(
            "[%s] Stripped %d tool(s) from request for Bedrock compatibility",
            request_id,
            stripped_tools,
        )

    if sanitized_tools:
        body["tools"] = sanitized_tools
    else:
        body.pop("tools", None)
        if "tool_choice" in body:
            body.pop("tool_choice", None)
            logging.info("[%s] Removed 'tool_choice' because all tools were stripped", request_id)
        return {"removed_fields": removed_fields, "stripped_tools": original_tool_count}

    if removed_fields:
        logging.info("[%s] Removed %d unsupported tool field(s) for Bedrock", request_id, removed_fields)

    return {"removed_fields": removed_fields, "stripped_tools": stripped_tools}


def sanitize_bedrock_tool_result_content(body: Dict[str, Any], request_id: str = "unknown") -> Dict[str, int]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return {"converted_blocks": 0, "normalized_tool_results": 0}

    allowed_content_types = {"text", "image", "document", "search_result"}
    converted_blocks = 0
    normalized_tool_results = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        content_list = message.get("content")
        if not isinstance(content_list, list):
            continue

        for content_item in content_list:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") != "tool_result":
                continue

            tool_result_content = content_item.get("content")

            if isinstance(tool_result_content, str):
                content_item["content"] = [{"type": "text", "text": tool_result_content}]
                normalized_tool_results += 1
                continue

            if isinstance(tool_result_content, list):
                new_blocks: List[Any] = []
                for block in tool_result_content:
                    if isinstance(block, dict) and block.get("type") in allowed_content_types:
                        new_blocks.append(block)
                        continue

                    if isinstance(block, dict) and block.get("type") == "tool_reference":
                        tool_name = block.get("tool_name")
                        suffix = f" tool_name={tool_name}" if isinstance(tool_name, str) else ""
                        new_blocks.append(
                            {
                                "type": "text",
                                "text": f"[tool_ref]{suffix}",
                            }
                        )
                        converted_blocks += 1
                        continue

                    try:
                        dumped = json.dumps(block, ensure_ascii=False)
                    except Exception:
                        dumped = str(block)
                    new_blocks.append(
                        {
                            "type": "text",
                            "text": f"[unsupported_tool_result_content] {dumped}",
                        }
                    )
                    converted_blocks += 1

                content_item["content"] = new_blocks
                continue

            try:
                dumped = json.dumps(tool_result_content, ensure_ascii=False)
            except Exception:
                dumped = str(tool_result_content)
            content_item["content"] = [{"type": "text", "text": dumped}]
            normalized_tool_results += 1

    if converted_blocks or normalized_tool_results:
        logging.info(
            "[%s] Sanitized tool_result content for Bedrock: converted_blocks=%d normalized_tool_results=%d",
            request_id,
            converted_blocks,
            normalized_tool_results,
        )

    return {"converted_blocks": converted_blocks, "normalized_tool_results": normalized_tool_results}


def convert_claude_to_openai(response, model):
    # Check if the model is Claude 3.7, 4, 4.5, or 4.6
    if is_claude_37_or_4(model):
        logging.info(f"Detected Claude model ('{model}'), using convert_claude37_to_openai.")
        return convert_claude37_to_openai(response, model)

    # Proceed with the original Claude conversion logic for other models
    logging.info(f"Using standard Claude conversion for model '{model}'.")

    try:
        logging.debug("Raw response from Claude API: %s", _LazyJSON(response, indent=4))

        # Ensure the response contains the expected structure
        if "content" not in response or not isinstance(response["content"], list):
            raise ValueError("Invalid response structure: 'content' is missing or not a list")

        first_content = response["content"][0]
        if not isinstance(first_content, dict) or "text" not in first_content:
            raise ValueError("Invalid response structure: 'content[0].text' is missing")

        # Conversion logic from Claude API to OpenAI format
        openai_response = {
            "choices": [
                {
                    "finish_reason": response.get("stop_reason", "stop"),
                    "index": 0,
                    "message": {
                        "content": first_content["text"],
                        "role": response.get("role", "assistant")
                    }
                }
            ],
            "created": int(time.time()),
            "id": response.get("id", "chatcmpl-unknown"),
            "model": response.get("model", "claude-v1"),
            "object": "chat.completion",
            "usage": {
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
                "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0)
            }
        }
        logging.debug("Converted response to OpenAI format: %s", _LazyJSON(openai_response, indent=4))
        return openai_response
    except Exception as e:
        logging.error(f"Error converting Claude response to OpenAI format: {e}")
        return {
            "error": "Invalid response from Claude API",
            "details": str(e)
        }

def convert_claude37_to_openai(response, model_name="claude-3.7"):
    """
    Converts a Claude 3.7/4/4.5/4.6 /converse API response payload (non-streaming)
    to the format expected by the OpenAI Chat Completion API.
    """
    try:
        logging.debug("Raw response from Claude API: %s", _LazyJSON(response, indent=2))

        # Validate the overall response structure
        if not isinstance(response, dict):
            raise ValueError("Invalid response format: response is not a dictionary")

        # --- Extract 'output' ---
        output = response.get("output")
        if not isinstance(output, dict):
            # Handle cases where the structure might differ unexpectedly
            # For now, strictly expect the documented /converse structure
            raise ValueError("Invalid response structure: 'output' field is missing or not a dictionary")

        # --- Extract 'message' from 'output' ---
        message = output.get("message")
        if not isinstance(message, dict):
            raise ValueError("Invalid response structure: 'output.message' field is missing or not a dictionary")

        # --- Extract 'content' list from 'message' ---
        content_list = message.get("content")
        if not isinstance(content_list, list) or not content_list:
            # Check if content is empty but maybe role/stopReason are still valid?
            # For now, require non-empty content for a standard completion response.
            raise ValueError("Invalid response structure: 'output.message.content' is missing, not a list, or empty")

        # --- Extract text from the first content block ---
        # Assuming the primary response content is in the first block and is text.
        # More complex handling might be needed for multi-modal or tool use responses.
        first_content_block = content_list[0]
        if not isinstance(first_content_block, dict) or "text" not in first_content_block:
            # Log the type if it's not text, for debugging.
            block_type = first_content_block.get("type", "unknown") if isinstance(first_content_block, dict) else "not a dict"
            logging.warning(f"First content block is not of type 'text' or missing 'text' key. Type: {block_type}. Content: {first_content_block}")
            # Decide how to handle non-text blocks. For now, raise error if no text found.
            # Find the first text block if available?
            content_text = None
            for block in content_list:
                if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                    content_text = block["text"]
                    logging.info(f"Found text content in block at index {content_list.index(block)}")
                    break
            if content_text is None:
                 raise ValueError("No text content block found in the response message content")
        else:
            content_text = first_content_block["text"]


        # --- Extract 'role' from 'message' ---
        message_role = message.get("role", "assistant") # Default to assistant if missing

        # --- Extract 'usage' information ---
        usage = response.get("usage")
        if not isinstance(usage, dict):
            logging.warning("Usage information missing or invalid in Claude response. Setting tokens to 0.")
            usage = {} # Use empty dict to avoid errors in .get() calls below

        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        # Claude /converse should provide totalTokens, but calculate as fallback
        total_tokens = usage.get("totalTokens", input_tokens + output_tokens)


        # --- Map Claude stopReason to OpenAI finish_reason ---
        stop_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls", # Map tool use if needed
            # Add other potential Claude stop reasons if they arise
        }
        claude_stop_reason = response.get("stopReason")
        finish_reason = stop_reason_map.get(claude_stop_reason, "stop") # Default to 'stop' if unknown or missing

        # --- Construct the OpenAI response ---
        openai_response = {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "content": content_text,
                        "role": message_role
                    },
                    # "logprobs": None, # Not available from Claude
                }
            ],
            "created": int(time.time()),
            "id": f"chatcmpl-claude37-{random.randint(10000, 99999)}", # More specific ID prefix
            "model": model_name, # Use the provided model name
            "object": "chat.completion",
            "usage": {
                "completion_tokens": output_tokens,
                "prompt_tokens": input_tokens,
                "total_tokens": total_tokens
            }
            # "system_fingerprint": None # Not available from Claude /converse
        }
        logging.debug("Converted response to OpenAI format: %s", _LazyJSON(openai_response, indent=2))
        return openai_response

    except Exception as e:
        # Log the error with traceback for better debugging
        logging.error(f"Error converting Claude 3.7/4 response to OpenAI format: {e}", exc_info=True)
        # Log the problematic response structure that caused the error
        logging.error(f"Problematic Claude response structure: {json.dumps(response, indent=2)}")
        # Return an error structure compliant with OpenAI format
        return {
            "object": "error",
            "message": f"Failed to convert Claude 3.7/4 response to OpenAI format. Error: {str(e)}. Check proxy logs for details.",
            "type": "proxy_conversion_error",
            "param": None,
            "code": None
            # Optionally include parts of the OpenAI structure if needed by the client
            # "choices": [],
            # "created": int(time.time()),
            # "id": f"chatcmpl-error-{random.randint(10000, 99999)}",
            # "model": model_name,
            # "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }

def convert_claude_chunk_to_openai(chunk, model):
    try:
        # Log the raw chunk received
        # Log the raw chunk received only if it's a 3.7 model
        logging.debug(f"{model} Raw Claude chunk received: {chunk}")
        # Parse the Claude chunk
        data = json.loads(chunk.replace("data: ", "").strip())
        
        # Initialize the OpenAI chunk structure
        openai_chunk = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": None,
                    "index": 0
                }
            ],
            "created": int(time.time()),
            "id": data.get("message", {}).get("id", "chatcmpl-unknown"),
            "model": "claude-v1",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_36b0c83da2"
        }

        # Map Claude's content to OpenAI's delta
        if data.get("type") == "content_block_delta":
            openai_chunk["choices"][0]["delta"]["content"] = data["delta"]["text"]
        elif data.get("type") == "message_delta" and data["delta"]["stop_reason"] == "end_turn":
            openai_chunk["choices"][0]["finish_reason"] = "stop"

        return f"data: {json.dumps(openai_chunk)}\n\n"
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return f"data: {{\"error\": \"Invalid JSON format\"}}\n\n"
    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        return f"data: {{\"error\": \"Error processing chunk\"}}\n\n"

# Configure logging if not already configured elsewhere
# logging.basicConfig(level=logging.DEBUG)

def convert_claude37_chunk_to_openai(claude_chunk, model_name):
    """
    Converts a single parsed Claude 3.7/4/4.5/4.6 /converse-stream chunk (dictionary)
    into an OpenAI-compatible Server-Sent Event (SSE) string.
    Returns None if the chunk doesn't map to an OpenAI event (e.g., metadata).
    """
    try:
        # Generate a consistent-ish ID for the stream parts
        # In a real scenario, this ID should be generated once per request stream
        # and potentially passed down or managed in the calling context.
        stream_id = f"chatcmpl-claude37-{random.randint(10000, 99999)}"
        created_time = int(time.time())

        openai_chunk_payload = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                    # "logprobs": None # Not available from Claude
                }
            ]
            # "system_fingerprint": None # Not typically sent in chunks
        }

        # Determine chunk type based on the first key in the dictionary
        # claude_chunk is string, so need to parse it
        if isinstance(claude_chunk, str):
            try:
                # claude_chunk = json.dumps(claude_chunk.replace("data: ", "").strip())
                logging.debug(f"Parsed Claude chunk: {claude_chunk}")
                claude_chunk = json.loads(claude_chunk)
                logging.debug("Decoded Claude chunk: %s", _LazyJSON(claude_chunk, indent=2))
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                return None

        if not isinstance(claude_chunk, dict) or not claude_chunk:
                logging.warning(f"Invalid or empty Claude chunk received: {claude_chunk}")
                return None

        chunk_type = next(iter(claude_chunk)) # Get the first key

        if chunk_type == "messageStart":
            # Extract role, default to assistant if not present
            role = claude_chunk.get("messageStart", {}).get("role", "assistant")
            openai_chunk_payload["choices"][0]["delta"]["role"] = role
            logging.debug(f"Converted messageStart chunk: {openai_chunk_payload}")

        elif chunk_type == "contentBlockDelta":
            # Extract text delta
            text_delta = claude_chunk.get("contentBlockDelta", {}).get("delta", {}).get("text")
            if text_delta is not None: # Send even if empty string delta? OpenAI usually does.
                openai_chunk_payload["choices"][0]["delta"]["content"] = text_delta
                logging.debug(f"Converted contentBlockDelta chunk: {openai_chunk_payload}")
            else:
                # If delta or text is missing, maybe log but don't send?
                logging.debug(f"Ignoring contentBlockDelta without text: {claude_chunk}")
                return None # Don't send chunk if no actual text delta

        elif chunk_type == "messageStop":
            # Extract stop reason
            stop_reason = claude_chunk.get("messageStop", {}).get("stopReason")
            # Map Claude stopReason to OpenAI finish_reason
            stop_reason_map = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "tool_use": "tool_calls", # Map tool use if needed
                # Add other potential Claude stop reasons if they arise
            }
            finish_reason = stop_reason_map.get(stop_reason)
            if finish_reason:
                    openai_chunk_payload["choices"][0]["finish_reason"] = finish_reason
                    # Delta should be empty or null for the final chunk with finish_reason
                    openai_chunk_payload["choices"][0]["delta"] = {} # Ensure delta is empty
                    logging.debug(f"Converted messageStop chunk: {openai_chunk_payload}")
            else:
                    logging.warning(f"Unmapped or missing stopReason in messageStop: {stop_reason}. Chunk: {claude_chunk}")
                    # Decide if to send a default stop or ignore
                    # Sending with finish_reason=null might be confusing. Let's ignore.
                    return None

        elif chunk_type in ["contentBlockStart", "contentBlockStop", "metadata"]:
            # These Claude events don't have a direct OpenAI chunk equivalent
            # containing message delta or finish reason. Ignore them for streaming output.
            # Metadata chunk should be handled separately in the calling function (`generate`)
            # to extract usage information.
            logging.debug(f"Ignoring Claude chunk type for OpenAI stream: {chunk_type}")
            return None
        else:
            logging.warning(f"Unknown Claude 3.7/4 chunk type encountered: {chunk_type}. Chunk: {claude_chunk}")
            return None

        # Format as SSE string if a valid payload was constructed
        sse_string = f"data: {json.dumps(openai_chunk_payload)}\n\n"
        return sse_string

    except Exception as e:
        logging.error(f"Error converting Claude 3.7/4 chunk to OpenAI format: {e}", exc_info=True)
        logging.error(f"Problematic Claude chunk: {json.dumps(claude_chunk, indent=2)}")
        # Optionally return an error chunk in SSE format to the client
        error_payload = {
                "id": f"chatcmpl-error-{random.randint(10000, 99999)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": f"[PROXY ERROR: Failed to convert upstream chunk - {str(e)}]"}, "finish_reason": "stop"}]
        }
        return f"data: {json.dumps(error_payload)}\n\n"


def is_claude_model(model):
    return any(keyword in model for keyword in ["claude", "clau", "claud", "sonnet", "sonne", "sonn", "CLAUDE", "SONNET"])

def is_claude_37_or_4(model):
    """
    Check if the model is Claude 3.7 or Claude 4.
    
    Args:
        model: The model name to check
        
    Returns:
        bool: True if the model is Claude 3.7 or Claude 4, False otherwise
    """
    return any(version in model for version in ["3.7", "4"]) or "3.5" not in model

def is_gemini_model(model):
    """
    Check if the model is a Gemini model.
    
    Args:
        model: The model name to check
        
    Returns:
        bool: True if the model is a Gemini model, False otherwise
    """
    return any(keyword in model.lower() for keyword in ["gemini", "gemini-1.5", "gemini-2.5", "gemini-pro", "gemini-flash"])

def convert_openai_to_gemini(payload):
    """
    Converts an OpenAI API request payload to the format expected by the
    Google Vertex AI Gemini generateContent endpoint.
    """
    logging.info(f"Original OpenAI payload for Gemini conversion: {json.dumps(payload, indent=2)}")

    # Extract system message if present
    system_message = ""
    messages = payload.get("messages", [])
    if messages and messages[0].get("role") == "system":
        system_message = messages.pop(0).get("content", "")

    # Build generation config
    generation_config = {}
    if "max_tokens" in payload:
        try:
            generation_config["maxOutputTokens"] = int(payload["max_tokens"])
        except (ValueError, TypeError):
            logging.warning(f"Invalid value for max_tokens: {payload['max_tokens']}. Using default or omitting.")
    
    if "temperature" in payload:
        try:
            generation_config["temperature"] = float(payload["temperature"])
        except (ValueError, TypeError):
            logging.warning(f"Invalid value for temperature: {payload['temperature']}. Using default or omitting.")
    
    if "top_p" in payload:
        try:
            generation_config["topP"] = float(payload["top_p"])
        except (ValueError, TypeError):
            logging.warning(f"Invalid value for top_p: {payload['top_p']}. Using default or omitting.")

    # Convert messages to Gemini format
    # For single message case (most common), create a simple structure
    if len(messages) == 1 and messages[0].get("role") == "user":
        # Single user message case - match the curl example structure
        user_content = messages[0].get("content", "")
        
        # Handle different content types (string or list)
        if isinstance(user_content, list):
            # Extract text from content blocks
            text_content = ""
            for block in user_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content += block.get("text", "")
                elif isinstance(block, str):
                    text_content += block
            user_content = text_content
        elif not isinstance(user_content, str):
            # Convert to string if it's neither list nor string
            user_content = str(user_content)
        
        # If there's a system message, prepend it to the user content
        if system_message:
            user_content = system_message + "\n\n" + user_content
        
        gemini_contents = {
            "role": "user",
            "parts": {
                "text": user_content
            }
        }
    else:
        # Multiple messages case - use array format
        gemini_contents = []
        
        # Add system message as the first user message if it exists
        if system_message:
            gemini_contents.append({
                "role": "user",
                "parts": {"text": system_message}
            })
        
        # Process remaining messages
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            # Map OpenAI roles to Gemini roles
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                logging.warning(f"Skipping message with unsupported role for Gemini: {role}")
                continue
            
            if content:
                # Handle different content types (string or list)
                if isinstance(content, list):
                    # Extract text from content blocks
                    text_content = ""
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content += block.get("text", "")
                        elif isinstance(block, str):
                            text_content += block
                    content = text_content
                elif not isinstance(content, str):
                    # Convert to string if it's neither list nor string
                    content = str(content)
                
                # Check if we can merge with the previous message (same role)
                if gemini_contents and gemini_contents[-1]["role"] == gemini_role:
                    # Merge with previous message
                    if isinstance(gemini_contents[-1]["parts"], dict):
                        gemini_contents[-1]["parts"]["text"] += "\n\n" + content
                    else:
                        # Handle case where parts might be a list (future extension)
                        gemini_contents[-1]["parts"] = {"text": gemini_contents[-1]["parts"]["text"] + "\n\n" + content}
                else:
                    # Add new message
                    gemini_contents.append({
                        "role": gemini_role,
                        "parts": {"text": content}
                    })

    # Build safety settings (as a single object to match the curl example)
    safety_settings = {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_LOW_AND_ABOVE"
    }

    # Construct the final Gemini payload
    gemini_payload = {
        "contents": gemini_contents
    }

    # Add generation config if not empty
    if generation_config:
        gemini_payload["generation_config"] = generation_config

    # Add safety settings
    gemini_payload["safety_settings"] = safety_settings

    logging.debug("Converted Gemini payload: %s", _LazyJSON(gemini_payload, indent=2))
    return gemini_payload

def convert_gemini_to_openai(response, model_name="gemini-pro"):
    """
    Converts a Gemini generateContent API response payload (non-streaming)
    to the format expected by the OpenAI Chat Completion API.
    """
    try:
        logging.debug("Raw response from Gemini API: %s", _LazyJSON(response, indent=2))

        # Validate the overall response structure
        if not isinstance(response, dict):
            raise ValueError("Invalid response format: response is not a dictionary")

        # Extract candidates
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("Invalid response structure: no candidates found")

        # Get the first candidate
        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            raise ValueError("Invalid response structure: candidate is not a dictionary")

        # Extract content from the candidate
        content = first_candidate.get("content", {})
        if not isinstance(content, dict):
            raise ValueError("Invalid response structure: content is not a dictionary")

        # Extract parts from content
        parts = content.get("parts", [])
        if not parts:
            raise ValueError("Invalid response structure: no parts found in content")

        # Extract text from the first part
        first_part = parts[0]
        if not isinstance(first_part, dict) or "text" not in first_part:
            raise ValueError("Invalid response structure: no text found in first part")

        content_text = first_part["text"]

        # Extract finish reason
        finish_reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "stop"
        }
        gemini_finish_reason = first_candidate.get("finishReason", "STOP")
        finish_reason = finish_reason_map.get(gemini_finish_reason, "stop")

        # Extract usage information
        usage_metadata = response.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", prompt_tokens + completion_tokens)

        # Construct the OpenAI response
        openai_response = {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "content": content_text,
                        "role": "assistant"
                    }
                }
            ],
            "created": int(time.time()),
            "id": f"chatcmpl-gemini-{random.randint(10000, 99999)}",
            "model": model_name,
            "object": "chat.completion",
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }
        
        logging.debug("Converted response to OpenAI format: %s", _LazyJSON(openai_response, indent=2))
        return openai_response

    except Exception as e:
        logging.error(f"Error converting Gemini response to OpenAI format: {e}", exc_info=True)
        logging.error(f"Problematic Gemini response structure: {json.dumps(response, indent=2)}")
        return {
            "object": "error",
            "message": f"Failed to convert Gemini response to OpenAI format. Error: {str(e)}. Check proxy logs for details.",
            "type": "proxy_conversion_error",
            "param": None,
            "code": None
        }


def convert_gemini_response_to_claude(response, model_name="gemini-pro"):
    """
    Converts a Gemini generateContent API response payload (non-streaming)
    to the format expected by the Anthropic Claude Messages API.
    """
    try:
        logging.debug("Raw response from Gemini API for Claude conversion: %s", _LazyJSON(response, indent=2))

        if not isinstance(response, dict) or "candidates" not in response or not response["candidates"]:
            raise ValueError("Invalid Gemini response: 'candidates' field is missing or empty")

        first_candidate = response["candidates"][0]
        content_parts = first_candidate.get("content", {}).get("parts", [])
        if not content_parts or "text" not in content_parts[0]:
            raise ValueError("Invalid Gemini response: text content not found in 'parts'")

        content_text = content_parts[0]["text"]

        # Map Gemini finishReason to Claude stop_reason
        gemini_finish_reason = first_candidate.get("finishReason", "STOP")
        stop_reason_map = {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
            "SAFETY": "stop_sequence",
            "RECITATION": "stop_sequence",
            "OTHER": "stop_sequence"
        }
        claude_stop_reason = stop_reason_map.get(gemini_finish_reason, "stop_sequence")

        # Extract usage
        usage_metadata = response.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)

        claude_response = {
            "id": f"msg_gemini_{random.randint(10000, 99999)}",
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [{"type": "text", "text": content_text}],
            "stop_reason": claude_stop_reason,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens
            }
        }
        logging.debug("Converted Gemini response to Claude format: %s", _LazyJSON(claude_response, indent=2))
        return claude_response

    except Exception as e:
        logging.error(f"Error converting Gemini response to Claude format: {e}", exc_info=True)
        return {
            "type": "error",
            "error": {
                "type": "proxy_conversion_error",
                "message": f"Failed to convert Gemini response to Claude format: {str(e)}"
            }
        }


def convert_openai_response_to_claude(response):
    """
    Converts an OpenAI Chat Completion API response payload (non-streaming)
    to the format expected by the Anthropic Claude Messages API.
    """
    try:
        logging.debug("Raw response from OpenAI API for Claude conversion: %s", _LazyJSON(response, indent=2))

        if not isinstance(response, dict) or "choices" not in response or not response["choices"]:
            raise ValueError("Invalid OpenAI response: 'choices' field is missing or empty")

        first_choice = response["choices"][0]
        message = first_choice.get("message", {})
        content_text = message.get("content")
        tool_calls = message.get("tool_calls", [])
        
        # Handle content based on whether there are tool calls
        claude_content = []
        if content_text:
            claude_content.append({"type": "text", "text": content_text})
        
        # Convert OpenAI tool calls to Claude format
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.get("type") == "function":
                    function = tool_call.get("function", {})
                    claude_tool_use = {
                        "type": "tool_use",
                        "id": tool_call.get("id", f"toolu_openai_{random.randint(10000, 99999)}"),
                        "name": function.get("name"),
                        "input": json.loads(function.get("arguments", "{}"))
                    }
                    claude_content.append(claude_tool_use)
        
        if not claude_content:
            raise ValueError("Invalid OpenAI response: no content or tool calls found")

        # Map OpenAI finish_reason to Claude stop_reason
        openai_finish_reason = first_choice.get("finish_reason")
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "stop_sequence",
            "tool_calls": "tool_use",
        }
        claude_stop_reason = stop_reason_map.get(openai_finish_reason, "stop_sequence")

        # Extract usage
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        claude_response = {
            "id": response.get("id", f"msg_openai_{random.randint(10000, 99999)}"),
            "type": "message",
            "role": "assistant",
            "model": response.get("model", "unknown_openai_model"),
            "content": claude_content,
            "stop_reason": claude_stop_reason,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens
            }
        }
        logging.debug("Converted OpenAI response to Claude format: %s", _LazyJSON(claude_response, indent=2))
        return claude_response

    except Exception as e:
        logging.error(f"Error converting OpenAI response to Claude format: {e}", exc_info=True)
        return {
            "type": "error",
            "error": {
                "type": "proxy_conversion_error",
                "message": f"Failed to convert OpenAI response to Claude format: {str(e)}"
            }
        }


def convert_gemini_chunk_to_claude_delta(gemini_chunk):
    """Extracts a Claude-formatted content delta from a Gemini streaming chunk."""
    text_delta = gemini_chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
    if text_delta:
        return {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text_delta}
        }
    return None

def get_claude_stop_reason_from_gemini_chunk(gemini_chunk):
    """Extracts and maps the stop reason from a final Gemini chunk."""
    finish_reason = gemini_chunk.get("candidates", [{}])[0].get("finishReason")
    if finish_reason:
        stop_reason_map = {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
            "SAFETY": "stop_sequence",
            "RECITATION": "stop_sequence",
            "OTHER": "stop_sequence"
        }
        return stop_reason_map.get(finish_reason, "stop_sequence")
    return None

def convert_openai_chunk_to_claude_delta(openai_chunk):
    """Extracts a Claude-formatted content delta from an OpenAI streaming chunk."""
    text_delta = openai_chunk.get("choices", [{}])[0].get("delta", {}).get("content")
    if text_delta:
        return {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text_delta}
        }
    return None

def get_claude_stop_reason_from_openai_chunk(openai_chunk):
    """Extracts and maps the stop reason from a final OpenAI chunk."""
    finish_reason = openai_chunk.get("choices", [{}])[0].get("finish_reason")
    if finish_reason:
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "stop_sequence",
            "tool_calls": "tool_use",
        }
        return stop_reason_map.get(finish_reason, "stop_sequence")
    return None


def convert_gemini_chunk_to_openai(gemini_chunk, model_name):
    """
    Converts a single Gemini streaming chunk to OpenAI-compatible SSE format.
    Returns None if the chunk doesn't map to an OpenAI event.
    """
    try:
        # Generate a consistent ID for the stream
        stream_id = f"chatcmpl-gemini-{random.randint(10000, 99999)}"
        created_time = int(time.time())

        openai_chunk_payload = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }
            ]
        }

        # Parse the chunk if it's a string
        if isinstance(gemini_chunk, str):
            try:
                gemini_chunk = json.loads(gemini_chunk)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                return None

        if not isinstance(gemini_chunk, dict):
            logging.warning(f"Invalid Gemini chunk received: {gemini_chunk}")
            return None

        # Extract candidates
        candidates = gemini_chunk.get("candidates", [])
        if not candidates:
            return None

        first_candidate = candidates[0]
        
        # Check for finish reason
        if "finishReason" in first_candidate:
            finish_reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length", 
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "OTHER": "stop"
            }
            gemini_finish_reason = first_candidate["finishReason"]
            finish_reason = finish_reason_map.get(gemini_finish_reason, "stop")
            openai_chunk_payload["choices"][0]["finish_reason"] = finish_reason
            # openai_chunk_payload["choices"][0]["delta"] = {}
            # Extract content delta
            content = first_candidate.get("content", {})
            parts = content.get("parts", [])
            
            if parts and "text" in parts[0]:
                text_delta = parts[0]["text"]
                logging.debug(f"Gemini text delta: {text_delta}")
                openai_chunk_payload["choices"][0]["delta"]["content"] = text_delta
        else:
            # Extract content delta
            content = first_candidate.get("content", {})
            parts = content.get("parts", [])

            if parts and "text" in parts[0]:
                text_delta = parts[0]["text"]
                logging.debug(f"Gemini text delta: {text_delta}")
                openai_chunk_payload["choices"][0]["delta"]["content"] = text_delta

        # Format as SSE string
        sse_string = f"data: {json.dumps(openai_chunk_payload)}\n\n"
        return sse_string

    except Exception as e:
        logging.error(f"Error converting Gemini chunk to OpenAI format: {e}", exc_info=True)
        error_payload = {
            "id": f"chatcmpl-error-{random.randint(10000, 99999)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": f"[PROXY ERROR: Failed to convert upstream chunk - {str(e)}]"},
                "finish_reason": "stop"
            }]
        }
        return f"data: {json.dumps(error_payload)}\n\n"

# Guards the round-robin counters on load_balance_url. Waitress serves with
# ~100 threads, and the counter read-modify-write below is not atomic; without
# this lock concurrent requests lose increments and skew the load balancing.
_load_balance_lock = threading.Lock()


def load_balance_url(model_name: str) -> tuple:
    """
    Load balance requests for a model across all subAccounts that have it deployed.

    Args:
        model_name: Name of the model to load balance

    Returns:
        Tuple of (selected_url, subaccount_name, resource_group, final_model_name)

    Raises:
        ValueError: If no subAccounts have the requested model
    """
    # Initialize counters dictionary if it doesn't exist
    if not hasattr(load_balance_url, "counters"):
        load_balance_url.counters = {}

    # Get list of subAccounts that have this model
    if model_name not in proxy_config.model_to_subaccounts or not proxy_config.model_to_subaccounts[model_name]:
        # Model not found, raise error
        logging.error(f"Model '{model_name}' not found in any subAccount")
        raise ValueError(f"Model '{model_name}' not available in any subAccount")

    subaccount_names = proxy_config.model_to_subaccounts[model_name]

    # Reserve a round-robin slot atomically. Only the counter bookkeeping is
    # inside the lock; the (unchanging) config lookups afterward stay outside it.
    with _load_balance_lock:
        # Create counter for this model if it doesn't exist
        if model_name not in load_balance_url.counters:
            load_balance_url.counters[model_name] = 0

        # Select subAccount using round-robin
        subaccount_index = load_balance_url.counters[model_name] % len(subaccount_names)
        selected_subaccount = subaccount_names[subaccount_index]

        # Increment counter for next request
        load_balance_url.counters[model_name] += 1

        # Reserve the URL slot for the selected subAccount up front so the whole
        # selection is consistent under concurrency. The URL list length is read
        # here; if it turns out empty we raise below (outside the lock).
        subaccount = proxy_config.subaccounts[selected_subaccount]
        url_list = subaccount.normalized_models.get(model_name, [])
        url_index = None
        if url_list:
            url_counter_key = f"{selected_subaccount}:{model_name}"
            if url_counter_key not in load_balance_url.counters:
                load_balance_url.counters[url_counter_key] = 0
            url_index = load_balance_url.counters[url_counter_key] % len(url_list)
            load_balance_url.counters[url_counter_key] += 1

    if not url_list:
        logging.error(f"Model '{model_name}' listed for subAccount '{selected_subaccount}' but no URLs found")
        raise ValueError(f"Configuration error: No URLs for model '{model_name}' in subAccount '{selected_subaccount}'")

    selected_url = url_list[url_index]

    # Get resource group for the selected subAccount
    resource_group = subaccount.resource_group

    logging.info(f"Selected subAccount '{selected_subaccount}' and URL '{selected_url}' for model '{model_name}'")
    return selected_url, selected_subaccount, resource_group, model_name

def handle_claude_request(payload, model="3.5-sonnet"):
    """Handle Claude model request with multi-subAccount support.
    
    Args:
        payload: Request payload from client
        model: The model name to use
        
    Returns:
        Tuple of (endpoint_url, modified_payload, subaccount_name)
    """
    stream = payload.get("stream", True)  
    logging.info(f"handle_claude_request: model={model} stream={stream}")
    
    # Get the selected URL, subaccount and resource group using our load balancer
    try:
        selected_url, subaccount_name, _, model = load_balance_url(model)
    except ValueError as e:
        logging.error(f"Failed to load balance URL for model '{model}': {e}")
        raise ValueError(f"No valid Claude model found for '{model}' in any subAccount")
    
    # Determine the endpoint path based on model and streaming settings
    if stream:
        # Check if the model is Claude 3.7 or 4 for streaming endpoint
        if is_claude_37_or_4(model):
            endpoint_path = "/converse-stream"
        else:
            endpoint_path = "/invoke-with-response-stream"
    else:
        # Check if the model is Claude 3.7 or 4
        if is_claude_37_or_4(model):
            endpoint_path = "/converse"
        else:
            endpoint_path = "/invoke"
    
    endpoint_url = f"{selected_url.rstrip('/')}{endpoint_path}"
    
    # Convert the payload to the right format
    if is_claude_37_or_4(model):
        modified_payload = convert_openai_to_claude37(payload)
    else:
        modified_payload = convert_openai_to_claude(payload)
    
    logging.info(f"handle_claude_request: {endpoint_url} (subAccount: {subaccount_name})")
    return endpoint_url, modified_payload, subaccount_name

def handle_gemini_request(payload, model="gemini-2.5-pro"):
    """Handle Gemini model request with multi-subAccount support.
    
    Args:
        payload: Request payload from client
        model: The model name to use
        
    Returns:
        Tuple of (endpoint_url, modified_payload, subaccount_name)
    """
    stream = payload.get("stream", True)  # Default to True if 'stream' is not provided
    logging.info(f"handle_gemini_request: model={model} stream={stream}")
    
    # Get the selected URL, subaccount and resource group using our load balancer
    try:
        selected_url, subaccount_name, _, model = load_balance_url(model)
    except ValueError as e:
        logging.error(f"Failed to load balance URL for model '{model}': {e}")
        raise ValueError(f"No valid Gemini model found for '{model}' in any subAccount")
    
    # Extract the model name for the endpoint (e.g., "gemini-2.5-pro" from the model)
    # The endpoint format is: /models/{model}:generateContent
    model_endpoint_name = model
    if ":" in model:
        model_endpoint_name = model.split(":")[0]
    
    # Determine the endpoint path based on streaming settings
    if stream:
        endpoint_path = f"/models/{model_endpoint_name}:streamGenerateContent"
    else:
        endpoint_path = f"/models/{model_endpoint_name}:generateContent"
    
    endpoint_url = f"{selected_url.rstrip('/')}{endpoint_path}"
    
    # Convert the payload to Gemini format
    modified_payload = convert_openai_to_gemini(payload)
    
    logging.info(f"handle_gemini_request: {endpoint_url} (subAccount: {subaccount_name})")
    return endpoint_url, modified_payload, subaccount_name

def handle_default_request(payload, model="gpt-4o"):
    """Handle default (non-Claude, non-Gemini) model request with multi-subAccount support.

    Args:
        payload: Request payload from client
        model: The model name to use

    Returns:
        Tuple of (endpoint_url, modified_payload, subaccount_name)
    """
    # Get the selected URL, subaccount and resource group using our load balancer
    try:
        selected_url, subaccount_name, _, model = load_balance_url(model)
    except ValueError as e:
        logging.error(f"Failed to load balance URL for model '{model}': {e}")
        # Try with default model if specified model not found
        try:
            fallback_model = "gpt-4o"  # Default fallback
            logging.info(f"Falling back to '{fallback_model}' model")
            selected_url, subaccount_name, _, model = load_balance_url(fallback_model)
        except ValueError:
            raise ValueError(f"No valid model found for '{model}' or fallback in any subAccount")

    # Determine API version based on model
    if any(m in model for m in ["o3", "o4-mini", "o3-mini"]):
        api_version = "2024-12-01-preview"
        # Remove unsupported parameters for o3-mini
        modified_payload = payload.copy()
        if 'temperature' in modified_payload:
            logging.info(f"Removing 'temperature' parameter for o3-mini model.")
            del modified_payload['temperature']
        # Add checks for other potentially unsupported parameters if needed
    else:
        api_version = "2023-05-15"
        modified_payload = payload.copy()

    # Remove reasoning-related parameters for GPT models
    # These parameters are Claude-specific and not supported by OpenAI GPT models
    unsupported_params = ['reasoning', 'reasoning_effort']
    removed_params = []
    for param in unsupported_params:
        if param in modified_payload:
            del modified_payload[param]
            removed_params.append(param)

    if removed_params:
        logging.info(f"Removed unsupported parameters for GPT model '{model}': {', '.join(removed_params)}")

    endpoint_url = f"{selected_url.rstrip('/')}/chat/completions?api-version={api_version}"

    logging.info(f"handle_default_request: {endpoint_url} (subAccount: {subaccount_name})")
    return endpoint_url, modified_payload, subaccount_name

@app.route('/v1/chat/completions', methods=['OPTIONS'])
def proxy_openai_stream2():
    logging.info("OPTIONS:Received request to /v1/chat/completions")
    logging.debug(f"Request headers: {request.headers}")
    logging.debug(f"Request payload as string: {request.data.decode('utf-8')}")
    return jsonify({
        "id": "gen-1747041021-KLZff2aBrJPmV6L1bZf1",
        "provider": "OpenAI",
        "model": "gpt-4o",
        "object": "chat.completion",
        "created": 1747041021,
        "choices": [
            {
                "logprobs": None,
                "finish_reason": "stop",
                "native_finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi.",
                    "refusal": None,
                    "reasoning": None
                }
            }
        ],
        "system_fingerprint": "fp_f5bdcc3276",
        "usage": {
            "prompt_tokens": 26,
            "completion_tokens": 3,
            "total_tokens": 29,
            "prompt_tokens_details": {
                "cached_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0
            }
        }
    }), 204

@app.route('/health', methods=['GET'])
def health_check():
    """Lightweight health check endpoint that doesn't require token verification."""
    return jsonify({
        "status": "ok",
        "timestamp": int(time.time()),
        "server": "sap-ai-core-llm-proxy"
    }), 200

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    """Lists all available models across all subAccounts."""
    logging.info("Received request to /v1/models")
    logging.info(f"Request headers: {request.headers}")
    # logging.info(f"Request payload: {request.get_json()}")

    # if not verify_request_token(request):
    #     logging.info("Unauthorized request to list models.")
    #     return jsonify({"error": "Unauthorized"}), 401

    # Collect all available models from all subAccounts
    models = []
    timestamp = int(time.time())

    for model_name in proxy_config.model_to_subaccounts.keys():
        models.append({
            "id": model_name,
            "object": "model",
            "created": timestamp,
            "owned_by": "sap-ai-core"
        })

    return jsonify({"object": "list", "data": models}), 200

content_type="Application/json"
@app.route('/v1/chat/completions', methods=['POST'])
def proxy_openai_stream():
    """Main handler for chat completions endpoint with multi-subAccount support."""
    start_time = time.time()
    request_id = f"{int(start_time * 1000)}-{random.randint(1000, 9999)}"

    _size = request_payload_size(request)
    logging.info("Received request to /v1/chat/completions (payload size: %s / %s bytes, ~%s tokens)",
                 _format_bytes(_size), _size, request_token_estimate(request))
    logging.debug(f"Request headers: {request.headers}")
    logging.debug("Request body:\n%s", _LazyJSON(request.get_json(), indent=4))

    # Verify client authentication token
    if not verify_request_token(request):
        logging.info("Unauthorized request received. Token verification failed.")
        return jsonify({"error": "Unauthorized"}), 401

    # Extract model from the request payload
    payload = request.json
    model = payload.get("model")
    if not model:
        logging.warning("No model specified in request, using default model")
        model = "gpt-5.4"  # Default model

    # Detect coding agent for file logging
    _agent = _detect_agent(request)

    # Stage 1: Log raw client request
    RequestFileLogger.log(request_id, model, 1, "client_request", payload, agent=_agent)

    # Check if model is available in any subAccount
    if model not in proxy_config.model_to_subaccounts:
        logging.warning(f"Model '{model}' not found in any subAccount, falling back to default")
        model = "gpt-5.4"  # Fallback model
        if model not in proxy_config.model_to_subaccounts:
            return jsonify({"error": f"Model '{model}' not available in any subAccount."}), 404

    # Check streaming mode
    is_stream = payload.get("stream", False)
    logging.info(f"Model: {model}, Streaming: {is_stream}")

    try:
        # Handle request based on model type
        if is_claude_model(model):
            endpoint_url, modified_payload, subaccount_name = handle_claude_request(payload, model)
        elif is_gemini_model(model):
            endpoint_url, modified_payload, subaccount_name = handle_gemini_request(payload, model)
        else:
            endpoint_url, modified_payload, subaccount_name = handle_default_request(payload, model)

        # Stage 2: Log transformed request (what we send to backend)
        RequestFileLogger.log(request_id, model, 2, "transformed_request", modified_payload, agent=_agent)

        # Get token for the selected subAccount
        subaccount_token = fetch_token(subaccount_name)
        
        # Get resource group for the selected subAccount
        resource_group = proxy_config.subaccounts[subaccount_name].resource_group
        
        # Get service key for tenant ID
        service_key = proxy_config.subaccounts[subaccount_name].service_key
        
        # Prepare headers for the backend request
        headers = {
            "AI-Resource-Group": resource_group,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {subaccount_token}",
            "AI-Tenant-Id": service_key.identityzoneid
        }
        
        logging.info(f"Forwarding request to {endpoint_url} with subAccount '{subaccount_name}'")
        
        # Handle non-streaming requests
        if not is_stream:
            return handle_non_streaming_request(endpoint_url, headers, modified_payload, model, subaccount_name, request_id=request_id, agent=_agent)

        # Handle streaming requests
        return Response(
            stream_with_context(generate_streaming_response(
                endpoint_url, headers, modified_payload, model, subaccount_name, request_id=request_id, agent=_agent
            )),
            content_type='text/event-stream'
        )
    
    except ValueError as err:
        logging.error(f"Value error during request handling: {err}")
        return jsonify({"error": str(err)}), 400
    
    except Exception as err:
        logging.error(f"Unexpected error during request handling: {err}", exc_info=True)
        return jsonify({"error": str(err)}), 500


# ============================================================
# OpenAI Responses API (/v1/responses)
# ============================================================

def _gen_resp_id():
    return f"resp_{uuid.uuid4().hex[:48]}"

def _gen_msg_id():
    return f"msg_{uuid.uuid4().hex[:48]}"

def _gen_item_id():
    return f"item_{uuid.uuid4().hex[:48]}"


def _normalize_response_text_parts(content):
    """Convert assistant content into Responses API output_text parts."""
    if not content:
        return []

    if isinstance(content, str):
        return [{"type": "output_text", "text": content, "annotations": []}]

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": "output_text", "text": part, "annotations": []})
                continue
            if not isinstance(part, dict):
                continue

            part_type = part.get("type")
            if part_type in ("text", "input_text", "output_text"):
                parts.append({
                    "type": "output_text",
                    "text": part.get("text", ""),
                    "annotations": part.get("annotations", [])
                })
            elif part_type == "refusal":
                parts.append({
                    "type": "refusal",
                    "refusal": part.get("refusal") or part.get("text", "")
                })
        return parts

    return []


def _build_response_usage(usage_data):
    """Build Responses API usage object from chat/completions usage."""
    input_tokens = usage_data.get("prompt_tokens", 0)
    output_tokens = usage_data.get("completion_tokens", 0)
    total_tokens = usage_data.get("total_tokens", input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0}
    }


def _build_response_object(resp_id, model, created_at, status, output, usage):
    """Build a consistent Responses API top-level object."""
    output_text = "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": usage,
        "user": None,
        "metadata": {},
        "output_text": output_text
    }


def convert_responses_input_to_messages(payload):
    """Convert Responses API input format to chat/completions messages."""
    messages = []

    # Handle instructions → system message
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_data = payload.get("input")
    if input_data is None:
        return messages

    # Simple string input → single user message
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages

    # Array input - first pass: build intermediate list, then merge consecutive function_calls
    if isinstance(input_data, list):
        raw_messages = []
        for item in input_data:
            if isinstance(item, str):
                raw_messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                item_type = item.get("type")
                role = item.get("role")

                # function_call items need special handling (merge consecutive ones)
                if item_type == "function_call":
                    raw_messages.append({"_type": "function_call", "tool_call": {
                        "id": item.get("call_id", item.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "")
                        }
                    }})

                # function_call_output → tool response
                elif item_type == "function_call_output":
                    raw_messages.append({
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", "")
                    })

                # Standard message format with role (and not a function_call)
                elif role and item_type != "function_call":
                    content = item.get("content", "")
                    if isinstance(content, list):
                        converted_parts = []
                        for part in content:
                            part_type = part.get("type", "")
                            if part_type in ("input_text", "output_text"):
                                converted_parts.append({"type": "text", "text": part.get("text", "")})
                            elif part_type == "input_image":
                                img_url = part.get("image_url") or part.get("url", "")
                                converted_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                            elif part_type == "text":
                                converted_parts.append({"type": "text", "text": part.get("text", "")})
                            elif part_type == "refusal":
                                converted_parts.append({"type": "text", "text": part.get("refusal", "")})
                            else:
                                logging.debug(f"Skipping unknown content part type: {part_type}")
                        if converted_parts:
                            raw_messages.append({"role": role, "content": converted_parts})
                    else:
                        raw_messages.append({"role": role, "content": content})

                # Handle message type items without explicit role at top level
                elif item_type == "message":
                    msg_role = item.get("role", "user")
                    content = item.get("content", "")
                    if isinstance(content, list):
                        converted_parts = []
                        for part in content:
                            part_type = part.get("type", "")
                            if part_type in ("input_text", "output_text", "text"):
                                converted_parts.append({"type": "text", "text": part.get("text", "")})
                            else:
                                logging.debug(f"Skipping content part type in message: {part_type}")
                        if converted_parts:
                            raw_messages.append({"role": msg_role, "content": converted_parts})
                    else:
                        raw_messages.append({"role": msg_role, "content": content})

        # Second pass: merge consecutive function_call items into single assistant messages
        for msg in raw_messages:
            if "_type" in msg and msg["_type"] == "function_call":
                # Check if last message is already an assistant with tool_calls
                if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                    messages[-1]["tool_calls"].append(msg["tool_call"])
                else:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [msg["tool_call"]]
                    })
            else:
                messages.append(msg)

    # Validate: strip orphaned tool_calls that have no matching tool response.
    # The backend rejects assistant messages with tool_calls if the corresponding
    # tool response messages are missing from the conversation.
    tool_response_ids = {
        msg["tool_call_id"] for msg in messages
        if msg.get("role") == "tool" and "tool_call_id" in msg
    }

    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            original_calls = msg["tool_calls"]
            orphaned = [tc for tc in original_calls if tc.get("id") not in tool_response_ids]
            if orphaned:
                logging.warning(f"Stripping {len(orphaned)} orphaned tool_call(s) with no response: "
                                f"{[tc.get('id') for tc in orphaned]}")
            msg["tool_calls"] = [
                tc for tc in original_calls
                if tc.get("id") in tool_response_ids
            ]
            if not msg["tool_calls"]:
                del msg["tool_calls"]
                if msg.get("content") is None:
                    msg["content"] = ""

    # Remove empty assistant messages that had all tool_calls stripped
    messages = [
        msg for msg in messages
        if not (msg.get("role") == "assistant" and msg.get("content") == "" and "tool_calls" not in msg)
    ]

    # Reorder: tool responses must immediately follow their assistant tool_call message.
    # Codex CLI can inject developer/user messages between a tool_call and its response
    # (e.g. permission approvals), which the backend rejects.
    # Build index: tool_call_id → index of that tool response in messages
    tool_msg_indices = {}
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            tool_msg_indices[msg["tool_call_id"]] = i

    # Check if reordering is needed
    needs_reorder = False
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tc_ids = [tc.get("id") for tc in msg["tool_calls"]]
            for offset, tc_id in enumerate(tc_ids):
                expected_idx = i + 1 + offset
                actual_idx = tool_msg_indices.get(tc_id)
                if actual_idx is not None and actual_idx != expected_idx:
                    needs_reorder = True
                    break
            if needs_reorder:
                break

    if needs_reorder:
        # Collect indices of all tool response messages that will be relocated
        relocated_indices = set()
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tc_ids = [tc.get("id") for tc in msg["tool_calls"]]
                for tc_id in tc_ids:
                    idx = tool_msg_indices.get(tc_id)
                    if idx is not None and idx != i + 1:
                        relocated_indices.add(idx)

        reordered = []
        for i, msg in enumerate(messages):
            if i in relocated_indices:
                continue
            reordered.append(msg)
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tc_ids = [tc.get("id") for tc in msg["tool_calls"]]
                for tc_id in tc_ids:
                    idx = tool_msg_indices.get(tc_id)
                    if idx is not None and idx in relocated_indices:
                        reordered.append(messages[idx])
        messages = reordered

    return messages


def convert_responses_tools(tools):
    """Convert Responses API tools to chat/completions tools format."""
    if not tools:
        return None
    converted = []
    for tool in tools:
        tool_type = tool.get("type", "")
        if tool_type == "function":
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                }
            })
    return converted if converted else None


def build_responses_api_response(completion_response, model, resp_id):
    """Convert a chat/completions response to Responses API format."""
    created_at = int(time.time())
    output = []

    if "choices" in completion_response and completion_response["choices"]:
        choice = completion_response["choices"][0]
        message = choice.get("message", {})

        # Handle tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                output.append({
                    "id": tc.get("id", _gen_item_id()),
                    "type": "function_call",
                    "status": "completed",
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "call_id": tc.get("id", _gen_item_id())
                })

        # Handle text content
        content_parts = _normalize_response_text_parts(message.get("content"))
        if content_parts:
            output.append({
                "id": _gen_msg_id(),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": content_parts
            })

    usage = _build_response_usage(completion_response.get("usage", {}))
    return _build_response_object(resp_id, model, created_at, "completed", output, usage)


def generate_responses_streaming(endpoint_url, headers, chat_payload, model, subaccount_name, request_id=None, agent=None):
    """Stream chat/completions and convert to Responses API SSE events."""
    resp_id = _gen_resp_id()
    msg_id = _gen_msg_id()
    created_at = int(time.time())
    sequence_number = 0

    # Build the base response object
    base_response = _build_response_object(resp_id, model, created_at, "in_progress", [], None)

    def emit_event(event_type, data):
        nonlocal sequence_number
        if isinstance(data, dict):
            data["sequence_number"] = sequence_number
        sequence_number += 1
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # Emit initial lifecycle events
    yield emit_event("response.created", {
        "type": "response.created",
        "response": base_response.copy()
    })
    yield emit_event("response.in_progress", {
        "type": "response.in_progress",
        "response": base_response.copy()
    })

    # Defer message/content_part events until we know there's text
    message_item = {
        "id": msg_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": []
    }
    message_emitted = False

    # Now stream from backend
    full_text = ""
    function_calls = {}  # id -> {name, arguments}
    input_tokens = 0
    output_tokens = 0

    try:
        chat_payload["stream"] = True
        # Remove fields not supported by SAP AI Core backend
        for key in list(chat_payload.keys()):
            if key not in ("model", "messages", "stream", "temperature", "top_p",
                           "max_tokens", "tools", "tool_choice", "n", "stop",
                           "presence_penalty", "frequency_penalty"):
                del chat_payload[key]
        response = _http_session.post(
            endpoint_url, headers=headers, json=chat_payload,
            stream=True, timeout=300
        )
        if response.status_code != 200:
            error_body = response.text
            logging.error(f"Backend returned {response.status_code}: {error_body}")
            yield emit_event("error", {
                "type": "error",
                "message": f"Backend error {response.status_code}: {error_body}"
            })
            return

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                logging.debug(f"Responses streaming: skipping non-data line: {line[:100]}")
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                logging.debug("Responses streaming: received [DONE]")
                break
            try:
                chunk = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                continue

            choices = chunk.get("choices", [])
            if not choices:
                # Check for usage-only chunk (sent after choices are done)
                usage = chunk.get("usage")
                if usage:
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                continue
            delta = choices[0].get("delta", {})

            # Handle text content deltas
            content = delta.get("content")
            if content:
                if not message_emitted:
                    yield emit_event("response.output_item.added", {
                        "type": "response.output_item.added",
                        "response_id": resp_id,
                        "output_index": 0,
                        "item": message_item
                    })
                    yield emit_event("response.content_part.added", {
                        "type": "response.content_part.added",
                        "response_id": resp_id,
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []}
                    })
                    message_emitted = True
                full_text += content
                yield emit_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "response_id": resp_id,
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": content
                })

            # Handle tool call deltas
            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                tc_id = tc.get("id")
                func = tc.get("function", {})

                if tc_id:
                    function_calls[tc_index] = {
                        "id": tc_id,
                        "name": func.get("name", ""),
                        "arguments": ""
                    }
                    fc_output_index = (1 if message_emitted else 0) + tc_index
                    yield emit_event("response.output_item.added", {
                        "type": "response.output_item.added",
                        "response_id": resp_id,
                        "output_index": fc_output_index,
                        "item": {
                            "id": tc_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "name": func.get("name", ""),
                            "arguments": "",
                            "call_id": tc_id
                        }
                    })

                if tc_index in function_calls:
                    arg_delta = func.get("arguments", "")
                    if arg_delta:
                        function_calls[tc_index]["arguments"] += arg_delta
                        yield emit_event("response.function_call_arguments.delta", {
                            "type": "response.function_call_arguments.delta",
                            "response_id": resp_id,
                            "item_id": function_calls[tc_index]["id"],
                            "output_index": (1 if message_emitted else 0) + tc_index,
                            "delta": arg_delta
                        })

            # Extract usage if present
            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)

        response.close()

    except Exception as e:
        logging.error(f"Error streaming responses API: {e}", exc_info=True)
        # Don't return - fall through to emit done/completed events
        # so the client sees a proper stream termination

    # Emit done events for text content
    if full_text:
        yield emit_event("response.output_text.done", {
            "type": "response.output_text.done",
            "response_id": resp_id,
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "text": full_text
        })

        yield emit_event("response.content_part.done", {
            "type": "response.content_part.done",
            "response_id": resp_id,
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": full_text, "annotations": []}
        })

        message_item["content"] = [{"type": "output_text", "text": full_text, "annotations": []}]
        message_item["status"] = "completed"
        yield emit_event("response.output_item.done", {
            "type": "response.output_item.done",
            "response_id": resp_id,
            "output_index": 0,
            "item": message_item
        })

    # Emit done events for function calls
    output_items = []
    if full_text:
        output_items.append(message_item)

    for tc_index in sorted(function_calls.keys()):
        tc = function_calls[tc_index]
        fc_item = {
            "id": tc["id"],
            "type": "function_call",
            "status": "completed",
            "name": tc["name"],
            "arguments": tc["arguments"],
            "call_id": tc["id"]
        }
        output_items.append(fc_item)

        yield emit_event("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "response_id": resp_id,
            "item_id": tc["id"],
            "output_index": len(output_items) - 1,
            "arguments": tc["arguments"]
        })

        yield emit_event("response.output_item.done", {
            "type": "response.output_item.done",
            "response_id": resp_id,
            "output_index": len(output_items) - 1,
            "item": fc_item
        })

    # Emit response.completed
    completed_response = _build_response_object(
        resp_id,
        model,
        created_at,
        "completed",
        output_items,
        _build_response_usage({
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        })
    )
    logging.info(f"Responses streaming: emitting response.completed for {resp_id}")
    yield emit_event("response.completed", {
        "type": "response.completed",
        "response": completed_response
    })

    # Stage 3+4: Log assembled streaming response
    if _log_requests_enabled and request_id:
        RequestFileLogger.log(request_id, model, 3, "backend_response", completed_response, agent=_agent)
        RequestFileLogger.log(request_id, model, 4, "client_response", completed_response, agent=_agent)


@app.route('/v1/responses', methods=['POST'])
def proxy_responses_request():
    """Handler for OpenAI Responses API endpoint."""
    start_time = time.time()
    request_id = f"{int(start_time * 1000)}-{random.randint(1000, 9999)}"

    _size = request_payload_size(request)
    logging.info("Received request to /v1/responses (payload size: %s / %s bytes, ~%s tokens)",
                 _format_bytes(_size), _size, request_token_estimate(request))
    logging.debug(f"Request headers: {request.headers}")
    logging.debug("Request body:\n%s", _LazyJSON(request.get_json(), indent=4))

    if not verify_request_token(request):
        logging.info("Unauthorized request received. Token verification failed.")
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.json
    model = payload.get("model")
    if not model:
        logging.warning("No model specified in /v1/responses request, using default")
        model = "gpt-5.4"

    if model not in proxy_config.model_to_subaccounts:
        logging.warning(f"Model '{model}' not found, falling back to default")
        model = "gpt-5.4"
        if model not in proxy_config.model_to_subaccounts:
            return jsonify({"error": {"type": "invalid_request_error", "message": f"Model '{model}' not available"}}), 404

    is_stream = payload.get("stream", False)
    logging.info(f"Responses API - Model: {model}, Streaming: {is_stream}")

    # Detect coding agent for file logging
    _agent = _detect_agent(request)

    # Stage 1: Log raw client request
    RequestFileLogger.log(request_id, model, 1, "client_request", payload, agent=_agent)

    # Convert Responses API format to chat/completions format
    messages = convert_responses_input_to_messages(payload)
    chat_payload = {
        "model": model,
        "messages": messages,
        "stream": is_stream
    }

    # Copy over compatible parameters
    if "temperature" in payload:
        chat_payload["temperature"] = payload["temperature"]
    if "top_p" in payload:
        chat_payload["top_p"] = payload["top_p"]
    if "max_output_tokens" in payload:
        chat_payload["max_tokens"] = payload["max_output_tokens"]

    # Convert tools
    tools = convert_responses_tools(payload.get("tools"))
    if tools:
        chat_payload["tools"] = tools
        tool_choice = payload.get("tool_choice", "auto")
        chat_payload["tool_choice"] = tool_choice

    try:
        if is_claude_model(model):
            endpoint_url, modified_payload, subaccount_name = handle_claude_request(chat_payload, model)
        elif is_gemini_model(model):
            endpoint_url, modified_payload, subaccount_name = handle_gemini_request(chat_payload, model)
        else:
            endpoint_url, modified_payload, subaccount_name = handle_default_request(chat_payload, model)

        # Stage 2: Log transformed request
        RequestFileLogger.log(request_id, model, 2, "transformed_request", modified_payload, agent=_agent)

        subaccount_token = fetch_token(subaccount_name)
        resource_group = proxy_config.subaccounts[subaccount_name].resource_group
        service_key = proxy_config.subaccounts[subaccount_name].service_key

        headers = {
            "AI-Resource-Group": resource_group,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {subaccount_token}",
            "AI-Tenant-Id": service_key.identityzoneid
        }

        logging.info(f"Forwarding /v1/responses request to {endpoint_url} with subAccount '{subaccount_name}'")

        if not is_stream:
            # Non-streaming: forward, get response, convert to Responses format
            # Clean payload of unsupported fields
            for key in list(modified_payload.keys()):
                if key not in ("model", "messages", "stream", "temperature", "top_p",
                               "max_tokens", "tools", "tool_choice", "n", "stop",
                               "presence_penalty", "frequency_penalty"):
                    del modified_payload[key]
            resp = _http_session.post(endpoint_url, headers=headers, json=modified_payload, timeout=300)
            if resp.status_code != 200:
                error_body = resp.text
                logging.error(f"Backend returned {resp.status_code} for /v1/responses: {error_body}")
                return jsonify({"error": {"type": "server_error", "message": f"Backend error: {error_body}"}}), resp.status_code
            completion = resp.json()
            resp_id = _gen_resp_id()
            response_data = build_responses_api_response(completion, model, resp_id)

            # Stage 3: Log raw backend response
            RequestFileLogger.log(request_id, model, 3, "backend_response", completion, agent=_agent)
            # Stage 4: Log transformed response sent to client
            RequestFileLogger.log(request_id, model, 4, "client_response", response_data, agent=_agent)

            return jsonify(response_data), 200

        # Streaming
        return Response(
            stream_with_context(generate_responses_streaming(
                endpoint_url, headers, modified_payload, model, subaccount_name, request_id=request_id, agent=_agent
            )),
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            }
        )

    except ValueError as err:
        logging.error(f"Value error in /v1/responses: {err}")
        return jsonify({"error": {"type": "invalid_request_error", "message": str(err)}}), 400

    except Exception as err:
        logging.error(f"Unexpected error in /v1/responses: {err}", exc_info=True)
        return jsonify({"error": {"type": "server_error", "message": str(err)}}), 500


@app.route('/v1/messages', methods=['POST'])
def proxy_claude_request():
    """Handles requests that are compatible with the Anthropic Claude Messages API using SAP AI SDK."""
    # Track request timing and ID for logging
    start_time = time.time()
    request_id = f"{int(start_time * 1000)}-{random.randint(1000, 9999)}"

    _size = request_payload_size(request)
    logging.info("[%s] Received request to /v1/messages (payload size: %s / %s bytes, ~%s tokens)",
                 request_id, _format_bytes(_size), _size, request_token_estimate(request))
    logging.debug(f"Request headers: {request.headers}")
    logging.debug("Request body:\n%s", _LazyJSON(request.get_json(), indent=4))

    # Validate API key using proxy config authentication
    api_key = request.headers.get("X-Api-Key", "")
    if not api_key:
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    
    if not verify_request_token(request):
        return jsonify({
            "type": "error",
            "error": { "type": "authentication_error", "message": "Invalid API Key provided." }
        }), 401

    # Get request body and extract model
    request_json = request.get_json(cache=False)
    request_model = request_json.get("model")
    logging.info(f"request_model is: {request_model}")

    # Detect coding agent for file logging
    _agent = _detect_agent(request)

    # Stage 1: Log raw client request
    RequestFileLogger.log(request_id, request_model or "unknown", 1, "client_request", request_json, agent=_agent)

    if not request_model:
        return jsonify({
            "type": "error", 
            "error": {"type": "invalid_request_error", "message": "Missing 'model' parameter"}
        }), 400

    # Validate model availability
    try:
        selected_url, subaccount_name, resource_group, model = load_balance_url(request_model)
    except ValueError as e:
        # Model not found, try to find any Claude/Sonnet model as fallback
        logging.warning(f"Model '{request_model}' not found, searching for Claude/Sonnet fallback")

        # Search for available Claude/Sonnet models
        fallback_model = None
        available_models = list(proxy_config.model_to_subaccounts.keys())

        # First, try the preferred default: claude 4.6 opus
        if "anthropic--claude-4.6-opus" in available_models:
            fallback_model = "anthropic--claude-4.6-opus"
            logging.info(f"Using default Claude 4.6 Opus as fallback")

        # If not available, search by priority: opus > sonnet, newer > older
        if not fallback_model:
            priority_versions = ["4.7", "4.6", "4.5", "4-", "3.7", "3.5", "3-"]
            for version in priority_versions:
                for model_name in available_models:
                    model_lower = model_name.lower()
                    if "claude" in model_lower and "opus" in model_lower and version in model_name:
                        fallback_model = model_name
                        logging.info(f"Found Claude Opus {version} model: {fallback_model}")
                        break
                if fallback_model:
                    break

        # Then try sonnet models
        if not fallback_model:
            priority_versions = ["4.6", "4.5", "4-", "3.7", "3.5", "3-"]
            for version in priority_versions:
                for model_name in available_models:
                    model_lower = model_name.lower()
                    if "claude" in model_lower and "sonnet" in model_lower and version in model_name:
                        fallback_model = model_name
                        logging.info(f"Found Claude Sonnet {version} model: {fallback_model}")
                        break
                if fallback_model:
                    break

        # If no claude+sonnet found, try to find any model with "claude"
        if not fallback_model:
            for model_name in available_models:
                if "claude" in model_name.lower():
                    fallback_model = model_name
                    logging.info(f"Found Claude model: {fallback_model}")
                    break

        # If no claude found, try to find any model with "sonnet"
        if not fallback_model:
            for model_name in available_models:
                if "sonnet" in model_name.lower():
                    fallback_model = model_name
                    logging.info(f"Found Sonnet model: {fallback_model}")
                    break

        if fallback_model:
            try:
                selected_url, subaccount_name, resource_group, model = load_balance_url(fallback_model)
                logging.info(f"Successfully using fallback model '{fallback_model}' for '{request_model}'")
            except ValueError as fallback_error:
                logging.error(f"Fallback model '{fallback_model}' failed: {fallback_error}")
                return jsonify({
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": f"Model '{request_model}' not available and fallback failed"}
                }), 400
        else:
            logging.error(f"No Claude/Sonnet models available for fallback")
            return jsonify({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": f"Model '{request_model}' not available and no Claude/Sonnet fallback found"}
            }), 400

    # Check if this is an Anthropic model that should use the SDK
    if not is_claude_model(model):
        logging.warning(f"Model '{model}' is not a Claude model, falling back to original implementation")
        # Fall back to original implementation for non-Claude models
        return proxy_claude_request_original()

    logging.info(f"Request from Claude API for model: {model}")
    
    # Extract streaming flag
    stream = request_json.get("stream", False)
    
    try:
        # Use cached SAP AI SDK client for the model
        logging.info(f"Obtaining SAP AI SDK client for model: {model}")
        bedrock = get_sapaicore_sdk_client(model)
        logging.info("SAP AI SDK client ready (cached)")
        
        # Get the conversation messages
        conversation = request_json.get("messages", [])
        logging.debug(f"Original conversation: {conversation}")

        thinking_cfg_preview = request_json.get("thinking")
        logging.info(
            "Claude request context: stream=%s, messages=%s, has_thinking=%s",
            stream,
            len(conversation) if isinstance(conversation, list) else "unknown",
            isinstance(thinking_cfg_preview, dict),
        )
        
        # Process conversation to handle empty text content and image compression
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                items_to_remove = []
                for i, item in enumerate(content):
                    if item.get("type") == "text" and (not item.get("text") or item.get("text") == ""):
                        # Mark empty text items for removal
                        items_to_remove.append(i)
                    elif (item.get("type") == "image" and 
                          item.get("source", {}).get("type") == "base64"):
                        # Compress image data if available (would need ImageCompressor utility)
                        image_data = item.get("source", {}).get("data")
                        if image_data:
                            # Note: ImageCompressor would need to be imported/implemented
                            # For now, keeping original data
                            logging.debug("Image data found in message content")
                
                # Remove empty text items (in reverse order to maintain indices)
                for i in reversed(items_to_remove):
                    content.pop(i)

        # Prepare the request body for Bedrock
        body = request_json.copy()

        # Remove model and stream from body as they're handled separately
        body.pop("model", None)
        body.pop("stream", None)

        # Extract system messages from messages array into top-level "system" parameter.
        # Bedrock's Anthropic API does NOT support "role": "system" inside messages —
        # it requires system prompts as a top-level "system" parameter.
        if "messages" in body:
            system_messages = [m for m in body["messages"] if m.get("role") == "system"]
            if system_messages:
                body["messages"] = [m for m in body["messages"] if m.get("role") != "system"]
                # Collect system content (may be string or list of content blocks)
                system_parts = []
                for sm in system_messages:
                    content = sm.get("content", "")
                    if isinstance(content, str):
                        system_parts.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        system_parts.extend(content)
                # Merge with any existing top-level system parameter
                existing_system = body.get("system")
                if existing_system is not None:
                    if isinstance(existing_system, str):
                        system_parts.insert(0, {"type": "text", "text": existing_system})
                    elif isinstance(existing_system, list):
                        system_parts = existing_system + system_parts
                body["system"] = system_parts
                logging.info("Extracted %d system message(s) from messages array into top-level 'system' parameter", len(system_messages))

        # Add required anthropic_version for Bedrock
        body["anthropic_version"] = "bedrock-2023-05-31"

        # Remove unsupported fields for Bedrock (e.g., context_management)
        if "context_management" in body:
            body.pop("context_management", None)
            logging.info("Removed unsupported field 'context_management' from request body")
        

        sanitize_bedrock_tools(body, request_id=request_id)
        sanitize_bedrock_tool_result_content(body, request_id=request_id)

        # Ensure max_tokens obeys thinking budget constraints
        thinking_cfg = body.get("thinking")
        raw_max_tokens = body.get("max_tokens")
        max_tokens_value = None
        if raw_max_tokens is not None:
            try:
                max_tokens_value = int(raw_max_tokens)
            except (TypeError, ValueError):
                logging.warning(f"Invalid max_tokens value '{raw_max_tokens}' in request; resetting to None")
                max_tokens_value = None

        if isinstance(thinking_cfg, dict):
            budget_tokens = thinking_cfg.get("budget_tokens")
            if isinstance(budget_tokens, int):
                required_min_tokens = budget_tokens + 1
                if max_tokens_value is None or max_tokens_value <= budget_tokens:
                    body["max_tokens"] = required_min_tokens
                    logging.info(
                        "Adjusted max_tokens to %s to satisfy thinking.budget_tokens=%s",
                        required_min_tokens,
                        budget_tokens,
                    )
                else:
                    logging.debug(
                        "max_tokens=%s already greater than thinking.budget_tokens=%s",
                        max_tokens_value,
                        budget_tokens,
                    )
            else:
                logging.debug("No integer thinking.budget_tokens found in request")
        elif thinking_cfg is not None:
            logging.debug("Ignoring non-dict thinking config in request body")

        if body.get("max_tokens") is not None:
            logging.info(
                "Final max_tokens for model %s request: %s",
                model,
                body["max_tokens"],
            )
        else:
            logging.info("No max_tokens specified after adjustment for model %s", model)

        # Remove output_config which is not supported by AWS Bedrock
        if "output_config" in body:
            output_config = body.pop("output_config")
            logging.debug("[%s] Removed unsupported output_config field from request: %s", request_id, _LazyJSON(output_config))
            logging.debug(f"[{request_id}] Note: AWS Bedrock doesn't support output_config/json_schema format constraints")

        # Convert body to JSON string for Bedrock API
        body_json = json.dumps(body)

        # Stage 2: Log transformed request (what we send to Bedrock)
        RequestFileLogger.log(request_id, model, 2, "transformed_request", body, agent=_agent)

        # Pretty-print for debugging only when DEBUG is enabled (lazy).
        logging.debug("Request body for Bedrock (pretty):\n%s", _LazyJSON(body, indent=2, ensure_ascii=False))

        if stream:
            # Extract user info for logging (before generator context)
            auth_header = request.headers.get("Authorization", "unknown")
            user_id = f"{auth_header[:20]}..." if auth_header and len(auth_header) > 20 else auth_header
            ip_address = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown")

            # Handle streaming response
            def stream_generate():
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                _msg_chunks = []  # Accumulate chunks for request file logging

                try:
                    response = bedrock.invoke_model_with_response_stream(body=body_json)
                    response_body = response.get("body")

                    if response_body is not None:
                        for event in response_body:
                            chunk = json.loads(event["chunk"]["bytes"])
                            logging.debug(f"Streaming chunk: {chunk}")

                            if _log_requests_enabled:
                                _msg_chunks.append(chunk)

                            chunk_type = chunk.get("type")

                            # Extract token usage from message_delta chunks
                            if chunk_type == "message_delta":
                                usage = chunk.get("usage", {})
                                if usage:
                                    completion_tokens = usage.get("output_tokens", completion_tokens)
                                    logging.debug(f"[{request_id}] Extracted tokens from message_delta: completion={completion_tokens}")

                            # Extract token usage from message_start chunks
                            elif chunk_type == "message_start":
                                message = chunk.get("message", {})
                                usage = message.get("usage", {})
                                if usage:
                                    prompt_tokens = usage.get("input_tokens", 0)
                                    completion_tokens = usage.get("output_tokens", 0)
                                    total_tokens = prompt_tokens + completion_tokens
                                    logging.debug(f"[{request_id}] Extracted tokens from message_start: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

                            # Handle different chunk types according to Claude streaming format
                            if chunk_type == "message_start":
                                yield f"event: message_start\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            elif chunk_type == "content_block_start":
                                yield f"event: content_block_start\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            elif chunk_type == "content_block_delta":
                                yield f"event: content_block_delta\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            elif chunk_type == "content_block_stop":
                                yield f"event: content_block_stop\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            elif chunk_type == "message_delta":
                                yield f"event: message_delta\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            elif chunk_type == "message_stop":
                                yield f"event: message_stop\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                yield "data: [DONE]\n\n"
                                break

                    # Calculate final total if not set
                    if total_tokens == 0:
                        total_tokens = prompt_tokens + completion_tokens

                    # Calculate request duration
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Log token usage at the end of stream
                    log_token_usage(
                        model=model,
                        subaccount_name=subaccount_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        user_id=user_id,
                        ip_address=ip_address,
                        is_streaming=True,
                        duration_ms=duration_ms,
                        request_id=request_id
                    )

                    # Stage 3+4: Log assembled streaming response (pass-through for Claude)
                    if _log_requests_enabled and _msg_chunks:
                        assembled = _assemble_stream_response(_msg_chunks, model, prompt_tokens, completion_tokens, total_tokens)
                        RequestFileLogger.log(request_id, model, 3, "backend_response", assembled, agent=_agent)
                        RequestFileLogger.log(request_id, model, 4, "client_response", assembled, agent=_agent)

                except Exception as e:
                    logging.error(f"Error in streaming response: {e}", exc_info=True)
                    error_chunk = {
                        "type": "error",
                        "error": {"type": "api_error", "message": str(e)}
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

            return Response(stream_generate(), mimetype="text/event-stream"), 200
            
        else:
            # Handle non-streaming response
            response = bedrock.invoke_model(body=body_json)
            response_body = response.get("body")
            
            if response_body is not None:
                # Read the response body
                chunk_data = ''
                for event in response_body:
                    if isinstance(event, bytes):
                        chunk_data += event.decode("utf-8")
                    else:
                        chunk_data += str(event)
                
                if chunk_data:
                    final_response = json.loads(chunk_data)
                    logging.debug(f"Non-streaming response: {final_response}")

                    # Extract token usage
                    usage = final_response.get("usage", {})
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    total_tokens = prompt_tokens + completion_tokens
                    logging.debug(f"[{request_id}] Non-streaming token usage - prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

                    # Calculate request duration
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Extract user info for logging
                    auth_header = request.headers.get("Authorization", "unknown")
                    user_id = f"{auth_header[:20]}..." if auth_header and len(auth_header) > 20 else auth_header
                    ip_address = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown")

                    # Log token usage
                    log_token_usage(
                        model=model,
                        subaccount_name=subaccount_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        user_id=user_id,
                        ip_address=ip_address,
                        is_streaming=False,
                        duration_ms=duration_ms,
                        request_id=request_id
                    )

                    # Stage 3+4: Log response (pass-through for Claude non-streaming)
                    RequestFileLogger.log(request_id, model, 3, "backend_response", final_response, agent=_agent)
                    RequestFileLogger.log(request_id, model, 4, "client_response", final_response, agent=_agent)

                    return jsonify(final_response), 200
                else:
                    return jsonify({}), 200
            else:
                return jsonify({}), 200

    except Exception as e:
        logging.error(f"Error handling Anthropic proxy request using SDK: {e}", exc_info=True)
        error_dict = {
            "type": "error",
            "error": {
                "type": "api_error", 
                "message": str(e)
            }
        }
        return jsonify(error_dict), 500


@app.route('/v1/messages/count_tokens', methods=['POST'])
def count_tokens():
    """Anthropic Messages token-counting endpoint.

    Claude Code and the Anthropic SDK call this before sending a request to
    estimate the prompt size. SAP AI Core's Bedrock backend has no native
    count_tokens, so we return a tiktoken-based estimate in the Anthropic
    response shape: {"input_tokens": <int>}.
    """
    request_id = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    _size = request_payload_size(request)
    logging.info("[%s] Received request to /v1/messages/count_tokens (payload size: %s / %s bytes)",
                 request_id, _format_bytes(_size), _size)

    if not verify_request_token(request):
        return jsonify({
            "type": "error",
            "error": {"type": "authentication_error", "message": "Invalid API Key provided."}
        }), 401

    request_json = request.get_json(silent=True) or {}
    if not request_json.get("model"):
        return jsonify({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "Missing 'model' parameter"}
        }), 400

    try:
        input_tokens = estimate_input_tokens(request_json)
    except Exception as e:
        logging.error(f"[{request_id}] Error estimating tokens: {e}", exc_info=True)
        return jsonify({
            "type": "error",
            "error": {"type": "api_error", "message": f"Failed to count tokens: {e}"}
        }), 500

    logging.info("[%s] count_tokens estimate: %d input tokens (tiktoken approximation)", request_id, input_tokens)
    return jsonify({"input_tokens": input_tokens}), 200


def proxy_claude_request_original():
    """Original implementation preserved as fallback."""
    logging.info("Using original Claude request implementation")
    
    if not verify_request_token(request):
        return jsonify({
            "type": "error",
            "error": { "type": "authentication_error", "message": "Invalid API Key provided." }
        }), 401

    payload = request.json
    model = payload.get("model")
    if not model:
        return jsonify({"type": "error", "error": {"type": "invalid_request_error", "message": "Missing 'model' parameter"}}), 400

    is_stream = payload.get("stream", False)
    logging.info(f"Claude API request for model: {model}, Streaming: {is_stream}")

    try:
        base_url, subaccount_name, resource_group, model = load_balance_url(model)
        subaccount_token = fetch_token(subaccount_name)

        # Convert incoming Claude payload to the format expected by the backend model
        if is_gemini_model(model):
            backend_payload = convert_claude_request_to_gemini(payload)
            endpoint_path = f"/models/{model}:streamGenerateContent" if is_stream else f"/models/{model}:generateContent"
        elif is_claude_model(model):
            backend_payload = convert_claude_request_for_bedrock(payload)
            if is_stream:
                endpoint_path = "/converse-stream" if is_claude_37_or_4(model) else "/invoke-with-response-stream"
            else:
                endpoint_path = "/converse" if is_claude_37_or_4(model) else "/invoke"
        else:  # Assume OpenAI-compatible
            backend_payload = convert_claude_request_to_openai(payload)
            api_version = "2024-12-01-preview" if any(m in model for m in ["o3", "o4-mini", "o3-mini"]) else "2023-05-15"
            endpoint_path = f"/chat/completions?api-version={api_version}"

        endpoint_url = f"{base_url.rstrip('/')}{endpoint_path}"

        service_key = proxy_config.subaccounts[subaccount_name].service_key
        headers = {
            "AI-Resource-Group": resource_group,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {subaccount_token}",
            "AI-Tenant-Id": service_key.identityzoneid
        }

        # Handle anthropic-specific headers
        for h in ['anthropic-version', 'anthropic-beta']:
            if h in request.headers:
                headers[h] = request.headers[h]
        
        # Add default anthropic-beta header for Claude streaming if not already present
        if is_claude_model(model) and is_stream:
            existing_beta = request.headers.get('anthropic-beta', '')
            if 'fine-grained-tool-streaming-2025-05-14' not in existing_beta:
                if existing_beta:
                    # Append to existing anthropic-beta header
                    headers['anthropic-beta'] = f"{existing_beta},fine-grained-tool-streaming-2025-05-14"
                else:
                    # Set new anthropic-beta header
                    headers['anthropic-beta'] = 'fine-grained-tool-streaming-2025-05-14'

        logging.info(f"Forwarding converted request to {endpoint_url} for subAccount '{subaccount_name}'")

        if not is_stream:
            backend_response = _http_session.post(endpoint_url, headers=headers, json=backend_payload, timeout=600)
            backend_response.raise_for_status()
            backend_json = backend_response.json()
            backend_response.close()  # Close connection

            if is_gemini_model(model):
                final_response = convert_gemini_response_to_claude(backend_json, model)
            elif is_claude_model(model):
                final_response = backend_json
            else:
                final_response = convert_openai_response_to_claude(backend_json)

            
            # Log the response for debug purposes
            logging.debug("Final response to client: %s", _LazyJSON(final_response, indent=2))


            return jsonify(final_response), backend_response.status_code
        else:
            return Response(
                stream_with_context(generate_claude_streaming_response(
                    endpoint_url, headers, backend_payload, model, subaccount_name
                )),
                content_type='text/event-stream'
            )

    except ValueError as err:
        logging.error(f"Value error during Claude request handling: {err}")
        return jsonify({"type": "error", "error": {"type": "invalid_request_error", "message": str(err)}}), 400
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error in Claude request: {err}")
        try:
            return jsonify(err.response.json()), err.response.status_code
        except:
            return jsonify({"error": str(err)}), err.response.status_code if err.response else 500
    except Exception as err:
        logging.error(f"Unexpected error during Claude request handling: {err}", exc_info=True)
        return jsonify({"type": "error", "error": {"type": "api_error", "message": "An unexpected error occurred."}}), 500


def handle_non_streaming_request(url, headers, payload, model, subaccount_name, request_id=None, agent=None):
    """Handle non-streaming request to backend API.

    Args:
        url: Backend API endpoint URL
        headers: Request headers
        payload: Request payload
        model: Model name
        subaccount_name: Name of the selected subAccount
        request_id: Optional caller-provided request ID for correlation

    Returns:
        Flask response with the API result
    """
    response = None  # Initialize for cleanup in finally block
    start_time = time.time()  # Track request duration
    if request_id is None:
        request_id = f"{int(start_time * 1000)}-{random.randint(1000, 9999)}"  # Generate unique request ID

    try:
        # Log the raw request body and payload being forwarded
        logging.debug("[%s] Raw request received (non-streaming): %s", request_id, _LazyJSON(request.json, indent=2))
        logging.debug("[%s] Forwarding payload to API (non-streaming): %s", request_id, _LazyJSON(payload, indent=2))

        # Make request to backend API using session
        response = _http_session.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        logging.info(f"Non-streaming request succeeded for model '{model}' using subAccount '{subaccount_name}'")
        
        # Process response based on model type
        backend_json = response.json()
        if is_claude_model(model):
            final_response = convert_claude_to_openai(backend_json, model)
        elif is_gemini_model(model):
            final_response = convert_gemini_to_openai(backend_json, model)
        else:
            final_response = backend_json

        # Stage 3: Log raw backend response
        RequestFileLogger.log(request_id, model, 3, "backend_response", backend_json, agent=agent)
        # Stage 4: Log transformed response sent to client
        RequestFileLogger.log(request_id, model, 4, "client_response", final_response, agent=agent)
        
        # Extract token usage
        total_tokens = final_response.get("usage", {}).get("total_tokens", 0)
        prompt_tokens = final_response.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = final_response.get("usage", {}).get("completion_tokens", 0)

        # Calculate request duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Extract user info for logging
        auth_header = request.headers.get("Authorization", "unknown")
        user_id = f"{auth_header[:20]}..." if auth_header and len(auth_header) > 20 else auth_header
        ip_address = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown")

        # Log token usage with structured format
        log_token_usage(
            model=model,
            subaccount_name=subaccount_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            user_id=user_id,
            ip_address=ip_address,
            is_streaming=False,
            duration_ms=duration_ms,
            request_id=request_id
        )

        return jsonify(final_response), 200
    
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error in non-streaming request: {err}")
        if err.response:
            logging.error(f"Error response status: {err.response.status_code}")
            logging.error(f"Error response headers: {dict(err.response.headers)}")
            logging.error(f"Error response body: {err.response.text}")
            try:
                error_data = err.response.json()
                logging.error(f"Error response JSON: {json.dumps(error_data, indent=2)}")
                return jsonify(error_data), err.response.status_code
            except json.JSONDecodeError:
                return jsonify({"error": err.response.text}), err.response.status_code
        return jsonify({"error": str(err)}), 500
    
    except Exception as err:
        logging.error(f"Error in non-streaming request: {err}", exc_info=True)
        return jsonify({"error": str(err)}), 500

    finally:
        # Always close the response to release the connection
        if response is not None:
            response.close()


def generate_streaming_response(url, headers, payload, model, subaccount_name, request_id=None, agent=None):
    """Generate streaming response from backend API.

    Args:
        url: Backend API endpoint URL
        headers: Request headers
        payload: Request payload
        model: Model name
        subaccount_name: Name of the selected subAccount
        request_id: Optional caller-provided request ID for correlation

    Yields:
        SSE formatted response chunks
    """
    # Track request timing
    start_time = time.time()
    if request_id is None:
        request_id = f"{int(start_time * 1000)}-{random.randint(1000, 9999)}"

    # Extract user info for logging (before generator context)
    auth_header = request.headers.get("Authorization", "unknown")
    user_id = f"{auth_header[:20]}..." if auth_header and len(auth_header) > 20 else auth_header
    ip_address = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown")

    # Log the raw request body and payload being forwarded
    logging.debug("[%s] Raw request received (streaming): %s", request_id, _LazyJSON(request.json, indent=2))
    logging.debug("[%s] Forwarding payload to API (streaming): %s", request_id, _LazyJSON(payload, indent=2))

    buffer = ""
    total_tokens = 0
    prompt_tokens = 0  # Initialize prompt tokens
    completion_tokens = 0  # Initialize completion tokens
    claude_metadata = {}  # For Claude 3.7 metadata
    chunk = None  # Initialize chunk variable to avoid reference errors
    _stream_chunks = []  # Accumulate chunks for request file logging

    # Hoist model-type predicates out of the per-chunk loops below — the model
    # doesn't change mid-stream, so these substring scans only need to run once.
    is_claude = is_claude_model(model)
    is_claude_new = is_claude and is_claude_37_or_4(model)
    is_gemini = is_gemini_model(model)

    # Make streaming request to backend using session
    with _http_session.post(url, headers=headers, json=payload, stream=True, timeout=600) as response:
        try:
            response.raise_for_status()

            # --- Claude 3.7/4/4.5/4.6 Streaming Logic ---
            if is_claude_new:
                logging.info(f"Using Claude streaming (model: {model}) for subAccount '{subaccount_name}'")
                for line_bytes in response.iter_lines():
                    if line_bytes:
                        line = line_bytes.decode('utf-8')
                        if line.startswith("data: "):
                            line_content = line.replace("data: ", "").strip()
                            # logging.info(f"Raw data chunk from Claude API: {line_content}")
                            try:
                                line_content = ast.literal_eval(line_content)
                                line_content = json.dumps(line_content)
                                claude_dict_chunk = json.loads(line_content)
                                chunk_type = claude_dict_chunk.get("type")
                                
                                # Handle metadata chunk
                                if chunk_type == "metadata":
                                    claude_metadata = claude_dict_chunk.get("metadata", {})
                                    logging.debug(f"Received Claude 3.7 metadata from '{subaccount_name}': {claude_metadata}")
                                    continue
                                
                                # Convert chunk to OpenAI format
                                openai_sse_chunk_str = convert_claude37_chunk_to_openai(claude_dict_chunk, model)
                                if openai_sse_chunk_str:
                                    if _log_requests_enabled:
                                        _stream_chunks.append(claude_dict_chunk)
                                    yield openai_sse_chunk_str
                            except Exception as e:
                                logging.error(f"Error processing Claude 3.7 chunk from '{subaccount_name}': {e}", exc_info=True)
                                error_payload = {
                                    "id": f"chatcmpl-error-{random.randint(10000, 99999)}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": f"[PROXY ERROR: Failed to process upstream data]"},
                                        "finish_reason": "stop"
                                    }]
                                }
                                yield f"data: {json.dumps(error_payload)}\n\n"
                
                # Extract token counts from metadata
                if claude_metadata and isinstance(claude_metadata.get("usage"), dict):
                    total_tokens = claude_metadata["usage"].get("totalTokens", 0)
                    prompt_tokens = claude_metadata["usage"].get("inputTokens", 0)
                    completion_tokens = claude_metadata["usage"].get("outputTokens", 0)
            
            # --- Gemini Streaming Logic ---
            elif is_gemini:
                logging.info(f"Using Gemini streaming for subAccount '{subaccount_name}'")
                for line_bytes in response.iter_lines():
                    if line_bytes:
                        line = line_bytes.decode('utf-8')
                        logging.debug(f"Gemini raw line received: {line}")

                        # Process Gemini streaming lines
                        line_content = ""
                        if line.startswith("data: "):
                            line_content = line.replace("data: ", "").strip()
                            logging.debug(f"Gemini data line content: {line_content}")
                        elif line.strip():
                            # Handle lines without "" prefix
                            line_content = line.strip()
                            logging.debug(f"Gemini line content (no prefix): {line_content}")

                        if line_content and line_content != "[DONE]":
                            try:
                                gemini_chunk = json.loads(line_content)
                                logging.debug("Gemini parsed chunk: %s", _LazyJSON(gemini_chunk, indent=2))

                                # Convert chunk to OpenAI format
                                openai_sse_chunk_str = convert_gemini_chunk_to_openai(gemini_chunk, model)
                                if openai_sse_chunk_str:
                                    logging.debug(f"Gemini converted to OpenAI chunk: {openai_sse_chunk_str}")
                                    if _log_requests_enabled:
                                        _stream_chunks.append(gemini_chunk)
                                    yield openai_sse_chunk_str
                                else:
                                    logging.debug(f"Gemini chunk conversion returned None")

                                # Extract token usage from usageMetadata if available
                                if "usageMetadata" in gemini_chunk:
                                    usage_metadata = gemini_chunk["usageMetadata"]
                                    total_tokens = usage_metadata.get("totalTokenCount", 0)
                                    prompt_tokens = usage_metadata.get("promptTokenCount", 0)
                                    completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
                                    logging.debug(f"Gemini token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
                                    
                            except json.JSONDecodeError as e:
                                logging.error(f"Error parsing Gemini chunk from '{subaccount_name}': {e}")
                                logging.error(f"Problematic line content: {line_content}")
                                continue
                            except Exception as e:
                                logging.error(f"Error processing Gemini chunk from '{subaccount_name}': {e}", exc_info=True)
                                logging.error(f"Problematic chunk: {gemini_chunk if 'gemini_chunk' in locals() else 'Failed to parse'}")
                                error_payload = {
                                    "id": f"chatcmpl-error-{random.randint(10000, 99999)}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": f"[PROXY ERROR: Failed to process upstream data]"},
                                        "finish_reason": "stop"
                                    }]
                                }
                                yield f"{json.dumps(error_payload)}\n\n"
            
            # --- Other Models (including older Claude) ---
            else:
                # 8 KiB chunks instead of 128 B: far fewer iterations/syscalls per
                # token stream while still flushing promptly to the client.
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        if is_claude:  # Older Claude
                            buffer += chunk.decode('utf-8')
                            while "data: " in buffer:
                                try:
                                    start = buffer.index("data: ") + len("data: ")
                                    end = buffer.index("\n\n", start)
                                    json_chunk_str = buffer[start:end].strip()
                                    buffer = buffer[end + 2:]

                                    # Convert Claude chunk to OpenAI format
                                    openai_sse_chunk_str = convert_claude_chunk_to_openai(json_chunk_str, model)
                                    if _log_requests_enabled:
                                        try:
                                            _stream_chunks.append(json.loads(json_chunk_str))
                                        except (json.JSONDecodeError, ValueError):
                                            _stream_chunks.append(json_chunk_str)
                                    yield openai_sse_chunk_str.encode('utf-8')
                                    
                                    # Parse token usage if available
                                    try:
                                        claude_data = json.loads(json_chunk_str)
                                        if "usage" in claude_data:
                                            prompt_tokens = claude_data["usage"].get("input_tokens", 0)
                                            completion_tokens = claude_data["usage"].get("output_tokens", 0)
                                            total_tokens = prompt_tokens + completion_tokens
                                    except json.JSONDecodeError:
                                        pass
                                except ValueError:
                                    break  # Not enough data in buffer
                                except Exception as e:
                                    logging.error(f"Error processing claude chunk: {e}", exc_info=True)
                                    break
                        else:  # OpenAI-like models
                            if _log_requests_enabled:
                                try:
                                    _stream_chunks.append(chunk.decode('utf-8'))
                                except Exception:
                                    pass
                            yield chunk
                            try:
                                # Try to extract token counts from final chunk
                                if chunk:
                                    chunk_text = chunk.decode('utf-8')
                                    if '"finish_reason":' in chunk_text:
                                        for line in chunk_text.strip().split('\n'):
                                            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                                                try:
                                                    data = json.loads(line[6:])
                                                    if "usage" in data:
                                                        total_tokens = data["usage"].get("total_tokens", 0)
                                                        prompt_tokens = data["usage"].get("prompt_tokens", 0)
                                                        completion_tokens = data["usage"].get("completion_tokens", 0)
                                                except json.JSONDecodeError:
                                                    pass
                            except Exception:
                                pass
            
            # Calculate request duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Log token usage at the end of the stream
            log_token_usage(
                model=model,
                subaccount_name=subaccount_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                user_id=user_id,
                ip_address=ip_address,
                is_streaming=True,
                duration_ms=duration_ms,
                request_id=request_id
            )

            # Stage 3+4: Log assembled streaming response (backend=client for streaming)
            if _log_requests_enabled and _stream_chunks:
                assembled = _assemble_stream_response(_stream_chunks, model, prompt_tokens, completion_tokens, total_tokens)
                RequestFileLogger.log(request_id, model, 3, "backend_response", assembled, agent=agent)
                RequestFileLogger.log(request_id, model, 4, "client_response", assembled, agent=agent)

            # Standard stream end
            yield "data: [DONE]\n\n"
            
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP Error in streaming response from '{subaccount_name}': {err}")
            if hasattr(err, 'response') and err.response is not None:
                logging.error(f"Error response status: {err.response.status_code}")
                logging.error(f"Error response headers: {dict(err.response.headers)}")
                try:
                    error_content = err.response.text
                    logging.error(f"Error response body: {error_content}")
                    # Try to parse as JSON for better formatting
                    try:
                        error_json = err.response.json()
                        logging.error(f"Error response JSON: {json.dumps(error_json, indent=2)}")
                    except json.JSONDecodeError:
                        pass
                except Exception as e:
                    logging.error(f"Could not read error response content: {e}")
            
            error_payload = {
                "id": f"error-{random.randint(10000, 99999)}",
                "object": "error",
                "created": int(time.time()),
                "model": model,
                "error": {
                    "message": str(err),
                    "type": "http_error",
                    "code": err.response.status_code if hasattr(err, 'response') and err.response else 500,
                    "subaccount": subaccount_name
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as err:
            logging.error(f"Error in streaming response from '{subaccount_name}': {err}", exc_info=True)
            error_payload = {
                "id": f"error-{random.randint(10000, 99999)}",
                "object": "error",
                "created": int(time.time()),
                "model": model,
                "error": {
                    "message": str(err),
                    "type": "proxy_error",
                    "code": 500,
                    "subaccount": subaccount_name
                }
            }
            # Use strings directly without referencing chunk to avoid errors
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"


def generate_claude_streaming_response(url, headers, payload, model, subaccount_name):
    """
    Generates a streaming response in the Anthropic Claude Messages API format.
    If the backend is a Claude model, it passes the stream through.
    If the backend is Gemini or OpenAI, it converts their SSE stream to Claude's format.
    """
    logging.info(f"Starting Claude streaming response for model '{model}' using subAccount '{subaccount_name}'")
    logging.debug("Forwarding payload to API (Claude streaming): %s", _LazyJSON(payload, indent=2))
    logging.debug(f"Request URL: {url}")
    logging.debug(f"Request headers: {headers}")

    # If the backend is already a Claude model, we need to convert the response format.
    if is_claude_model(model):
        logging.info(f"Backend is Claude model, converting response format for '{model}'")
        try:
            with _http_session.post(url, headers=headers, json=payload, stream=True, timeout=600) as response:
                response.raise_for_status()
                logging.debug(f"Claude backend response status: {response.status_code}")
                
                # Send message_start event
                message_start_data = {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{random.randint(10000, 99999)}", 
                        "type": "message", 
                        "role": "assistant",
                        "content": [], 
                        "model": model, 
                        "stop_reason": None, 
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0}
                    }
                }
                message_start_event = f"event: message_start\ndata: {json.dumps(message_start_data)}\n\n"
                yield message_start_event.encode('utf-8')

                # Send content_block_start event
                content_block_start_data = {
                    "type": "content_block_start", 
                    "index": 0, 
                    "content_block": {"type": "text", "text": ""}
                }
                content_block_start_event = f"event: content_block_start\ndata: {json.dumps(content_block_start_data)}\n\n"
                yield content_block_start_event.encode('utf-8')
                
                chunk_count = 0
                stop_reason = None
                
                for line in response.iter_lines():
                    chunk_count += 1
                    if not line:
                        continue
                        
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    logging.debug(f"Claude backend chunk {chunk_count}: {line_str}")
                    
                    if line_str.startswith('data: '):
                        data_content = line_str[6:].strip()  # Remove 'data: ' prefix
                        
                        # Handle different data formats
                        if data_content == '[DONE]':
                            break
                        
                        try:
                            # Try to parse as JSON first
                            try:
                                parsed_data = json.loads(data_content)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, try to evaluate as Python dict
                                # This handles the case where single quotes are used instead of double quotes
                                parsed_data = ast.literal_eval(data_content)
                            
                            # Convert Claude backend format to standard Claude API format
                            if 'contentBlockDelta' in parsed_data:
                                # Extract text from the delta and format it the same way as OpenAI conversion
                                text_content = parsed_data['contentBlockDelta']['delta'].get('text', '')
                                if text_content:
                                    delta_data = {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {"type": "text_delta", "text": text_content}
                                    }
                                    delta_event = f"event: content_block_delta\ndata: {json.dumps(delta_data)}\n\n"
                                    yield delta_event.encode('utf-8')
                                
                            elif 'contentBlockStop' in parsed_data:
                                content_block_stop_data = {
                                    "type": "content_block_stop",
                                    "index": parsed_data['contentBlockStop'].get('contentBlockIndex', 0)
                                }
                                content_block_stop_event = f"event: content_block_stop\ndata: {json.dumps(content_block_stop_data)}\n\n"
                                yield content_block_stop_event.encode('utf-8')
                                
                            elif 'messageStop' in parsed_data:
                                stop_reason = parsed_data['messageStop'].get('stopReason', 'end_turn')
                                
                            elif 'metadata' in parsed_data:
                                # Extract token usage information
                                usage_info = parsed_data.get('metadata', {}).get('usage', {})
                                message_delta_data = {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": stop_reason or "end_turn", "stop_sequence": None},
                                    "usage": {"output_tokens": usage_info.get('outputTokens', 0)}
                                }
                                message_delta_event = f"event: message_delta\ndata: {json.dumps(message_delta_data)}\n\n"
                                yield message_delta_event.encode('utf-8')
                                
                                message_stop_event = f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                                yield message_stop_event.encode('utf-8')
                                
                        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                            logging.warning(f"Could not parse Claude backend data: {data_content}, error: {e}")
                            continue
                
                logging.info(f"Claude backend conversion completed with {chunk_count} chunks")
        except Exception as e:
            logging.error(f"Error in Claude backend conversion for '{model}': {e}", exc_info=True)
            raise
        return

    # For other models, we need to convert the stream to Claude's event format.
    logging.info(f"Converting non-Claude model '{model}' stream to Claude format")
    
    # 1. Send message_start event
    message_start_data = {
        "type": "message_start",
        "message": {
            "id": f"msg_{random.randint(10000, 99999)}", "type": "message", "role": "assistant",
            "content": [], "model": model, "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }
    }
    message_start_event = f"event: message_start\ndata: {json.dumps(message_start_data)}\n\n"
    logging.debug(f"Sending message_start event: {message_start_event}")
    yield message_start_event.encode('utf-8')

    # 2. Send content_block_start event
    content_block_start_data = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    content_block_start_event = f"event: content_block_start\ndata: {json.dumps(content_block_start_data)}\n\n"
    logging.debug(f"Sending content_block_start event: {content_block_start_event}")
    yield content_block_start_event.encode('utf-8')

    stop_reason = None
    chunk_count = 0
    delta_count = 0

    try:
        with _http_session.post(url, headers=headers, json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()
            logging.debug(f"Backend response status: {response.status_code}")
            logging.debug(f"Backend response headers: {dict(response.headers)}")

            # 3. Iterate and yield content_block_delta events
            for line in response.iter_lines():
                chunk_count += 1
                logging.debug(f"Processing backend chunk {chunk_count}: {line}")
                
                if not line or not line.strip().startswith(b'data:'):
                    logging.debug(f"Skipping non-data line {chunk_count}: {line}")
                    continue

                line_str = line.decode('utf-8', errors='ignore')[5:].strip()
                logging.debug(f"Extracted line content: {line_str}")
                
                if line_str == "[DONE]":
                    logging.info(f"Received [DONE] signal at chunk {chunk_count}")
                    break

                try:
                    backend_chunk = json.loads(line_str)
                    logging.debug("Parsed backend chunk: %s", _LazyJSON(backend_chunk, indent=2))

                    claude_delta = None
                    if is_gemini_model(model):
                        logging.debug(f"Converting Gemini chunk to Claude delta")
                        claude_delta = convert_gemini_chunk_to_claude_delta(backend_chunk)
                        if not stop_reason: 
                            stop_reason = get_claude_stop_reason_from_gemini_chunk(backend_chunk)
                            if stop_reason:
                                logging.debug(f"Extracted stop reason from Gemini: {stop_reason}")
                    else:  # Assume OpenAI-compatible
                        logging.debug(f"Converting OpenAI chunk to Claude delta")
                        claude_delta = convert_openai_chunk_to_claude_delta(backend_chunk)
                        if not stop_reason: 
                            stop_reason = get_claude_stop_reason_from_openai_chunk(backend_chunk)
                            if stop_reason:
                                logging.debug(f"Extracted stop reason from OpenAI: {stop_reason}")

                    if claude_delta:
                        delta_count += 1
                        delta_event = f"event: content_block_delta\ndata: {json.dumps(claude_delta)}\n\n"
                        logging.debug(f"Sending content_block_delta {delta_count}: {delta_event}")
                        yield delta_event.encode('utf-8')
                    else:
                        logging.debug(f"No delta extracted from chunk {chunk_count}")

                except json.JSONDecodeError as e:
                    logging.warning(f"Could not decode JSON from stream chunk {chunk_count}: {line_str}, error: {e}")
                    continue
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_count}: {e}", exc_info=True)
                    continue

            logging.info(f"Processed {chunk_count} chunks, generated {delta_count} deltas")

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error in Claude streaming conversion for '{model}': {e}", exc_info=True)
        if hasattr(e, 'response') and e.response:
            logging.error(f"Error response status: {e.response.status_code}")
            logging.error(f"Error response body: {e.response.text}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in Claude streaming conversion for '{model}': {e}", exc_info=True)
        raise

    # 4. Send stop events
    logging.debug(f"Sending stop events with stop_reason: {stop_reason}")
    
    content_block_stop_event = f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    logging.debug(f"Sending content_block_stop event: {content_block_stop_event}")
    yield content_block_stop_event.encode('utf-8')

    message_delta_data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason or "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 0}  # Token usage is not available in most streams
    }
    message_delta_event = f"event: message_delta\ndata: {json.dumps(message_delta_data)}\n\n"
    logging.debug(f"Sending message_delta event: {message_delta_event}")
    yield message_delta_event.encode('utf-8')

    message_stop_event = f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    logging.debug(f"Sending message_stop event: {message_stop_event}")
    yield message_stop_event.encode('utf-8')
    
    logging.info(f"Claude streaming response completed for model '{model}' with {delta_count} content deltas")


# ------------------------
# Cleanup on shutdown
# ------------------------
def cleanup_on_exit():
    """Cleanup resources on server shutdown"""
    logging.info("Cleaning up HTTP session...")
    _http_session.close()
    logging.info("Server shutdown complete")

atexit.register(cleanup_on_exit)

# Periodic connection pool maintenance
# ------------------------
def maintain_connection_pool():
    """Periodic heartbeat for the HTTP session pool.

    Previously this called ``poolmanager.clear()`` every 5 minutes, which
    discarded *all* pooled keep-alive connections — including warm, healthy
    ones — forcing a fresh TLS handshake on the next request to each host.
    urllib3 already reaps genuinely idle/broken connections on checkout, so
    the periodic clear cost more (re-handshakes under load) than it saved.

    We keep the thread as a lightweight liveness log and no longer tear down
    the pool.
    """
    while True:
        try:
            time.sleep(300)  # Run every 5 minutes
            logging.debug("Connection pool heartbeat (no teardown)")
        except Exception as e:
            logging.error(f"Error in connection pool heartbeat: {e}")


if __name__ == '__main__':
    args = parse_arguments()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled")

    # Enable request/response file logging
    if args.log_requests:
        _log_requests_enabled = True
        os.makedirs(RequestFileLogger.BASE_DIR, exist_ok=True)
        logging.info("Request/response file logging ENABLED -> %s", RequestFileLogger.BASE_DIR)
    
    logging.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Check if this is the new format with subAccounts
    if isinstance(config, ProxyConfig):
        proxy_config = config
        # Initialize all subaccounts and build model mappings
        proxy_config.initialize()
        
        # Get server configuration
        host = proxy_config.host
        port = proxy_config.port
        
        logging.info(f"Loaded multi-subAccount configuration with {len(proxy_config.subaccounts)} subAccounts")
        logging.info(f"Available subAccounts: {', '.join(proxy_config.subaccounts.keys())}")
        logging.info(f"Available models: {', '.join(proxy_config.model_to_subaccounts.keys())}")
    else:
        # Legacy configuration support
        logging.warning("Using legacy configuration format (single subAccount)")
        
        # Initialize global variables for backward compatibility
        service_key_json = config['service_key_json']
        model_deployment_urls = config['deployment_models']
        secret_authentication_tokens = config['secret_authentication_tokens']
        resource_group = config['resource_group']
        
        # Keep model_deployment_urls keys as-is
        normalized_model_deployment_urls = model_deployment_urls.copy()

        # Load service key
        service_key = load_config(service_key_json)

        host = config.get('host', '127.0.0.1')  # Use host from config, default to 127.0.0.1 if not specified
        port = config.get('port', 3001)  # Use port from config, default to 3001 if not specified

        # Initialize the proxy_config for compatibility with new code
        proxy_config.secret_authentication_tokens = secret_authentication_tokens
        proxy_config.host = host
        proxy_config.port = port
        
        # Create a default subAccount
        default_subaccount = SubAccountConfig(
            name="default",
            resource_group=resource_group,
            service_key_json=service_key_json,
            deployment_models=model_deployment_urls
        )
        
        # Add service key
        default_subaccount.service_key = ServiceKey(
            clientid=service_key.get('clientid', ''),
            clientsecret=service_key.get('clientsecret', ''),
            url=service_key.get('url', ''),
            identityzoneid=service_key.get('identityzoneid', '')
        )
        
        # Normalize model names
        default_subaccount.normalized_models = normalized_model_deployment_urls
        
        # Add to proxy_config
        proxy_config.subaccounts["default"] = default_subaccount
        
        # Build model mappings
        proxy_config.build_model_mapping()

    logging.info(f"Starting proxy server on host {host} and port {port}...")
    logging.info(f"API Host: http://{host}:{port}/v1")
    logging.info(f"Available endpoints:")
    logging.info(f"  - Health Check: http://{host}:{port}/health")
    logging.info(f"  - OpenAI Compatible API: http://{host}:{port}/v1/chat/completions")
    logging.info(f"  - Anthropic Claude API: http://{host}:{port}/v1/messages")
    logging.info(f"  - Models Listing: http://{host}:{port}/v1/models")
    logging.info(f"  - Embeddings API: http://{host}:{port}/v1/embeddings")

    # Start connection pool maintenance thread
    maintenance_thread = threading.Thread(target=maintain_connection_pool, daemon=True)
    maintenance_thread.start()
    logging.info("Started connection pool maintenance thread")

    # Start request log cleanup thread (only if logging enabled)
    if _log_requests_enabled:
        cleanup_thread = threading.Thread(target=_cleanup_request_logs, daemon=True)
        cleanup_thread.start()
        logging.info("Started request log cleanup thread (7-day retention)")

    # Try to use waitress if available (production server)
    # Otherwise fall back to Flask dev server with thread limiting
    try:
        from waitress import serve
        logging.info("Using Waitress production server with improved connection handling")
        logging.info("Configuration: threads=100, connection_limit=2000, cleanup_interval=15, channel_timeout=300")
        logging.info("To install waitress: pip3 install waitress")
        serve(app, host=host, port=port,
              threads=100,              # Increased from 40 to handle more concurrent requests
              connection_limit=2000,    # Increased from 1000
              cleanup_interval=15,      # Decreased from 30 for faster cleanup
              channel_timeout=300,      # Increased from 120 for long streaming requests
              recv_bytes=65536,         # Increase receive buffer
              send_bytes=65536,         # Increase send buffer
              asyncore_use_poll=True)   # Use poll() instead of select() for better scalability
    except ImportError:
        logging.warning("Waitress not available, using Flask development server")
        logging.warning("For production use, install waitress: pip3 install waitress")
        logging.warning("Flask dev server has limitations with many persistent connections")
        app.run(host=host, port=port, debug=args.debug, threaded=True)

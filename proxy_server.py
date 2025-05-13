import logging
from flask import Flask, request, jsonify, Response, stream_with_context
import requests
import time
import threading
import json
import base64
import random
import os
from datetime import datetime
import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ServiceKey:
    clientid: str
    clientsecret: str
    url: str

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
            url=key_data.get('url')
        )
        
    def normalize_model_names(self):
        """Normalize model names by removing prefixes like 'anthropic--'"""
        self.normalized_models = {
            key.replace("anthropic--", ""): value 
            for key, value in self.deployment_models.items()
        }

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

app = Flask(__name__)

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
            response = requests.post(token_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            token_data = response.json()
            new_token = token_data.get('access_token')
            
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
    """Verifies the Authorization header from the incoming client request."""
    token = request.headers.get("Authorization")
    logging.info(f"verify_request_token, Token received in request: {token[:15]}..." if token and len(token) > 15 else token)
    
    if not proxy_config.secret_authentication_tokens:
        logging.warning("Client authentication disabled - no tokens configured.")
        return True
        
    if not token:
        logging.error("Missing token in request.")
        return False
        
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
    logging.debug(f"Original OpenAI payload for Claude 3.7 conversion: {json.dumps(payload, indent=2)}")

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
        if role in ["user", "assistant"] :
            if content and isinstance(content, str):
                # Claude /converse expects content as a list of blocks, typically [{"text": "..."}]
                converted_messages.append({
                    "role": role,
                    "content": [{"text": content}]
                })
            elif content and isinstance(content, list): # Handle potential pre-formatted content (less common from OpenAI)
                 logging.warning(f"Received list content for role {role}, attempting to use as is for Claude.")
                 converted_messages.append({
                     "role": role,
                     "content": content # Assume it's already in Claude block format
                 })
            else:
                logging.warning(f"Skipping message for role {role} due to missing or invalid content: {msg}")
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

    logging.debug(f"Converted Claude 3.7 payload: {json.dumps(claude_payload, indent=2)}")
    return claude_payload

def convert_claude_to_openai(response, model):
    # Check if the model name indicates Claude 3.7
    if "3.7" in model:
        logging.info(f"Detected Claude 3.7 model ('{model}'), using convert_claude37_to_openai.")
        return convert_claude37_to_openai(response, model)

    # Proceed with the original Claude conversion logic for other models
    logging.info(f"Using standard Claude conversion for model '{model}'.")

    try:
        logging.info(f"Raw response from Claude API: {json.dumps(response, indent=4)}")

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
        logging.debug(f"Converted response to OpenAI format: {json.dumps(openai_response, indent=4)}")
        return openai_response
    except Exception as e:
        logging.error(f"Error converting Claude response to OpenAI format: {e}")
        return {
            "error": "Invalid response from Claude API",
            "details": str(e)
        }

def convert_claude37_to_openai(response, model_name="claude-3.7"):
    """
    Converts a Claude 3.7 /converse API response payload (non-streaming)
    to the format expected by the OpenAI Chat Completion API.
    """
    try:
        logging.debug(f"Raw response from Claude 3.7 API: {json.dumps(response, indent=2)}")

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
        # Claude 3.7 /converse should provide totalTokens, but calculate as fallback
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
        logging.debug(f"Converted response to OpenAI format: {json.dumps(openai_response, indent=2)}")
        return openai_response

    except Exception as e:
        # Log the error with traceback for better debugging
        logging.error(f"Error converting Claude 3.7 response to OpenAI format: {e}", exc_info=True)
        # Log the problematic response structure that caused the error
        logging.error(f"Problematic Claude response structure: {json.dumps(response, indent=2)}")
        # Return an error structure compliant with OpenAI format
        return {
            "object": "error",
            "message": f"Failed to convert Claude 3.7 response to OpenAI format. Error: {str(e)}. Check proxy logs for details.",
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
        logging.info(f"{model} Raw Claude chunk received: {chunk}")
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
    Converts a single parsed Claude 3.7 /converse-stream chunk (dictionary)
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
                logging.info(f"Parsed Claude chunk: {claude_chunk}")
                claude_chunk = json.loads(claude_chunk)
                logging.info(f"Decoded Claude chunk: {json.dumps(claude_chunk, indent=2)}")
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
            logging.warning(f"Unknown Claude 3.7 chunk type encountered: {chunk_type}. Chunk: {claude_chunk}")
            return None

        # Format as SSE string if a valid payload was constructed
        sse_string = f"data: {json.dumps(openai_chunk_payload)}\n\n"
        return sse_string

    except Exception as e:
        logging.error(f"Error converting Claude 3.7 chunk to OpenAI format: {e}", exc_info=True)
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
    return "claude" in model or "sonnet" in model

def load_balance_url(model_name: str) -> tuple:
    """
    Load balance requests for a model across all subAccounts that have it deployed.
    
    Args:
        model_name: Name of the model to load balance
        
    Returns:
        Tuple of (selected_url, subaccount_name, resource_group)
        
    Raises:
        ValueError: If no subAccounts have the requested model
    """
    # Initialize counters dictionary if it doesn't exist
    if not hasattr(load_balance_url, "counters"):
        load_balance_url.counters = {}
    
    # Get list of subAccounts that have this model
    if model_name not in proxy_config.model_to_subaccounts or not proxy_config.model_to_subaccounts[model_name]:
        logging.error(f"No subAccounts with model '{model_name}' found")
        raise ValueError(f"Model '{model_name}' not available in any subAccount")
    
    subaccount_names = proxy_config.model_to_subaccounts[model_name]
    
    # Create counter for this model if it doesn't exist
    if model_name not in load_balance_url.counters:
        load_balance_url.counters[model_name] = 0
    
    # Select subAccount using round-robin
    subaccount_index = load_balance_url.counters[model_name] % len(subaccount_names)
    selected_subaccount = subaccount_names[subaccount_index]
    
    # Increment counter for next request
    load_balance_url.counters[model_name] += 1
    
    # Get the model URL list from the selected subAccount
    subaccount = proxy_config.subaccounts[selected_subaccount]
    url_list = subaccount.normalized_models.get(model_name, [])
    
    if not url_list:
        logging.error(f"Model '{model_name}' listed for subAccount '{selected_subaccount}' but no URLs found")
        raise ValueError(f"Configuration error: No URLs for model '{model_name}' in subAccount '{selected_subaccount}'")
    
    # Select URL using round-robin within the subAccount
    url_counter_key = f"{selected_subaccount}:{model_name}"
    if url_counter_key not in load_balance_url.counters:
        load_balance_url.counters[url_counter_key] = 0
    
    url_index = load_balance_url.counters[url_counter_key] % len(url_list)
    selected_url = url_list[url_index]
    
    # Increment URL counter for next request
    load_balance_url.counters[url_counter_key] += 1
    
    # Get resource group for the selected subAccount
    resource_group = subaccount.resource_group
    
    logging.info(f"Selected subAccount '{selected_subaccount}' and URL '{selected_url}' for model '{model_name}'")
    return selected_url, selected_subaccount, resource_group

def handle_claude_request(payload, model="3.5-sonnet"):
    """Handle Claude model request with multi-subAccount support.
    
    Args:
        payload: Request payload from client
        model: The model name to use
        
    Returns:
        Tuple of (endpoint_url, modified_payload, subaccount_name)
    """
    stream = payload.get("stream", True)  # Default to True if 'stream' is not provided
    logging.info(f"handle_claude_request: model={model} stream={stream}")
    
    # Get the selected URL, subaccount and resource group using our load balancer
    try:
        selected_url, subaccount_name, _ = load_balance_url(model)
    except ValueError as e:
        logging.error(f"Failed to load balance URL for model '{model}': {e}")
        raise ValueError(f"No valid Claude model found for '{model}' in any subAccount")
    
    # Determine the endpoint path based on model and streaming settings
    if stream:
        # Check if the model name contains '3.7' for streaming endpoint
        if "3.7" in model:
            endpoint_path = "/converse-stream"
        else:
            endpoint_path = "/invoke-with-response-stream"
    else:
        # Check if the model name contains '3.7'
        if "3.7" in model:
            endpoint_path = "/converse"
        else:
            endpoint_path = "/invoke"
    
    endpoint_url = f"{selected_url.rstrip('/')}{endpoint_path}"
    
    # Convert the payload to the right format
    if "3.7" in model:
        modified_payload = convert_openai_to_claude37(payload)
    else:
        modified_payload = convert_openai_to_claude(payload)
    
    logging.info(f"handle_claude_request: {endpoint_url} (subAccount: {subaccount_name})")
    return endpoint_url, modified_payload, subaccount_name

def handle_default_request(payload, model="gpt-4o"):
    """Handle default (non-Claude) model request with multi-subAccount support.
    
    Args:
        payload: Request payload from client
        model: The model name to use
        
    Returns:
        Tuple of (endpoint_url, modified_payload, subaccount_name)
    """
    # Get the selected URL, subaccount and resource group using our load balancer
    try:
        selected_url, subaccount_name, _ = load_balance_url(model)
    except ValueError as e:
        logging.error(f"Failed to load balance URL for model '{model}': {e}")
        # Try with default model if specified model not found
        try:
            fallback_model = "gpt-4o"  # Default fallback
            logging.info(f"Falling back to '{fallback_model}' model")
            selected_url, subaccount_name, _ = load_balance_url(fallback_model)
            model = fallback_model  # Update model to the fallback
        except ValueError:
            raise ValueError(f"No valid model found for '{model}' or fallback in any subAccount")
    
    # Determine API version based on model
    if "o3-mini" in model:
        api_version = "2024-12-01-preview"
        # Remove unsupported parameters for o3-mini
        modified_payload = payload.copy()
        if 'temperature' in modified_payload:
            logging.info(f"Removing 'temperature' parameter for o3-mini model.")
            del modified_payload['temperature']
        # Add checks for other potentially unsupported parameters if needed
    else:
        api_version = "2023-05-15"
        modified_payload = payload
    
    endpoint_url = f"{selected_url.rstrip('/')}/chat/completions?api-version={api_version}"
    
    logging.info(f"handle_default_request: {endpoint_url} (subAccount: {subaccount_name})")
    return endpoint_url, modified_payload, subaccount_name

@app.route('/v1/chat/completions', methods=['OPTIONS'])
def proxy_openai_stream2():
    logging.info("OPTIONS:Received request to /v1/chat/completions")
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request payload as string: {request.data.decode('utf-8')}")
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

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    """Lists all available models across all subAccounts."""
    logging.info("Received request to /v1/models")
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request payload: {request.get_json()}")
    
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
    logging.info("Received request to /v1/chat/completions")
    logging.debug(f"Request headers: {request.headers}")
    logging.debug(f"Request body:\n{json.dumps(request.get_json(), indent=4)}")
    
    # Verify client authentication token
    if not verify_request_token(request):
        logging.info("Unauthorized request received. Token verification failed.")
        return jsonify({"error": "Unauthorized"}), 401

    # Extract model from the request payload
    payload = request.json
    model = payload.get("model")
    if not model:
        logging.warning("No model specified in request, using default model")
        model = "gpt-4o"  # Default model
    
    # Check if model is available in any subAccount
    if model not in proxy_config.model_to_subaccounts:
        logging.warning(f"Model '{model}' not found in any subAccount, falling back to default")
        model = "gpt-4o"  # Fallback model
        if model not in proxy_config.model_to_subaccounts:
            return jsonify({"error": f"Model '{model}' not available in any subAccount."}), 404
    
    # Check streaming mode
    is_stream = payload.get("stream", True)
    logging.info(f"Model: {model}, Streaming: {is_stream}")
    
    try:
        # Handle request based on model type
        if is_claude_model(model):
            endpoint_url, modified_payload, subaccount_name = handle_claude_request(payload, model)
        else:
            endpoint_url, modified_payload, subaccount_name = handle_default_request(payload, model)
        
        # Get token for the selected subAccount
        subaccount_token = fetch_token(subaccount_name)
        
        # Get resource group for the selected subAccount
        resource_group = proxy_config.subaccounts[subaccount_name].resource_group
        
        # Prepare headers for the backend request
        headers = {
            "AI-Resource-Group": resource_group,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {subaccount_token}"
        }
        
        logging.info(f"Forwarding request to {endpoint_url} with subAccount '{subaccount_name}'")
        
        # Handle non-streaming requests
        if not is_stream:
            return handle_non_streaming_request(endpoint_url, headers, modified_payload, model, subaccount_name)
        
        # Handle streaming requests
        return Response(
            stream_with_context(generate_streaming_response(
                endpoint_url, headers, modified_payload, model, subaccount_name
            )),
            content_type='text/event-stream'
        )
    
    except ValueError as err:
        logging.error(f"Value error during request handling: {err}")
        return jsonify({"error": str(err)}), 400
    
    except Exception as err:
        logging.error(f"Unexpected error during request handling: {err}", exc_info=True)
        return jsonify({"error": str(err)}), 500


def handle_non_streaming_request(url, headers, payload, model, subaccount_name):
    """Handle non-streaming request to backend API.
    
    Args:
        url: Backend API endpoint URL
        headers: Request headers
        payload: Request payload
        model: Model name
        subaccount_name: Name of the selected subAccount
    
    Returns:
        Flask response with the API result
    """
    try:
        # Log the raw request body and payload being forwarded
        logging.info(f"Raw request received (non-streaming): {json.dumps(request.json, indent=2)}")
        logging.info(f"Forwarding payload to API (non-streaming): {json.dumps(payload, indent=2)}")
        
        # Make request to backend API
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        logging.info(f"Non-streaming request succeeded for model '{model}' using subAccount '{subaccount_name}'")
        
        # Process response based on model type
        if is_claude_model(model):
            final_response = convert_claude_to_openai(response.json(), model)
        else:
            final_response = response.json()
        
        # Extract token usage
        total_tokens = final_response.get("usage", {}).get("total_tokens", 0)
        prompt_tokens = final_response.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = final_response.get("usage", {}).get("completion_tokens", 0)
        
        # Log token usage with subAccount information
        user_id = request.headers.get("Authorization", "unknown")
        if user_id and len(user_id) > 20:
            user_id = f"{user_id[:20]}..."
        ip_address = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown_ip")
        token_logger.info(f"User: {user_id}, IP: {ip_address}, Model: {model}, SubAccount: {subaccount_name}, "
                          f"PromptTokens: {prompt_tokens}, CompletionTokens: {completion_tokens}, TotalTokens: {total_tokens}")
        
        return jsonify(final_response), 200
    
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error in non-streaming request: {err}")
        if err.response:
            logging.error(f"Error response: {err.response.text}")
            try:
                error_data = err.response.json()
                return jsonify(error_data), err.response.status_code
            except json.JSONDecodeError:
                return jsonify({"error": err.response.text}), err.response.status_code
        return jsonify({"error": str(err)}), 500
    
    except Exception as err:
        logging.error(f"Error in non-streaming request: {err}", exc_info=True)
        return jsonify({"error": str(err)}), 500


def generate_streaming_response(url, headers, payload, model, subaccount_name):
    """Generate streaming response from backend API.
    
    Args:
        url: Backend API endpoint URL
        headers: Request headers
        payload: Request payload
        model: Model name
        subaccount_name: Name of the selected subAccount
    
    Yields:
        SSE formatted response chunks
    """
    # Log the raw request body and payload being forwarded
    logging.info(f"Raw request received (streaming): {json.dumps(request.json, indent=2)}")
    logging.info(f"Forwarding payload to API (streaming): {json.dumps(payload, indent=2)}")
    
    buffer = ""
    total_tokens = 0
    claude_metadata = {}  # For Claude 3.7 metadata
    chunk = None  # Initialize chunk variable to avoid reference errors
    
    # Make streaming request to backend
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as response:
        try:
            response.raise_for_status()
            
            # --- Claude 3.7 Streaming Logic ---
            if is_claude_model(model) and "3.7" in model:
                logging.info(f"Using Claude 3.7 streaming for subAccount '{subaccount_name}'")
                for line_bytes in response.iter_lines():
                    if line_bytes:
                        line = line_bytes.decode('utf-8')
                        if line.startswith("data: "):
                            line_content = line.replace("data: ", "").strip()
                            import ast
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
            
            # --- Other Models (including older Claude) ---
            else:
                for chunk in response.iter_content(chunk_size=128):
                    if chunk:
                        if is_claude_model(model):  # Older Claude
                            buffer += chunk.decode('utf-8')
                            while "data: " in buffer:
                                try:
                                    start = buffer.index("data: ") + len("data: ")
                                    end = buffer.index("\n\n", start)
                                    json_chunk_str = buffer[start:end].strip()
                                    buffer = buffer[end + 2:]
                                    
                                    # Convert Claude chunk to OpenAI format
                                    openai_sse_chunk_str = convert_claude_chunk_to_openai(json_chunk_str, model)
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
            
            # Log token usage at the end of the stream
            user_id = request.headers.get("Authorization", "unknown")
            if user_id and len(user_id) > 20:
                user_id = f"{user_id[:20]}..."
            ip_address = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown_ip")
            
            # Log with subAccount information
            token_logger.info(f"User: {user_id}, IP: {ip_address}, Model: {model}, SubAccount: {subaccount_name}, "
                             f"PromptTokens: {prompt_tokens if 'prompt_tokens' in locals() else 0}, "
                             f"CompletionTokens: {completion_tokens if 'completion_tokens' in locals() else 0}, "
                             f"TotalTokens: {total_tokens} (Streaming)")
            
            # Standard stream end
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

if __name__ == '__main__':
    args = parse_arguments()
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
        
        # Normalize model_deployment_urls keys
        normalized_model_deployment_urls = {
            key.replace("anthropic--", ""): value for key, value in model_deployment_urls.items()
        }

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
            url=service_key.get('url', '')
        )
        
        # Normalize model names
        default_subaccount.normalized_models = normalized_model_deployment_urls
        
        # Add to proxy_config
        proxy_config.subaccounts["default"] = default_subaccount
        
        # Build model mappings
        proxy_config.build_model_mapping()

    logging.info(f"Starting proxy server on host {host} and port {port}...")
    logging.info(f"API Host: http://{host}:{port}/v1")
    app.run(host=host, port=port, debug=False)
import logging
from flask import Flask, request, jsonify, Response, stream_with_context, Request # Added Request for type hinting
import requests # Keep for fetch_token AND manual Anthropic calls
import time
import threading
import json
import base64
import random
import os
from datetime import datetime
import argparse
from typing import Optional, Dict, Any, Generator, List

# Import Litellm (used only for Azure/OpenAI/Gemini models now)
import litellm
from litellm import ModelResponse
import traceback # For detailed error logging

# --- Flask App and Logging Setup ---
app = Flask(__name__)

# Configure basic logging
# Use INFO for production, DEBUG for development/troubleshooting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

# Create a new logger for token usage
token_logger = logging.getLogger('token_usage')
token_logger.setLevel(logging.INFO)
token_logger.propagate = False # Prevent double logging in main logger

# Create a file handler for token usage logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True) # Ensure directory exists

log_file = os.path.join(log_directory, 'token_usage.log')
try:
    # Use append mode 'a' so logs are not overwritten on restart
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    token_logger.addHandler(file_handler)
    logging.info(f"Token usage logging configured to: {log_file}")
except IOError as e:
    logging.error(f"Could not open token usage log file {log_file}: {e}")
    # Decide if this is critical or if the proxy can run without token logging

# --- Global variables ---
backend_auth_token: Optional[str] = None
token_expiry_time: float = 0.0
token_fetch_lock = threading.Lock()

# --- Configuration Placeholders (loaded in __main__) ---
config: Dict[str, Any] = {}
service_key: Optional[Dict[str, str]] = None # REQUIRED for this setup
# This map MUST contain ALL supported models mapped to SAP AI Core deployment base URLs
aicore_deployment_urls: Dict[str, list[str]] = {}
secret_authentication_tokens: list[str] = []
resource_group: str = "" # REQUIRED for SAP AI Core Header


# --- Config Loading and Parsing ---
def load_config(file_path: str) -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        logging.info(f"Successfully loaded configuration from {file_path}")
        return config_data
    except FileNotFoundError:
        logging.critical(f"Configuration file not found: '{file_path}'")
        raise
    except json.JSONDecodeError as e:
         logging.critical(f"Error decoding configuration JSON from '{file_path}': {e}")
         raise
    except Exception as e:
        logging.critical(f"An unexpected error occurred while loading config '{file_path}': {e}")
        raise

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="SAP AI Core Universal LLM Proxy (Hybrid - Litellm/Manual)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file (default: config.json)")
    return parser.parse_args()


# --- Backend Token Fetching (OAuth for SAP AI Core - MANDATORY) ---
def fetch_token() -> str:
    """Fetches or retrieves a cached SAP AI Core authentication token. Raises error on failure."""
    global backend_auth_token, token_expiry_time
    if not service_key: # Should be checked before calling, but safety first
        logging.error("CRITICAL: fetch_token called but service_key config is missing.")
        raise ValueError("OAuth service_key configuration is missing.")

    if not all(k in service_key for k in ['clientid', 'clientsecret', 'url']):
        raise ValueError("Service key config missing required keys ('clientid', 'clientsecret', 'url').")

    with token_fetch_lock:
        current_time = time.time()
        if backend_auth_token and current_time < token_expiry_time:
            logging.debug("Using cached backend token.")
            return backend_auth_token

        logging.info("Fetching new backend token for SAP AI Core.")
        client_id = service_key['clientid']
        client_secret = service_key['clientsecret']
        token_endpoint_base = service_key['url']

        try:
            auth_string = f"{client_id}:{client_secret}"
            encoded_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('ascii')
            token_url = f"{token_endpoint_base.rstrip('/')}/oauth/token?grant_type=client_credentials"
            headers = {"Authorization": f"Basic {encoded_auth_string}"}

            response = requests.post(token_url, headers=headers, timeout=15)
            response.raise_for_status()

            token_data = response.json()
            new_token = token_data.get('access_token')
            expires_in = int(token_data.get('expires_in', 14400)) # Default 4 hours

            if not new_token: raise ValueError("Fetched token is empty")

            backend_auth_token = new_token
            token_expiry_time = current_time + expires_in - 300 # 5-minute buffer
            logging.info(f"Backend token fetched/refreshed successfully.")
            return backend_auth_token

        except requests.exceptions.Timeout as err:
             logging.error(f"Timeout fetching token: {err}")
             backend_auth_token = None; token_expiry_time = 0
             raise TimeoutError(f"Timeout connecting to token endpoint") from err
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP error fetching token: {err.response.status_code} - {err.response.text}")
            backend_auth_token = None; token_expiry_time = 0
            raise ConnectionError(f"HTTP Error {err.response.status_code} fetching token") from err
        except requests.exceptions.RequestException as err:
            logging.error(f"Network/Request error fetching token: {err}")
            backend_auth_token = None; token_expiry_time = 0
            raise ConnectionError(f"Network error fetching token: {err}") from err
        except Exception as err:
            logging.error(f"Unexpected error during token fetching: {err}", exc_info=True)
            backend_auth_token = None; token_expiry_time = 0
            raise RuntimeError(f"Unexpected error processing token response: {err}") from err


# --- Client Request Token Verification ---
def verify_request_token(request: Request) -> bool:
    """Verifies the Authorization header from the incoming client request."""
    if not secret_authentication_tokens:
        logging.warning("PROXY SECURITY: No client auth tokens configured. Allowing request.")
        return True

    client_token = request.headers.get("Authorization")
    log_token_prefix = client_token[:15] + "..." if client_token and len(client_token) > 15 else client_token
    logging.debug(f"Verifying client token. Header prefix: '{log_token_prefix}'")

    if not client_token: logging.warning("Client request missing Authorization header."); return False
    if not any(secret_key in client_token for secret_key in secret_authentication_tokens):
        logging.warning(f"Invalid client token provided."); return False

    logging.debug("Client request token verified successfully.")
    return True


# --- Load Balancing (Round-robin for SAP AI Core BASE URLs) ---
def load_balance_url(urls: list[str], model_key: str) -> Optional[str]:
    """Selects a BASE URL from a list using round-robin. Returns None if list is empty."""
    logging.debug(f"Load balancing for model_key: {model_key} with base urls: {urls}")
    if not urls:
        logging.error(f"No BASE URLs available for AI Core model '{model_key}'.")
        return None

    if not hasattr(load_balance_url, "counters"): load_balance_url.counters = {}
    if model_key not in load_balance_url.counters: load_balance_url.counters[model_key] = 0

    with threading.Lock():
        index = load_balance_url.counters[model_key] % len(urls)
        selected_base_url = urls[index]
        load_balance_url.counters[model_key] += 1

    logging.info(f"Selected AI Core base URL for '{model_key}': {selected_base_url}")
    return selected_base_url


# --- Helper to determine UNDERLYING provider type ---
def get_underlying_provider(model_name: str) -> str:
    """Determines the underlying LLM type (azure, anthropic, gemini) for SAP AI Core routing."""
    model_name_lower = model_name.lower() if model_name else ""
    if "claude" in model_name_lower or "sonnet" in model_name_lower:
        logging.debug(f"Model '{model_name}' mapped to 'anthropic' underlying provider.")
        return "anthropic"
    if "gemini" in model_name_lower:
        logging.debug(f"Model '{model_name}' mapped to 'gemini' underlying provider.")
        return "gemini"
    logging.debug(f"Model '{model_name}' mapped to 'azure' underlying provider.")
    return "azure" # Default for gpt-* and others via AI Core


# --- Token Usage Logging Helper ---
def log_token_usage(request: Request, model_name: str, usage_object: Optional[Dict[str, int]]):
    """Logs token usage details using Litellm's usage object OR manually parsed data."""
    # This function expects an OpenAI-like usage dictionary
    try:
        if usage_object and isinstance(usage_object, dict):
            prompt_tokens = usage_object.get('prompt_tokens', 0)
            completion_tokens = usage_object.get('completion_tokens', 0)
            total_tokens = usage_object.get('total_tokens', int(prompt_tokens or 0) + int(completion_tokens or 0))

            # Get user identifier (sanitize client token)
            client_token = request.headers.get("Authorization", "unknown")
            max_len = 30
            if client_token.lower().startswith("bearer "): token_part = client_token[7:]; user_id = f"Bearer {token_part[:8]}..." if len(token_part)>8 else f"Bearer {token_part}"
            else: user_id = f"{client_token[:max_len]}..." if len(client_token)>max_len else client_token

            ip_address = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown_ip")

            token_logger.info(
                f"User: {user_id}, IP: {ip_address}, Model: {model_name}, "
                f"PromptTokens: {prompt_tokens}, CompletionTokens: {completion_tokens}, TotalTokens: {total_tokens}"
            )
        # Don't log warning if usage was None (e.g., error before completion)
    except Exception as e:
        logging.error(f"Failed to log token usage for model {model_name}: {e}", exc_info=True)


# --- *** MANUAL Conversion Functions for Anthropic *** ---
def convert_openai_to_anthropic_payload(openai_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Converts OpenAI format payload to Anthropic Messages API format for SAP AI Core."""
    logging.debug("Converting OpenAI payload to Anthropic format for AI Core")
    messages = openai_payload.get("messages", [])
    system_prompt = ""
    processed_messages = []

    if messages and messages[0].get("role") == "system":
        system_prompt = messages[0].get("content", "")
        messages = messages[1:]

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        # Basic user/assistant mapping
        if role in ["user", "assistant"]:
             processed_messages.append({"role": role, "content": content})
        else:
             logging.warning(f"Ignoring message with unhandled role for Anthropic: {role}")

    anthropic_payload = {
        # ** ADDED REQUIRED KEY ** - Verify this value with SAP AI Core docs
        "anthropic_version": "bedrock-2023-05-31",
        "messages": processed_messages,
        "max_tokens": openai_payload.get("max_tokens", 4096), # Use provided or default
        # Map other common parameters if they exist
        **{k_out: openai_payload[k_in] for k_in, k_out in {
             "temperature": "temperature",
             "top_p": "top_p",
             "top_k": "top_k", # Note OpenAI uses top_p, Anthropic uses top_k
             "stop": "stop_sequences" # OpenAI 'stop' -> Anthropic 'stop_sequences'
           }.items() if k_in in openai_payload}
    }
    if system_prompt:
        anthropic_payload["system"] = system_prompt

    # Remove None values
    anthropic_payload = {k: v for k, v in anthropic_payload.items() if v is not None}

    logging.debug(f"Converted Anthropic payload (with version): {json.dumps(anthropic_payload)}")
    return anthropic_payload

# --- Make sure this converter ONLY gets the JSON string ---
def convert_anthropic_chunk_to_openai(anthropic_json_str: str, model_name: str) -> Optional[str]:
    """Converts an Anthropic Messages API JSON data chunk string to OpenAI SSE format string."""
    logging.debug(f"Attempting to convert Anthropic JSON chunk: {anthropic_json_str}")
    try:
        # The input is *already* the JSON part, just load it
        data = json.loads(anthropic_json_str)
        # Determine the event type from the JSON data itself
        event_type = data.get("type")
        if not event_type:
            logging.warning(f"Anthropic JSON chunk missing 'type' field: {anthropic_json_str}")
            return None

    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from Anthropic data string: {anthropic_json_str}")
        return None # Or return an error chunk

    # --- Mapping Logic (same as before) ---
    openai_chunk = {
        "id": f"chatcmpl-anthropic-{int(time.time()*1000)}-{random.randint(100,999)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}]
    }
    finish_reason = None; content_delta = None; role = None

    if event_type == "message_start": role = data.get("message", {}).get("role", "assistant")
    elif event_type == "content_block_delta":
        if data.get("delta", {}).get("type") == "text_delta": content_delta = data["delta"].get("text")
    elif event_type == "message_delta":
        stop_reason = data.get("delta", {}).get("stop_reason")
        if stop_reason:
             if stop_reason == "end_turn": finish_reason = "stop"
             elif stop_reason == "max_tokens": finish_reason = "length"
             elif stop_reason == "stop_sequence": finish_reason = "stop"
             else: finish_reason = "stop"
    # Ignore "content_block_start", "content_block_stop", "message_stop" for standard OpenAI format conversion

    # Populate delta
    if role: openai_chunk["choices"][0]["delta"]["role"] = role
    if content_delta: openai_chunk["choices"][0]["delta"]["content"] = content_delta
    if finish_reason: openai_chunk["choices"][0]["finish_reason"] = finish_reason

    # Don't yield empty delta chunks unless setting role or finish reason
    if not openai_chunk["choices"][0]["delta"] and finish_reason is None:
         # Only skip if delta is truly empty (no role, no content)
         if "role" not in openai_chunk["choices"][0]["delta"]:
              logging.debug(f"Skipping empty Anthropic delta chunk (Type: {event_type})")
              return None

    # Format as OpenAI SSE string
    openai_sse_str = f"data: {json.dumps(openai_chunk)}\n\n"
    logging.debug(f"Converted OpenAI SSE chunk: {openai_sse_str.strip()}")
    return openai_sse_str

# --- Flask Routes ---

# OPTIONS handler
@app.route('/v1/chat/completions', methods=['OPTIONS'])
def handle_options():
    resp=jsonify({}); resp.headers.add('Access-Control-Allow-Origin','*'); resp.headers.add('Access-Control-Allow-Headers','Content-Type,Authorization'); resp.headers.add('Access-Control-Allow-Methods','POST,OPTIONS'); resp.headers.add('Access-Control-Max-Age','86400'); return resp, 204

# Model Listing Endpoint
@app.route('/v1/models', methods=['GET'])
def list_models_endpoint():
    if not verify_request_token(request): return jsonify({"error":"Unauthorized"}), 401
    models=[{"id":m,"object":"model","created":int(time.time()),"owned_by":"sap-ai-core"} for m in aicore_deployment_urls.keys()]
    return jsonify({"object": "list", "data": models}), 200


# Main Chat Completions Proxy Endpoint (HYBRID APPROACH)
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions_proxy_endpoint():
    """Proxies chat completions: Uses Litellm for Azure/Gemini, Manual requests for Anthropic."""
    endpoint = "/v1/chat/completions"; request_start_time = time.time(); logging.info(f"POST {endpoint}")

    # 1. Verify Client Token
    if not verify_request_token(request): return jsonify({"error": "Invalid API Key"}), 401

    # 2. Fetch Backend Auth Token (MANDATORY)
    current_backend_token: Optional[str] = None
    try: current_backend_token = fetch_token()
    except Exception as e: logging.error(f"CRITICAL - Failed fetch token: {e}",exc_info=True); return jsonify({"error":"Proxy backend auth failed"}), 503

    # 3. Parse Request Body
    try: request_data = request.get_json(); assert request_data
    except: return jsonify({"error": "Invalid JSON body"}), 400
    logging.debug(f"Request body:\n{json.dumps(request_data, indent=2)}")

    # 4. Extract Parameters
    client_model_name: Optional[str] = request_data.get("model")
    messages: Optional[list] = request_data.get("messages")
    stream: bool = request_data.get("stream", False)
    if not client_model_name: return jsonify({"error": "Missing 'model'"}), 400
    if not messages: return jsonify({"error": "Missing 'messages'"}), 400

    # 5. Validate Model Config & Get Base URL
    if client_model_name not in aicore_deployment_urls: return jsonify({"error": f"Model '{client_model_name}' not configured."}), 404
    base_urls = aicore_deployment_urls[client_model_name]; selected_base_url = load_balance_url(base_urls, client_model_name)
    if not selected_base_url: return jsonify({"error": "Internal config error: No base URL."}), 500

    # 6. Determine Underlying Provider
    underlying_provider = get_underlying_provider(client_model_name)

    # --- Start of try block for the specific request handling ---
    try:
        # 7. --- Branch Logic: Manual Anthropic vs Litellm Azure/Gemini ---
        if underlying_provider == "anthropic":
            # --- MANUAL ANTHROPIC CALL ---
            logging.info(f"Handling '{client_model_name}' manually (Anthropic via AI Core)")

            # Construct URL
            path = "/invoke-with-response-stream" if stream else "/invoke"
            final_endpoint_url = f"{selected_base_url.rstrip('/')}{path}"
            logging.info(f"Using AI Core Anthropic-style URL: {final_endpoint_url}")

            # Prepare Headers
            headers = {
                "Authorization": f"Bearer {current_backend_token}",
                "Content-Type": "application/json",
                "AI-Resource-Group": resource_group,
                # Add Accept based on streaming
                "Accept": "text/event-stream" if stream else "application/json"
            }

            # Convert Payload
            anthropic_payload = convert_openai_to_anthropic_payload(request_data)
            logging.debug(f"Manual Anthropic Request Payload: {json.dumps(anthropic_payload)}")

            # Make Request using requests library
            response = requests.post( final_endpoint_url, headers=headers, json=anthropic_payload, stream=stream, timeout=600 )
            logging.info(f"Manual request to {final_endpoint_url} sent. Status: {response.status_code}")
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx) BEFORE processing stream

            if stream:
                # --- Manual Streaming Response Handling ---
                # Pass the requests response object to the generator
                return Response(stream_with_context(generate_anthropic_stream(response, endpoint, client_model_name, request)), mimetype='text/event-stream')
            else:
                # --- Manual Non-Streaming Anthropic ---
                # TODO: Implement non-streaming manual Anthropic call if needed
                #       - response_data = response.json()
                #       - openai_response = convert_anthropic_to_openai(response_data) # Needs implementing
                #       - log_token_usage(...) # Parse usage from response_data
                #       - return jsonify(openai_response)
                logging.error("Non-streaming manual Anthropic call not implemented.")
                return jsonify({"error": "Non-streaming not implemented for this model"}), 501

        elif underlying_provider == "azure" or underlying_provider == "gemini":
            # --- LITELLM AZURE / GEMINI (via AI Core) CALL ---
            logging.info(f"Handling '{client_model_name}' using Litellm (ProviderHint=azure)")

            # Construct final URL
            final_endpoint_url: Optional[str] = None; path="/chat/completions"; api_ver="2023-05-15" # Default Azure
            if underlying_provider=="gemini": api_ver="2023-12-01-preview" # Assumed for Gemini
            elif "o3-mini" in client_model_name.lower(): api_ver="2024-12-01-preview"
            final_endpoint_url = f"{selected_base_url.rstrip('/')}{path}?api-version={api_ver}"
            logging.info(f"Using AI Core Azure/Gemini-style URL: {final_endpoint_url}")

            # Prepare Litellm Params
            litellm_params: Dict[str, Any] = {
                "model": client_model_name, "custom_llm_provider": "azure", "api_base": final_endpoint_url,
                "api_key": current_backend_token, "headers": { "AI-Resource-Group": resource_group },
                "messages": messages, "stream": stream,
                **{k: request_data[k] for k in ["temperature","top_p",...] if k in request_data} # Optional params
            }

            # Call Litellm
            logging.info(f"Attempting Litellm call (Provider Hint: azure) for model='{client_model_name}'")
            logging.debug(f"Litellm params: { {k: (v if k!='api_key' else '***') for k,v in litellm_params.items()} }")

            if stream:
                response_generator = litellm.completion(**litellm_params)
                # Use the Litellm stream helper here
                return Response(stream_with_context(generate_litellm_stream_response(response_generator, endpoint, client_model_name, request)), mimetype='text/event-stream')
            else:
                response: ModelResponse = litellm.completion(**litellm_params)
                response_dict = response.dict(); log_token_usage(request, client_model_name, response_dict.get('usage')); return jsonify(response_dict)

        else:
            # Should not be reached
            logging.error(f"Internal Error: Unknown provider '{underlying_provider}'")
            return jsonify({"error": "Internal server configuration error."}), 500

    # --- Exception Handling for both Litellm and Manual Paths ---
    # Handle requests exceptions specifically for the manual path
    except requests.exceptions.RequestException as e:
         status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 503 # 503 Service Unavailable if no response
         error_msg = f"Connection/HTTP Error calling AI Core for Anthropic: {e}"
         logging.error(f"{endpoint}: {error_msg}", exc_info=True)
         detail = None
         if hasattr(e, 'response') and e.response is not None:
             logging.error(f"Backend Response: {e.response.text[:500]}") # Log start of response
             try: detail = e.response.json()
             except: detail = {"raw_response": e.response.text[:500]}
         return jsonify({"error": {"message": error_msg, "backend_detail": detail}}), status_code
    # Handle Litellm exceptions for the Azure/Gemini path
    except litellm.exceptions.AuthenticationError as e: logging.error(f"Litellm Auth Error: {e}",exc_info=True); return jsonify({"error":f"Backend auth failed: {e}"}), 500
    except litellm.exceptions.NotFoundError as e: logging.error(f"Litellm Not Found: {e}",exc_info=True); return jsonify({"error":f"Deployment/Model not found: {e}"}), 404
    except litellm.exceptions.RateLimitError as e: logging.error(f"Litellm Rate Limit: {e}",exc_info=True); return jsonify({"error":f"Rate limit hit: {e}"}), 429
    except litellm.exceptions.BadRequestError as e: logging.error(f"Litellm Bad Request: {e}",exc_info=True); return jsonify({"error":f"Invalid request for backend: {e}"}), 400
    except litellm.exceptions.APIConnectionError as e: logging.error(f"Litellm Connection Error: {e}",exc_info=True); return jsonify({"error":f"Connection error: {e}"}), 503
    except litellm.exceptions.Timeout as e: logging.error(f"Litellm Timeout: {e}",exc_info=True); return jsonify({"error":f"Timeout connecting: {e}"}), 504
    except litellm.exceptions.APIError as e: logging.error(f"Litellm API Error: Status={e.status_code}, Err={e}",exc_info=True); code=e.status_code if isinstance(e.status_code,int) and 400<=e.status_code<600 else 500; return jsonify({"error":f"API error: {e}"}), code
    # Catch-all for any other unexpected errors
    except Exception as e:
        logging.error(f"{endpoint}: Unhandled Exception: {e}", exc_info=True)
        logging.error(traceback.format_exc())
        return jsonify({"error": {"message": f"An unexpected internal server error occurred: {str(e)}", "type": "proxy_error"}}), 500
    finally:
        duration = time.time()-request_start_time
        logging.info(f"{endpoint}: Request for '{client_model_name}' completed. Duration: {duration:.3f}s")


# --- Helper function for MANUAL Anthropic streaming response generation (Corrected Parsing) ---
def generate_anthropic_stream(response: requests.Response, endpoint: str, client_model_name: str, flask_request: Request) -> Generator[bytes, None, None]:
    """
    Generator function to process Anthropic stream MANUALLY, parse SSE correctly,
    convert chunks to OpenAI SSE format.
    """
    buffer = "" # Use the same buffer name as original code for clarity
    chunks_yielded = 0
    usage_reported = False # TODO: Implement usage tracking if possible
    start_time = time.time()
    logging.info(f"{endpoint}: Starting MANUAL stream processing for Anthropic model {client_model_name}")

    try:
        # Iterate using iter_content, decode manually like original
        for chunk in response.iter_content(chunk_size=512): # Use a chunk size
            if not chunk:
                continue

            try:
                decoded_chunk = chunk.decode('utf-8')
                buffer += decoded_chunk
            except UnicodeDecodeError:
                logging.warning(f"Unicode decode error in chunk for {client_model_name}, skipping chunk part.")
                continue

            # Process the buffer similar to the original working code
            while True: # Keep processing buffer until no more complete messages found
                try:
                    # Find the start of the data payload
                    data_prefix = "data: "
                    start_index = buffer.find(data_prefix)
                    if start_index == -1:
                        # No "data: " found in the current buffer, need more chunks
                        break # Exit the inner while loop, continue outer for loop

                    # Find the end of the message (double newline) after the data prefix
                    end_index = buffer.find("\n\n", start_index)
                    if end_index == -1:
                        # Found "data: " but not the end yet, need more chunks
                        break # Exit the inner while loop, continue outer for loop

                    # Extract the JSON part (between "data: " and "\n\n")
                    json_part = buffer[start_index + len(data_prefix) : end_index].strip()

                    # Consume the processed message (including \n\n) from the buffer
                    buffer = buffer[end_index + 2:]

                    # Now, convert this JSON part using the converter
                    if json_part: # Ensure we extracted something
                        # The converter expects the JSON string
                        openai_sse_chunk_str = convert_anthropic_chunk_to_openai(json_part, client_model_name)
                        if openai_sse_chunk_str:
                            try:
                                yield openai_sse_chunk_str.encode('utf-8')
                                chunks_yielded += 1
                            except Exception as yield_err:
                                logging.error(f"Stream yield error: {yield_err}")
                                raise StopIteration # Stop generation if yielding fails
                    else:
                         logging.debug("Extracted empty JSON part, skipping.")

                except ValueError:
                    # This might happen if find fails unexpectedly, indicates buffer issue?
                    logging.warning(f"ValueError during buffer processing, breaking inner loop. Buffer: {buffer[:100]}...")
                    break
                except Exception as parse_err:
                    # Catch errors during index finding or slicing
                    logging.error(f"Error processing buffer segment: {parse_err}. Buffer: {buffer[:100]}...")
                    # Decide how to handle - maybe clear buffer partially? For now, break inner loop.
                    break

        # --- After the loop, process any remaining buffer content (less likely needed with this logic) ---
        # This part might catch incomplete final messages if the stream ends abruptly
        if buffer.strip():
            logging.warning(f"Processing potentially incomplete remaining buffer content: {buffer.strip()}")
            # Try one last conversion attempt? Might be risky.
            # For safety, probably best to just log it unless specific handling is needed.

        # Send the DONE signal
        yield "data: [DONE]\n\n".encode('utf-8')
        elapsed = time.time() - start_time
        logging.info(f"{endpoint}: Manual stream finished for {client_model_name}. Chunks yielded: {chunks_yielded}. Time: {elapsed:.2f}s")

    except requests.exceptions.ChunkedEncodingError as cee:
        logging.error(f"{endpoint}: ChunkedEncodingError during stream for {client_model_name}: {cee}", exc_info=True)
        error_payload = {"error": {"message": f"Stream interrupted (ChunkedEncodingError): {str(cee)}", "type": "connection_error"}}
        # (yield error chunk logic)
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"{endpoint}: Error during manual stream processing for {client_model_name} after {elapsed:.2f}s: {e}", exc_info=True)
        error_payload = {"error": {"message": f"Stream failed: {str(e)}", "type": "proxy_stream_error"}}
        # (yield error chunk logic)
    finally:
        response.close()
        # TODO: Manual usage logging for Anthropic
        if not usage_reported: logging.warning(f"Token usage logging for manual stream '{client_model_name}' needs implementation.")

# --- Helper function for Litellm streaming response generation ---
def generate_litellm_stream_response(response_generator: Generator, endpoint: str, client_model_name: str, flask_request: Request) -> Generator[bytes, None, None]:
    """Generator function to yield Litellm SSE formatted chunks and log usage."""
    final_usage_data = None; start_time = time.time(); chunks_yielded = 0
    try:
        logging.info(f"{endpoint}: Litellm stream starting for {client_model_name}")
        for chunk in response_generator:
            try: chunk_dict = chunk.dict()
            except: logging.error("Stream chunk format error"); continue
            if chunk_dict.get('usage'): final_usage_data = chunk_dict['usage']
            try: yield f"data: {json.dumps(chunk_dict)}\n\n".encode('utf-8'); chunks_yielded+=1
            except: logging.error("Stream yield error"); break
        yield "data: [DONE]\n\n".encode('utf-8')
        logging.info(f"{endpoint}: Litellm stream finished for {client_model_name}. Chunks: {chunks_yielded}. Time: {time.time()-start_time:.2f}s")
    except Exception as e:
        logging.error(f"{endpoint}: Litellm stream error for {client_model_name}: {e}", exc_info=True)
        try: error_payload = {"error": {"message": f"Stream failed: {e}"}}; yield f"data: {json.dumps(error_payload)}\n\n".encode('utf-8'); yield "data: [DONE]\n\n".encode('utf-8')
        except: logging.error(f"Failed writing stream error")
    finally: log_token_usage(flask_request, client_model_name, final_usage_data)


# --- Main Execution Block ---
if __name__ == '__main__':
    args = parse_arguments()
    logging.info(f"--- Starting SAP AI Core Universal LLM Proxy v6 (Hybrid Approach) ---")
    logging.info(f"Loading configuration from: {args.config}")
    try: config = load_config(args.config)
    except: logging.critical("Exiting: Config loading failure."); exit(1)

    # Load config values into globals
    service_key_json_path = config.get('service_key_json')
    aicore_deployments_config = config.get('deployment_models', {})
    secret_authentication_tokens = config.get('secret_authentication_tokens', [])
    resource_group = config.get('resource_group', '')
    host = config.get('host', '0.0.0.0'); port = config.get('port', 3001)

    # --- Configuration Validation ---
    crit_err = False
    if not aicore_deployments_config: logging.critical("Config Error: 'deployment_models' empty."); crit_err=True
    if not service_key_json_path: logging.critical("Config Error: 'service_key_json' missing."); crit_err=True
    if not resource_group: logging.critical("Config Error: 'resource_group' missing."); crit_err=True
    if not isinstance(secret_authentication_tokens,list): logging.critical("Config Error: 'secret_authentication_tokens' not list."); crit_err=True
    if not secret_authentication_tokens: logging.warning("PROXY SECURITY: Client auth disabled.")

    # Validate URLs
    aicore_deployment_urls = {}
    for model, urls in aicore_deployments_config.items():
        if not isinstance(urls, list) or not urls: logging.critical(f"Config Error: URLs for '{model}' invalid."); crit_err=True; continue
        for i, url in enumerate(urls):
             if not isinstance(url,str) or not url.startswith(('http://','https://')): logging.critical(f"Config Error: Invalid URL '{url}' for model '{model}'."); crit_err=True
        if not crit_err: aicore_deployment_urls[model] = urls
    if crit_err: logging.critical("Exiting due to critical config errors."); exit(1)

    # Load service key JSON
    try:
        service_key = load_config(service_key_json_path)
        if not isinstance(service_key,dict) or not all(k in service_key for k in ['clientid','clientsecret','url']): logging.critical("Service key missing required keys."); exit(1)
    except Exception: logging.critical(f"Exiting: Failed loading/validating service key."); exit(1)

    logging.info(f"Proxy configured for SAP AI Core models: {list(aicore_deployment_urls.keys())}")

    # --- Start Server ---
    logging.info(f"Starting SAP AI Core Proxy Server on http://{host}:{port}")
    try: from waitress import serve; logging.info("Using Waitress."); serve(app, host=host, port=port, threads=10)
    except ImportError: logging.warning("Waitress not found. Using Flask dev server."); app.run(host=host, port=port, debug=False)
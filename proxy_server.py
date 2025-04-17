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
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Proxy server for AI models")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    return parser.parse_args()

def fetch_token():
    global token, token_expiry
    with lock:
        if time.time() < token_expiry:
            logging.info("Using cached token.")
            return token

        logging.info("Fetching new token.")
        # Encode client_id and client_secret
        secret = base64.b64encode(f"{service_key['clientid']}:{service_key['clientsecret']}".encode()).decode()
        token_url = f"{service_key['url']}/oauth/token?grant_type=client_credentials"
        headers = {"Authorization": f"Basic {secret}"}

        response = requests.post(token_url, headers=headers)
        try:
            response.raise_for_status()
            token = response.json().get('access_token')
            token_expiry = time.time() + 4 * 3600  # Token valid for 4 hours
            logging.info("Token fetched successfully.")
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP error occurred while fetching token: {err}")
            raise
        except Exception as err:
            logging.error(f"An error occurred while fetching token: {err}")
            raise
        return token

def verify_request_token(request):
    token = request.headers.get("Authorization")
    logging.info(f"verify_request_token, Token received in request: {token}")
    if not token or not any(secret_key in token for secret_key in secret_authentication_tokens):
        logging.error("Invalid or missing token.")
        return False
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
    Converts a Claude 3.7 /converse API response payload to the format
    expected by the OpenAI Chat Completion API.
    """
    try:
        logging.debug(f"Raw response from Claude 3.7 API: {json.dumps(response, indent=2)}")

        # Validate the structure of the Claude response
        if not isinstance(response, dict):
            raise ValueError("Invalid response format: response is not a dictionary")

        output = response.get("output")
        if not isinstance(output, dict):
            raise ValueError("Invalid response structure: 'output' is missing or not a dictionary")

        message = output.get("message")
        if not isinstance(message, dict):
            raise ValueError("Invalid response structure: 'output.message' is missing or not a dictionary")

        content_list = message.get("content")
        if not isinstance(content_list, list) or not content_list:
            raise ValueError("Invalid response structure: 'output.message.content' is missing, not a list, or empty")

        first_content_block = content_list[0]
        if not isinstance(first_content_block, dict) or "text" not in first_content_block:
            raise ValueError("Invalid response structure: 'output.message.content[0].text' is missing or invalid")

        usage = response.get("usage")
        if not isinstance(usage, dict):
            logging.warning("Usage information missing in Claude response.")
            usage = {} # Use empty dict if usage is missing

        # Map Claude stopReason to OpenAI finish_reason
        stop_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            # Add other mappings if needed
        }
        finish_reason = stop_reason_map.get(response.get("stopReason"), "stop") # Default to 'stop'

        # Conversion logic from Claude 3.7 /converse API to OpenAI format
        openai_response = {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "content": first_content_block["text"],
                        "role": message.get("role", "assistant") # Default role to assistant
                    }
                }
            ],
            "created": int(time.time()),
            "id": f"chatcmpl-{random.randint(10000, 99999)}", # Generate a simple random ID
            "model": model_name, # Use the provided model name
            "object": "chat.completion",
            "usage": {
                "completion_tokens": usage.get("outputTokens", 0),
                "prompt_tokens": usage.get("inputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0)
            }
            # Optionally include system_fingerprint if needed/available
            # "system_fingerprint": response.get("system_fingerprint")
        }
        logging.debug(f"Converted response to OpenAI format: {json.dumps(openai_response, indent=2)}")
        return openai_response
    except Exception as e:
        logging.error(f"Error converting Claude 3.7 response to OpenAI format: {e}")
        logging.error(f"Problematic Claude response: {json.dumps(response, indent=2)}")
        # Return an error structure in OpenAI format
        return {
            "choices": [],
            "created": int(time.time()),
            "id": f"chatcmpl-error-{random.randint(10000, 99999)}",
            "model": model_name,
            "object": "chat.completion",
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
            "error": {
                "message": f"Failed to convert Claude 3.7 response: {str(e)}",
                "type": "conversion_error",
                "param": None,
                "code": None
            }
        }



def convert_claude_chunk_to_openai(chunk):
    try:
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

def is_claude_model(model):
    return "claude" in model or "sonnet" in model

def load_balance_url(urls, model_key):
    # Implement a simple round-robin load balancing mechanism
    logging.debug(f"load_balance_url called with model_key: {model_key} and urls: {urls}")
    if not hasattr(load_balance_url, "counters"):
        logging.debug("Initializing 'counters' attribute for load balancing.")
        load_balance_url.counters = {}
    
    if model_key not in load_balance_url.counters:
        logging.debug(f"Initializing counter for model key '{model_key}'.")
        load_balance_url.counters[model_key] = 0
    
    # Ensure the list of URLs is not empty to avoid division by zero
    if not urls:
        logging.error(f"No URLs available for model key '{model_key}'.")
        raise ValueError(f"No URLs available for model key '{model_key}'.")

    index = load_balance_url.counters[model_key] % len(urls)
    logging.debug(f"Model key '{model_key}' selected index {index}. Counter value: {load_balance_url.counters[model_key]}")
    load_balance_url.counters[model_key] += 1
    logging.info(f"load_balance_url for {model_key}: Selected URL: {urls[index]}")
    return urls[index]

def handle_claude_request(payload, model="3.5-sonnet"):
    stream = payload.get("stream", True)  # Default to True if 'stream' is not provided
    # stream = False
    logging.info(f"handle_claude_request: model={model} stream={stream}")
    for key in normalized_model_deployment_urls:
        # logging.info(f"handle_claude_request: key={key}")
        # if 'claud' in key or 'sonnet' in key:
        if model == key:
            urls = normalized_model_deployment_urls[key]
            if stream:
                url = f"{load_balance_url(urls, key)}/invoke-with-response-stream"
            else:
                # Check if the model name contains '3.7'
                if "3.7" in model:
                    url = f"{load_balance_url(urls, key)}/converse"
                else:
                    url = f"{load_balance_url(urls, key)}/invoke"
            break
    else:
        raise ValueError("No valid Claude or Sonnet model found in deployment URLs.")
    if "3.7" in model:
        payload = convert_openai_to_claude37(payload)
    else:
        payload = convert_openai_to_claude(payload)
    logging.info(f"handle_claude_request: {url}")
    return url, payload

def handle_default_request(payload, model="gpt-4o"):
    urls = normalized_model_deployment_urls.get(model, normalized_model_deployment_urls['gpt-4o'])
    if "o3-mini" in model:
        url = f"{load_balance_url(urls, model)}/chat/completions?api-version=2024-12-01-preview"
    else:
        url = f"{load_balance_url(urls, model)}/chat/completions?api-version=2023-05-15"
    return url, payload

@app.route('/v1/chat/completions', methods=['OPTIONS'])
def proxy_openai_stream2():
    logging.info("OPTIONS:Received request to /v1/chat/completions")
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request body:\n {json.dumps(request.get_json(), indent=4)}")
    return jsonify({
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hi.",
                    "role": "assistant"
                }
            }
        ],
        "created": 1721357889,
        "id": "chatcmpl-9mY6rfxwzY7Q9IyzWairiHEYZfD8a",
        "model": "gpt-4o-2024-05-13",
        "object": "chat.completion",
        "system_fingerprint": "fp_abc28019ad",
        "usage": {
            "completion_tokens": 2,
            "prompt_tokens": 26,
            "total_tokens": 28
        }
    }), 200

@app.route('/v1/models', methods=['GET'])
def list_models():
    logging.info("Received request to /v1/models")
    models = [
        {
            "id": model,
            "object": "model",
            "created": 1686935002,  # Example timestamp, replace with actual if available
            "owned_by": "organization-owner"  # Replace with actual owner if available
        }
        for model in model_deployment_urls.keys()
    ]
    
    return jsonify({"object": "list", "data": models}), 200

content_type="Application/json"
@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def proxy_openai_stream():
    logging.info("Received request to /v1/chat/completions")
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request body:\n{json.dumps(request.get_json(), indent=4)}")
    if not verify_request_token(request):
        logging.info("Unauthorized request received. Token verification failed.")
        return jsonify({"error": "Unauthorized"}), 401

    global token
    token = fetch_token()

    # Extract model from the request payload
    payload = request.json
    model = payload.get("model")
    # Check if model contains '3.7' and force non-streaming if it does
    if model and "3.7" in model:
        logging.info(f"Model '{model}' contains '3.7', forcing non-streaming mode (isStream set to False).")
        isStream = False
        payload["stream"] = False # Ensure payload reflects the non-streaming requirement

    isStream = payload.get("stream", True)
    logging.info(f"Extracted model from request payload: {model}")
    logging.info(f"Streaming mode: {isStream}")

    if not model or model not in normalized_model_deployment_urls:
        logging.info("Model not found in deployment URLs, falling back to 3.5-sonnet")
        model = "gpt-4o"

    if is_claude_model(model):
        url, payload = handle_claude_request(payload, model)
    else:
        url, payload = handle_default_request(payload, model)

    headers = {
        "AI-Resource-Group": resource_group,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    logging.info(f"Forwarding request to {url} with payload:\n {json.dumps(payload, indent=4)}")

    # Check if streaming is disabled
    if not isStream:  # Use the isStream variable directly
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logging.info("Request to actual API succeeded (non-streaming).")
            
            # Log the final response before returning it
            final_response = jsonify(response.json())
            if is_claude_model(model):
                final_response = jsonify(convert_claude_to_openai(response.json(), model))
            logging.info(f"Final response sent to client: {json.dumps(response.json(), indent=4)}")
            user_id = request.headers.get("Authorization", "unknown")
            max_user_id_length = 30
            if len(user_id) < max_user_id_length:
                user_id = user_id.ljust(max_user_id_length, '_')
            else:
                user_id = user_id[:max_user_id_length]
            ip_address = request.remote_addr
            total_tokens = response.json().get("usage", {}).get("total_tokens", 0)
            token_logger.info(f"User: {user_id}, IP: {ip_address}, Model: {model}, Tokens: {total_tokens}")
            return final_response, response.status_code
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP error occurred while forwarding request: {err}")
            if err.response is not None:
                logging.error(f"Error response content: {err.response.text}")
                return jsonify({"error": err.response.text}), err.response.status_code
        except Exception as err:
            logging.error(f"An error occurred while forwarding request: {err}")
            return jsonify({"error": "Internal server error"}), 500

    # Streaming logic
    def generate():
        buffer = ""
        total_tokens = 0
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            try:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=128):  # Reduced chunk size
                    if chunk:
                        if is_claude_model(model):
                            buffer += chunk.decode('utf-8')
                            while "data: " in buffer:
                                try:
                                    start = buffer.index("data: ") + len("data: ")
                                    end = buffer.index("\n\n", start)
                                    json_chunk = buffer[start:end].strip()
                                    buffer = buffer[end + 2:]
                                    json_chunk = convert_claude_chunk_to_openai(json_chunk)
                                    yield json_chunk.encode('utf-8')
                                except ValueError:
                                    break
                        else:
                            yield chunk
                            try:
                                chunk_text = chunk.decode('utf-8')
                                match = re.search(r'"total_tokens":(\d+)', chunk_text)
                                if match:
                                    total_tokens += int(match.group(1))
                            except Exception as e:
                                logging.error(f"An unexpected error occurred while decoding JSON: {e}")
                        time.sleep(0.01)
                user_id = request.headers.get("Authorization", "unknown")
                max_user_id_length = 30
                if len(user_id) < max_user_id_length:
                    user_id = user_id.ljust(max_user_id_length, '_')
                else:
                    user_id = user_id[:max_user_id_length]
                ip_address = request.remote_addr
                token_logger.info(f"User: {user_id}, IP: {ip_address}, Model: {model}, Tokens: {total_tokens}")
                logging.info("Request to actual API succeeded.")
            except requests.exceptions.HTTPError as err:
                logging.error(f"HTTP error occurred while forwarding request: {err}")
                if err.response is not None:
                    logging.error(f"Error response content: {err.response.text}")
                raise
            except Exception as err:
                logging.error(f"An error occurred while forwarding request: {err}")
                raise

    return Response(stream_with_context(generate()), content_type=content_type)


if __name__ == '__main__':
    args = parse_arguments()
    logging.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Update these variables with the new config
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

    logging.info(f"Starting proxy server on host {host} and port {port}...")
    logging.info(f"API Host: http://{host}:{port}/v1")
    app.run(host=host, port=port, debug=True)
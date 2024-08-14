import logging  
from flask import Flask, request, jsonify, Response, stream_with_context  
import requests
import time
import threading
import json
import base64

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for token management
token = None
token_expiry = 0
lock = threading.Lock()

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Load configuration
config = load_config('config.json')
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
        system_message = messages.pop(0)["content"][0]["text"]

    # Conversion logic from OpenAI to Claude API format
    claude_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": payload.get("max_tokens", 4096),
        "system": system_message,
        "messages": messages
    }
    return claude_payload

def convert_claude_to_openai(response):
    # Conversion logic from Claude API to OpenAI format
    openai_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": response["choices"][0]["message"]["content"],
                    "role": "assistant"
                }
            }
        ],
        "created": int(time.time()),
        "id": response.get("id", "chatcmpl-unknown"),
        "model": "claude-v1",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
            "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
            "total_tokens": response.get("usage", {}).get("total_tokens", 0)
        }
    }
    return openai_response

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

def handle_claude_request(payload):
    for key in normalized_model_deployment_urls:
        if 'claud' in key or 'sonnet' in key:
            url = f"{normalized_model_deployment_urls[key]}/invoke-with-response-stream"
            break
    else:
        raise ValueError("No valid Claude or Sonnet model found in deployment URLs.")
    payload = convert_openai_to_claude(payload)
    return url, payload

def handle_default_request(payload):
    url = f"{normalized_model_deployment_urls['gpt-4o']}/chat/completions?api-version=2023-05-15"
    return url, payload

@app.route('/v1/chat/completions', methods=['OPTIONS'])
def proxy_openai_stream2():
    logging.info("OPTIONS:Received request to /v1/chat/completions")
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request body: {request.get_json()}")
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
@app.route('/v1/chat/completions', methods=['POST'])
def proxy_openai_stream():
    logging.info("Received request to /v1/chat/completions")
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request body: {request.get_json()}")
    if not verify_request_token(request):
        logging.info("Unauthorized request received. Token verification failed.")
        return jsonify({"error": "Unauthorized"}), 401

    global token
    token = fetch_token()

    # Extract model from the request payload
    payload = request.json
    model = payload.get("model")
    logging.info(f"Extracted model from request payload: {model}")

    if not model or model not in normalized_model_deployment_urls:
        logging.info("Model not found in deployment URLs, falling back to gpt-4o")
        model = "gpt-4o"

    if is_claude_model(model):
        url, payload = handle_claude_request(payload)
    else:
        url, payload = handle_default_request(payload)

    headers = {
        "AI-Resource-Group": resource_group,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    logging.info(f"Forwarding request to {url} with payload: {payload}")

    def generate():
        buffer = ""
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
                        time.sleep(0.01)  # Small sleep to avoid overwhelming the client
                logging.info("Request to actual API succeeded.")
            except requests.exceptions.HTTPError as err:
                logging.error(f"HTTP error occurred while forwarding request: {err}")
                raise
            except Exception as err:
                logging.error(f"An error occurred while forwarding request: {err}")
                raise

    return Response(stream_with_context(generate()), content_type=content_type)

if __name__ == '__main__':
    logging.info("Starting proxy server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
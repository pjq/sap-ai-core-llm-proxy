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
    if not model or model not in model_deployment_urls:
        logging.info("Model not found in deployment URLs, falling back to gpt-4o")
        model = "gpt-4o"
    logging.info(f"Extracted model from request payload: {model}")
    if not model or model not in model_deployment_urls:
        return jsonify({"error": "Invalid or missing model"}), 400

    url = f"{model_deployment_urls[model]}/chat/completions?api-version=2023-05-15"

    headers = {
        "AI-Resource-Group": resource_group,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    logging.info(f"Forwarding request to {url} with payload: {payload}")

    # Stream the response from the OpenAI API
    def generate():
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            try:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type')
                for chunk in response.iter_content(chunk_size=128):  # Reduced chunk size
                    if chunk:
                        print(chunk)
                        yield chunk
                        # Explicitly flush the response to ensure timely delivery
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
    #app.run(host='127.0.0.1', port=8443, debug=True, ssl_context=('cert.pem', 'key.pem'))
    #app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='127.0.0.1', port=5000, debug=True)

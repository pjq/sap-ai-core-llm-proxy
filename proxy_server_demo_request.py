import logging
import requests
import json

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Load configuration
config = load_config('config.json')

def demo_request():
    url = "http://127.0.0.1:5000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['secret_authentication_tokens'][0]}"  # Updated
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, who are you?"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None
    }

    logging.info(f"Sending demo request to {url} with payload: {payload}")
    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
        logging.info("Demo request succeeded.")
        print(response.json())
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred during demo request: {err}")
    except Exception as err:
        logging.error(f"An error occurred during demo request: {err}")


def demo_request_stream():
    url = "http://127.0.0.1:5000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['secret_authentication_tokens'][0]}"  # Updated
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, who are you? what can you do"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False,
        "stop": None,
        "model": "claude-3.5-sonnet"
    }

    logging.info(f"Sending demo request to {url} with payload: {payload}")
    response = requests.post(url, headers=headers, json=payload, stream=True, verify=False)
    try:
        response.raise_for_status()
        logging.info("Demo request succeeded.")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # json_line = json.loads(decoded_line)
                print(decoded_line)
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred during demo request: {err}")
    except Exception as err:
        logging.error(f"An error occurred during demo request: {err}")

def test_list_models():
    url = "http://127.0.0.1:5000/v1/models"
    headers = {
        "Authorization": f"Bearer {config['secret_authentication_tokens'][0]}"  # Updated
    }

    logging.info(f"Sending request to {url}")
    response = requests.get(url, headers=headers)
    try:
        response.raise_for_status()
        logging.info("Request to /v1/models succeeded.")
        print(response.json())
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred during request to /v1/models: {err}")
    except Exception as err:
        logging.error(f"An error occurred during request to /v1/models: {err}")


# demo_request()
demo_request_stream()
test_list_models()
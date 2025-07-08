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
    url = "http://127.0.0.1:3001/v1/chat/completions"
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
        "model": "gpt-4o",
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
    url = "http://127.0.0.1:3001/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['secret_authentication_tokens'][0]}"  # Updated
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hi"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": True,
        "stop": None,
        # "model": "claude-3.5-sonnet"
        "model": "gpt-4o"
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


def demo_request_gemini_stream():
    """Test Gemini model with streaming request"""
    url = "http://127.0.0.1:3001/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['secret_authentication_tokens'][0]}"
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hi, who are you?"
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.8,
        "stream": True,
        "model": "gemini-2.5-pro"
    }

    logging.info(f"Sending Gemini streaming demo request to {url} with payload: {payload}")
    response = requests.post(url, headers=headers, json=payload, stream=True, verify=False)
    try:
        response.raise_for_status()
        logging.info("Gemini streaming demo request succeeded.")
        print("=== Gemini Streaming Response ===")
        print("🤖 Gemini Assistant Response (streaming):")
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(decoded_line)  # Print raw SSE data
                
                # Also extract and display content for better readability
                if decoded_line.startswith('') and not decoded_line.endswith('[DONE]'):
                    try:
                        data_content = decoded_line[6:]  # Remove '' prefix
                        chunk_data = json.loads(data_content)
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                full_response += content
                    except json.JSONDecodeError:
                        continue
        
        print(f"\n📝 Full Gemini Response: {full_response}")
        
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred during Gemini streaming demo request: {err}")
        print(f"Error response: {response.text}")
    except Exception as err:
        logging.error(f"An error occurred during Gemini streaming demo request: {err}")


def test_list_models():
    url = "http://127.0.0.1:3001/v1/models"
    headers = {
        "Authorization": f"Bearer {config['secret_authentication_tokens'][0]}"  # Updated
    }

    logging.info(f"Sending request to {url}")
    response = requests.get(url, headers=headers)
    try:
        response.raise_for_status()
        logging.info("Request to /v1/models succeeded.")
        models_response = response.json()
        print("=== Available Models ===")
        print(json.dumps(models_response, indent=2))
        
        # Check if Gemini models are available
        if "data" in models_response:
            gemini_models = [model for model in models_response["data"] if "gemini" in model.get("id", "").lower()]
            if gemini_models:
                print(f"\n🔍 Found {len(gemini_models)} Gemini model(s):")
                for model in gemini_models:
                    print(f"  - {model['id']}")
            else:
                print("\n⚠️  No Gemini models found in the available models list.")
                
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred during request to /v1/models: {err}")
    except Exception as err:
        logging.error(f"An error occurred during request to /v1/models: {err}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run all tests
    # run_all_tests()

# Individual test functions (uncomment to run specific tests)
# demo_request()
# demo_request_stream()
# demo_request_gemini_stream()
# test_list_models()

import logging
import requests
import json
import threading
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(file_path):
    logger.info(f"Loading configuration from {file_path}")
    with open(file_path, 'r') as file:
        config = json.load(file)
    logger.debug(f"Configuration loaded successfully")
    return config

# Load configuration
config = load_config('config.json')

# ... existing code ...

def load_test(num_threads=10, total_requests=100, endpoint="chat/completions", timeout=30, model="gpt-4o"):
    """
    Run a load test with specified number of concurrent threads and total requests.
    
    Args:
        num_threads: Number of concurrent threads to use
        total_requests: Total number of requests to make
        endpoint: API endpoint to test ("chat/completions" or "models")
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary with test results
    """
    logger.info(f"Initializing load test: {num_threads} threads, {total_requests} requests, endpoint: {endpoint}")
    url = f"http://127.0.0.1:3001/v1/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['secret_authentication_tokens'][0]}"
    }
    
    # Default payload for chat completions
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "ONLY REPLY: Y"
            }
        ],
        "max_tokens": 50,
        "temperature": 0.0,
        "model": model,
        "stream": False
    }
    
    # Tracking metrics
    response_times = []
    status_codes = Counter()
    errors = []
    lock = threading.Lock()
    completed_requests = 0
    
    def make_request():
        nonlocal completed_requests
        try:
            request_id = threading.get_ident()
            logger.debug(f"Thread {request_id}: Starting request")
            start_time = time.time()
            
            if endpoint == "chat/completions":
                logger.debug(f"Thread {request_id}: Sending POST to {url}")
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            else:  # models endpoint
                logger.debug(f"Thread {request_id}: Sending GET to {url}")
                response = requests.get(url, headers=headers, timeout=timeout)
            
            elapsed = time.time() - start_time
            
            with lock:
                response_times.append(elapsed)
                # Log the response content
                try:
                    resp_content = response.json()
                    logger.debug(f"Thread {request_id}: Response content: {resp_content}")
                    status_codes[response.status_code] += 1
                    # Check if the content matches the expected "Y" response
                    if "choices" in resp_content and resp_content["choices"]:
                        assistant_message = resp_content["choices"][0].get("message", {})
                        if "content" in assistant_message and assistant_message["content"].strip() == "Y":
                            logger.info(f"Thread {request_id}: Received expected 'Y' response")
                        else:
                            logger.info(f"Thread {request_id}: Unexpected response content: {assistant_message.get('content')}")
                except Exception as e:
                    logger.info(f"Thread {request_id}: Could not parse response as JSON: {e}")
                    logger.info(f"Thread {request_id}: {response.status_code} Raw response content: {response.text}")
                    # Log error code if status code is not 200
                    if response.status_code != 200:
                        logger.error(f"Thread {request_id}: Error code: {response.status_code}, message: {response.text}")
                completed_requests += 1
                if completed_requests % max(1, total_requests // 10) == 0:  # Log progress at 10% intervals
                    logger.info(f"Progress: {completed_requests}/{total_requests} requests completed ({completed_requests/total_requests*100:.1f}%)")
                
            logger.debug(f"Thread {request_id}: Response received, status code {response.status_code}, time {elapsed:.4f}s")
            response.raise_for_status()
            return True
            
        except requests.exceptions.HTTPError as err:
            with lock:
                errors.append(f"HTTP error: {err}")
            logger.error(f"Thread {request_id}: HTTP error: {err}")
            return False
            
        except Exception as err:
            with lock:
                errors.append(f"Error: {err}")
            logger.error(f"Thread {request_id}: Error: {err}")
            return False
    
    logger.info(f"Starting load test with {num_threads} concurrent threads, {total_requests} total requests")
    logger.info(f"Testing endpoint: {url}")
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor to manage threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        logger.debug(f"Thread pool created with {num_threads} workers")
        futures = [executor.submit(make_request) for _ in range(total_requests)]
        logger.debug(f"Submitted {total_requests} tasks to thread pool")
        
        # Wait for all futures to complete
        for future in futures:
            future.result()
    
    total_time = time.time() - start_time
    logger.info(f"Load test completed in {total_time:.2f} seconds")
    
    # Calculate metrics
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        median_response_time = statistics.median(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        requests_per_second = total_requests / total_time
        success_rate = (status_codes.get(200, 0) / total_requests) * 100
    else:
        avg_response_time = median_response_time = min_response_time = max_response_time = 0
        requests_per_second = 0
        success_rate = 0
    
    # Print results
    logger.info("\n===== LOAD TEST RESULTS =====")
    logger.info(f"Model: {model}")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Concurrent threads: {num_threads}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Requests per second: {requests_per_second:.2f}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Status codes: {dict(status_codes)}")
    logger.info(f"Average response time: {avg_response_time:.4f} seconds")
    logger.info(f"Median response time: {median_response_time:.4f} seconds")
    logger.info(f"Min response time: {min_response_time:.4f} seconds")
    logger.info(f"Max response time: {max_response_time:.4f} seconds")
    
    if errors:
        logger.warning(f"\nErrors ({len(errors)}):")
        for i, error in enumerate(errors[:10]):  # Show first 10 errors
            logger.warning(f"  {i+1}. {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")
    
    return {
        "total_requests": total_requests,
        "concurrent_threads": num_threads,
        "total_time": total_time,
        "requests_per_second": requests_per_second,
        "success_rate": success_rate,
        "status_codes": dict(status_codes),
        "avg_response_time": avg_response_time,
        "median_response_time": median_response_time,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "errors": errors
    }

# load_test(num_threads=10, total_requests=100, endpoint="chat/completions", model="gpt-4o")
load_test(num_threads=10, total_requests=100, endpoint="chat/completions", model="3.7-sonnet")
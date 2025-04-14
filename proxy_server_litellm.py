import logging
from flask import Flask, request, jsonify, Response, stream_with_context, Request
import requests
import time
import threading
import json
import base64
import random
import os
from datetime import datetime
import argparse
from typing import Optional, Dict, Any, Generator, List, Tuple

# Import Litellm
import litellm
from litellm import ModelResponse
import traceback

# --- Flask App and Logging Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING); logging.getLogger("requests").setLevel(logging.WARNING); logging.getLogger("httpx").setLevel(logging.WARNING)
token_logger = logging.getLogger('token_usage'); token_logger.setLevel(logging.INFO); token_logger.propagate = False
log_directory = 'logs'; os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, 'token_usage.log')
try:
    fh=logging.FileHandler(log_file,mode='a+',encoding='utf-8'); fh.setLevel(logging.INFO); fmt=logging.Formatter('%(asctime)s - %(message)s'); fh.setFormatter(fmt); token_logger.addHandler(fh)
    token_logger.info("--- Token Logger Initialized ---"); logging.info(f"Token usage logging configured to: {log_file}")
except Exception as e: logging.error(f"CRITICAL: Could not configure token log file {log_file}: {e}", exc_info=True)

# --- Global variables ---
backend_auth_token: Optional[str] = None; token_expiry_time: float = 0.0; token_fetch_lock = threading.Lock()
config: Dict[str, Any] = {}; service_key: Optional[Dict[str, str]] = None
aicore_deployment_urls: Dict[str, list[str]] = {}; secret_authentication_tokens: list[str] = []; resource_group: str = ""

# --- Config Loading and Parsing ---
def load_config(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path,'r', encoding='utf-8') as f: data=json.load(f); logging.info(f"Loaded config: {file_path}"); return data
    except Exception as e: logging.critical(f"Failed loading config '{file_path}': {e}", exc_info=True); raise
def parse_arguments() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="SAP AI Core Hybrid Proxy v5"); parser.add_argument("--config",default="config.json"); return parser.parse_args()

# --- Backend Token Fetching ---
def fetch_token() -> str:
    global backend_auth_token, token_expiry_time;
    if not service_key: raise ValueError("OAuth config missing");
    with token_fetch_lock:
        now=time.time();
        if backend_auth_token and now<token_expiry_time: logging.debug("Using cached token."); return backend_auth_token
        logging.info("Fetching new backend token for SAP AI Core.")
        try: # ...(same fetch logic)...
            cid=service_key['clientid'];cs=service_key['clientsecret'];burl=service_key['url']; auth=f"{cid}:{cs}";enc=base64.b64encode(auth.encode('utf-8')).decode('ascii'); turl=f"{burl.rstrip('/')}/oauth/token?grant_type=client_credentials";h={"Authorization":f"Basic {enc}"}; res=requests.post(turl,headers=h,timeout=15); res.raise_for_status(); td=res.json(); nt=td.get('access_token'); exp_in=int(td.get('expires_in',14400));
            if not nt: raise ValueError("Fetched token empty")
            backend_auth_token=nt; token_expiry_time=now+exp_in-300; logging.info("Backend token fetched successfully.")
            return backend_auth_token
        except Exception as err: logging.error(f"Backend token fetch failed: {err}",exc_info=True); backend_auth_token=None; token_expiry_time=0; raise

# --- Client Request Token Verification ---
def verify_request_token(request: Request) -> bool:
     if not secret_authentication_tokens: logging.warning("Client Auth disabled."); return True
     ct = request.headers.get("Authorization"); log_prefix = ct[:15]+"..." if ct and len(ct)>15 else ct; logging.debug(f"Verifying client token prefix: '{log_prefix}'")
     if not ct or not any(sk in ct for sk in secret_authentication_tokens): logging.warning("Invalid/Missing client token."); return False
     logging.debug("Client token verified."); return True

# --- Load Balancing ---
def load_balance_url(urls: list[str], model_key: str) -> Optional[str]:
    if not urls: logging.error(f"No BASE URLs for model '{model_key}'."); return None
    if not hasattr(load_balance_url,"counters"): load_balance_url.counters={}
    if model_key not in load_balance_url.counters: load_balance_url.counters[model_key]=0
    with threading.Lock(): idx=load_balance_url.counters[model_key]%len(urls); url=urls[idx]; load_balance_url.counters[model_key]+=1
    logging.info(f"Selected AI Core base URL for '{model_key}': {url}")
    return url

# --- Helper to determine UNDERLYING provider type ---
def get_underlying_provider(model_name: str) -> str:
    model_name_lower = model_name.lower() if model_name else ""
    if "claude" in model_name_lower or "sonnet" in model_name_lower: return "anthropic"
    if "gemini" in model_name_lower: return "gemini"
    return "azure"

# --- Token Usage Logging Helper (MODIFIED TO ALWAYS LOG) ---
def log_token_usage(request: Request, model_name: str, usage_object: Optional[Dict[str, int]]):
    """Logs token usage details. Logs defaults (0) if data is missing/invalid."""
    log_prefix = f"[Token Log for {model_name}]"
    logging.debug(f"{log_prefix} Attempting log. Received usage object: {usage_object}")
    estimated = False # Flag to indicate if we used default values

    try:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # Try to extract real values, default to 0 if missing or invalid
        if usage_object and isinstance(usage_object, dict):
            try: prompt_tokens = int(usage_object.get('prompt_tokens', 0) or 0)
            except (ValueError, TypeError): logging.warning(f"{log_prefix} Invalid prompt_tokens value: {usage_object.get('prompt_tokens')}")
            try: completion_tokens = int(usage_object.get('completion_tokens', 0) or 0)
            except (ValueError, TypeError): logging.warning(f"{log_prefix} Invalid completion_tokens value: {usage_object.get('completion_tokens')}")

            # Calculate total only if individual counts are valid, otherwise default total also
            if prompt_tokens > 0 or completion_tokens > 0:
                 total_tokens = usage_object.get('total_tokens', prompt_tokens + completion_tokens)
                 try: total_tokens = int(total_tokens or 0)
                 except (ValueError, TypeError): total_tokens = prompt_tokens + completion_tokens
            else:
                 total_tokens = 0 # Keep total as 0 if components were invalid/zero
        else:
            # Input usage_object was None or not a dict
            logging.warning(f"{log_prefix} Valid usage data object not provided. Logging defaults.")
            estimated = True

        # If all counts ended up as 0, mark as estimated (unless it was genuinely 0)
        if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0 and usage_object:
             # Check if original object *actually* had all zeros
             original_pt = usage_object.get('prompt_tokens')
             original_ct = usage_object.get('completion_tokens')
             if not (original_pt == 0 and original_ct == 0) :
                  estimated = True # Mark as estimated if we defaulted non-zero values to zero

        # Get user identifier (sanitize client token)
        client_token = request.headers.get("Authorization", "unknown"); max_len=30; user_id="***" # Placeholder/Sanitize
        if client_token and client_token != "unknown":
             if client_token.lower().startswith("bearer "): 
                 token_part = client_token[7:]
                 user_id = f"Bearer {token_part[:max_len]}..." if len(token_part) > max_len else f"Bearer {token_part}"
             else: user_id=f"{client_token[:max_len]}..." if len(client_token)>max_len else client_token

        ip_address = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown_ip")

        # Construct log message, add note if estimated
        log_message = (f"User: {user_id}, IP: {ip_address}, Model: {model_name}, "
                       f"PromptTokens: {prompt_tokens}, CompletionTokens: {completion_tokens}, TotalTokens: {total_tokens}"
                       f"{' (Estimated/Default)' if estimated else ''}") # Add note if defaulted

        token_logger.info(log_message)
        logging.info(f"{log_prefix} Logged usage to token_usage.log {'(Estimated)' if estimated else ''}.")

    except Exception as e:
        logging.error(f"{log_prefix} Failed during token usage logging processing: {e}", exc_info=True)


# --- Manual Anthropic Chunk Converter (Returns usage) ---
def convert_anthropic_chunk_to_openai(anthropic_json_str: str, model_name: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
    # ...(Same as previous version - parses usage from chunk)...
    logging.debug(f"Converting Anthropic JSON chunk: {anthropic_json_str}")
    usage_info: Dict[str,Any]={}; openai_sse_str: Optional[str]=None
    try: data=json.loads(anthropic_json_str); event_type=data.get("type"); assert event_type
    except: logging.error(f"Failed decoding/parsing Anthropic JSON: {anthropic_json_str}"); return None, None
    openai_chunk={"id":f"chatcmpl-anth-{int(time.time()*1000)}-{random.randint(100,999)}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name, "choices":[{"index":0,"delta":{},"finish_reason":None}]}
    finish_reason=None; content_delta=None; role=None
    if event_type == "message_start":
        role=data.get("message",{}).get("role","assistant");
        if data.get("message", {}).get("usage"): usage_info['prompt_tokens'] = data["message"]["usage"].get("input_tokens")
    elif event_type == "content_block_delta":
        if data.get("delta",{}).get("type")=="text_delta": content_delta=data["delta"].get("text")
    elif event_type == "message_delta":
        stop_reason=data.get("delta",{}).get("stop_reason");
        if stop_reason: finish_reason = {"end_turn":"stop","max_tokens":"length","stop_sequence":"stop"}.get(stop_reason,"stop")
        if data.get("usage"): usage_info['completion_tokens']=data["usage"].get("output_tokens")
    elif event_type == "message_stop":
        bm=data.get("amazon-bedrock-invocationMetrics");
        if bm: usage_info['prompt_tokens']=bm.get("inputTokenCount"); usage_info['completion_tokens']=bm.get("outputTokenCount")
    if role: openai_chunk["choices"][0]["delta"]["role"]=role
    if content_delta: openai_chunk["choices"][0]["delta"]["content"]=content_delta
    if finish_reason: openai_chunk["choices"][0]["finish_reason"]=finish_reason
    if openai_chunk["choices"][0]["delta"] or finish_reason: openai_sse_str = f"data: {json.dumps(openai_chunk)}\n\n"
    final_usage = {k:v for k,v in usage_info.items() if v is not None}
    return openai_sse_str, final_usage if final_usage else None

# --- Manual Anthropic Non-Streaming Response Converter ---
def convert_anthropic_response_to_openai(anthropic_response_data: Dict[str, Any], client_model_name: str) -> Dict[str, Any]:
    # ...(Same as previous version - extracts final usage)...
    logging.debug("Converting full Anthropic response to OpenAI format")
    openai_response={"id":anthropic_response_data.get("id",f"chatcmpl-anth-{int(time.time())}"),"object":"chat.completion","created":int(time.time()),"model":client_model_name,"choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
    content_text = ""
    if isinstance(anthropic_response_data.get("content"),list) and len(anthropic_response_data["content"])>0:
        block=anthropic_response_data["content"][0];
        if block.get("type")=="text": content_text=block.get("text","")
    anthropic_stop=anthropic_response_data.get("stop_reason"); openai_finish="stop"
    if anthropic_stop=="end_turn": openai_finish="stop"
    elif anthropic_stop=="max_tokens": openai_finish="length"
    elif anthropic_stop=="stop_sequence": openai_finish="stop"
    openai_response["choices"].append({"index":0,"message":{"role":anthropic_response_data.get("role","assistant"),"content":content_text},"finish_reason":openai_finish})
    usage=anthropic_response_data.get("usage")
    if isinstance(usage,dict):
        pt=int(usage.get("input_tokens",0) or 0); ct=int(usage.get("output_tokens",0) or 0)
        openai_response["usage"]={"prompt_tokens":pt,"completion_tokens":ct,"total_tokens":pt+ct}
    logging.debug(f"Converted OpenAI response: {json.dumps(openai_response)}")
    return openai_response

# --- Flask Routes ---
@app.route('/v1/chat/completions', methods=['OPTIONS'])
def handle_options():
    resp=jsonify({}); resp.headers.add('Access-Control-Allow-Origin','*'); resp.headers.add('Access-Control-Allow-Headers','Content-Type,Authorization'); resp.headers.add('Access-Control-Allow-Methods','POST,OPTIONS'); resp.headers.add('Access-Control-Max-Age','86400'); return resp, 204

@app.route('/v1/models', methods=['GET'])
def list_models_endpoint():
    if not verify_request_token(request): return jsonify({"error":"Unauthorized"}), 401
    models=[{"id":m,"object":"model","created":int(time.time()),"owned_by":"sap-ai-core"} for m in aicore_deployment_urls.keys()]
    return jsonify({"object": "list", "data": models}), 200

# Main Chat Completions Proxy Endpoint
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions_proxy_endpoint():
    endpoint = "/v1/chat/completions"; request_start_time = time.time(); logging.info(f"POST {endpoint}")
    # ...(Steps 1-6: Verify Client, Fetch Token, Parse Request, Extract Params, Validate Config, Get Base URL, Get Provider)...
    if not verify_request_token(request): return jsonify({"error": "Invalid API Key"}), 401
    current_backend_token: Optional[str] = None
    try: current_backend_token = fetch_token()
    except Exception as e: logging.error(f"CRITICAL - Failed fetch token: {e}",exc_info=True); return jsonify({"error":"Proxy backend auth failed"}), 503
    try: request_data = request.get_json(); assert request_data
    except: return jsonify({"error": "Invalid JSON body"}), 400
    client_model_name: Optional[str] = request_data.get("model"); messages: Optional[list] = request_data.get("messages"); stream: bool = request_data.get("stream", False)
    if not client_model_name or not messages: return jsonify({"error": "Missing model or messages"}), 400
    if client_model_name not in aicore_deployment_urls: return jsonify({"error": f"Model '{client_model_name}' not configured."}), 404
    base_urls = aicore_deployment_urls[client_model_name]; selected_base_url = load_balance_url(base_urls, client_model_name)
    if not selected_base_url: return jsonify({"error": "Internal config error: No base URL."}), 500
    underlying_provider = get_underlying_provider(client_model_name)

    # --- Start of try block for the specific request handling ---
    try:
        if underlying_provider == "anthropic":
            # --- MANUAL ANTHROPIC CALL ---
            # ...(same logic as previous version, calls generate_anthropic_stream)...
            logging.info(f"Handling '{client_model_name}' manually (Anthropic via AI Core)")
            path = "/invoke-with-response-stream" if stream else "/invoke"
            final_endpoint_url = f"{selected_base_url.rstrip('/')}{path}"; logging.info(f"AI Core Anthropic URL: {final_endpoint_url}")
            headers = {"Authorization": f"Bearer {current_backend_token}", "Content-Type": "application/json", "AI-Resource-Group": resource_group, "Accept": "text/event-stream" if stream else "application/json"}
            anthropic_payload = convert_openai_to_anthropic_payload(request_data); logging.debug(f"Manual Anthropic Payload: {json.dumps(anthropic_payload)}")
            response = requests.post( final_endpoint_url, headers=headers, json=anthropic_payload, stream=stream, timeout=600 ); logging.info(f"Manual request sent. Status: {response.status_code}"); response.raise_for_status()
            if stream: return Response(stream_with_context(generate_anthropic_stream(response, endpoint, client_model_name, request)), mimetype='text/event-stream')
            else: # Non-streaming manual Anthropic
                 logging.info(f"Processing non-streaming manual response for {client_model_name}")
                 try: anthropic_response_data = response.json(); logging.debug(f"Raw Anthropic response: {json.dumps(anthropic_response_data)}")
                 except json.JSONDecodeError as json_err: logging.error(f"Failed decoding Anthropic JSON: {json_err}. Text: {response.text[:500]}"); return jsonify({"error":"Failed parse backend response."}), 502
                 openai_response_dict = convert_anthropic_response_to_openai(anthropic_response_data, client_model_name)
                 log_token_usage(request, client_model_name, openai_response_dict.get('usage')) # Log from converted data
                 return jsonify(openai_response_dict)

        elif underlying_provider == "azure" or underlying_provider == "gemini":
            # --- LITELLM AZURE / GEMINI (via AI Core) CALL ---
            # ...(same logic as previous version, calls generate_litellm_stream_response)...
            logging.info(f"Handling '{client_model_name}' using Litellm (ProviderHint=azure)")
            final_endpoint_url: Optional[str] = None; path="/chat/completions"; api_ver="2023-05-15"
            if underlying_provider=="gemini": api_ver="2023-12-01-preview" # Assumed
            elif "o3-mini" in client_model_name.lower(): api_ver="2024-12-01-preview"
            final_endpoint_url = f"{selected_base_url.rstrip('/')}{path}?api-version={api_ver}"; logging.info(f"AI Core Azure/Gemini URL: {final_endpoint_url}")
            litellm_params = { # ...(assemble params)...
                 "model": client_model_name, "custom_llm_provider": "azure", "api_base": final_endpoint_url,
                 "api_key": current_backend_token, "headers": { "AI-Resource-Group": resource_group },
                 "messages": messages, "stream": stream,
                 **{k: request_data[k] for k in ["temperature","top_p","max_tokens","n","stop","presence_penalty","frequency_penalty","user","logit_bias","logprobs","top_logprobs","tools","tool_choice","response_format"] if k in request_data}
            }
            logging.info(f"Attempting Litellm call (Hint: azure) for '{client_model_name}'"); logging.debug(f"Litellm params: { {k: (v if k!='api_key' else '***') for k,v in litellm_params.items()} }")
            if stream:
                response_generator = litellm.completion(**litellm_params)
                return Response(stream_with_context(generate_litellm_stream_response(response_generator, endpoint, client_model_name, request)), mimetype='text/event-stream')
            else: # Non-streaming Litellm
                response: ModelResponse = litellm.completion(**litellm_params); response_dict = response.dict(); logging.debug("Litellm non-streaming response received.")
                log_token_usage(request, client_model_name, response_dict.get('usage')); return jsonify(response_dict)
        else: # Should not be reached
            logging.error(f"Internal Error: Unknown provider '{underlying_provider}'"); return jsonify({"error": "Internal config error."}), 500

    # --- Exception Handling ---
    # ...(Keep combined exception handling)...
    except requests.exceptions.RequestException as e: status_code = e.response.status_code if hasattr(e,'response') and e.response is not None else 503; error_msg = f"Conn/HTTP Err (Manual): {e}"; logging.error(f"{endpoint}: {error_msg}", exc_info=True); detail=None; return jsonify({"error": {"message": error_msg, "backend_detail": detail}}), status_code
    except litellm.exceptions.AuthenticationError as e: logging.error(f"Litellm Auth Err: {e}",exc_info=True); return jsonify({"error":f"Backend auth failed: {e}"}), 500
    #...(other litellm exceptions)...
    except Exception as e: logging.error(f"{endpoint}: Unhandled Exception: {e}", exc_info=True); logging.error(traceback.format_exc()); return jsonify({"error":f"Internal proxy error: {e}"}), 500
    finally: # Corrected variable name
        duration = time.time()-request_start_time
        logging.info(f"{endpoint}: Request for '{client_model_name}' completed. Provider='{underlying_provider}'. Duration: {duration:.3f}s")


# --- Updated Helper function for MANUAL Anthropic streaming ---
def generate_anthropic_stream(response: requests.Response, endpoint: str, client_model_name: str, flask_request: Request) -> Generator[bytes, None, None]:
    """Processes Anthropic stream, converts chunks, aggregates usage, yields OpenAI SSE format."""
    sse_buffer = ""; chunks_yielded = 0; final_usage_data={}; usage_found = False
    start_time = time.time(); logging.info(f"{endpoint}: Starting robust manual stream processing for {client_model_name}")
    try:
        for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
            if not chunk: continue
            sse_buffer += chunk
            while '\n\n' in sse_buffer: # Process complete messages
                msg_end_idx=sse_buffer.find('\n\n'); complete_msg=sse_buffer[:msg_end_idx+2]; sse_buffer=sse_buffer[msg_end_idx+2:]
                cleaned_msg = "\n".join(line.strip() for line in complete_msg.strip().split('\n'))
                json_start_idx = cleaned_msg.find("data: ")
                if json_start_idx != -1:
                     json_part = cleaned_msg[json_start_idx+len("data: "):].strip()
                     if json_part:
                          openai_sse_str, usage_info = convert_anthropic_chunk_to_openai(json_part, client_model_name)
                          if usage_info: logging.debug(f"Usage from Anthropic chunk: {usage_info}"); final_usage_data.update({k:v for k,v in usage_info.items() if v is not None}); usage_found=True
                          if openai_sse_str:
                              try: yield openai_sse_str.encode('utf-8'); chunks_yielded+=1
                              except: logging.error("Stream yield error"); raise StopIteration
        #...(process remaining buffer)...
        if sse_buffer.strip(): # Process remaining buffer
             logging.warning(f"Processing remaining buffer: {sse_buffer.strip()}")
             cleaned_final_message = "\n".join(line.strip() for line in sse_buffer.strip().split('\n'))
             json_start_index = cleaned_final_message.find("data: ")
             if json_start_index != -1:
                 json_part = cleaned_final_message[json_start_index + len("data: "):].strip()
                 if json_part:
                     openai_sse_str, usage_info = convert_anthropic_chunk_to_openai(json_part, client_model_name)
                     if usage_info: final_usage_data.update({k:v for k,v in usage_info.items() if v is not None}); usage_found = True
                     if openai_sse_str:
                         try: yield openai_sse_str.encode('utf-8'); chunks_yielded += 1
                         except Exception as yield_err: logging.error(f"Stream yield error on final buffer: {yield_err}")

        yield "data: [DONE]\n\n".encode('utf-8')
        logging.info(f"{endpoint}: Manual stream finished for {client_model_name}. Chunks: {chunks_yielded}. Time: {time.time()-start_time:.2f}s")
    except requests.exceptions.ChunkedEncodingError as cee: logging.error(f"ChunkedEncodingError: {cee}"); #...(yield error chunk)...
    except Exception as e: logging.error(f"Manual stream error: {e}", exc_info=True); #...(yield error chunk)...
    finally:
        response.close()
        # --- Always call log_token_usage in finally block ---
        logging.debug(f"[Anthropic Stream Finally] Final Usage Data: {final_usage_data}, Usage Found Flag: {usage_found}")
        # Calculate total tokens if components exist
        if final_usage_data and 'total_tokens' not in final_usage_data:
             final_usage_data["total_tokens"] = final_usage_data.get("prompt_tokens", 0) + final_usage_data.get("completion_tokens", 0)
        log_token_usage(flask_request, client_model_name, final_usage_data if final_usage_data else {}) # Pass empty dict if None


# --- Updated Helper function for Litellm streaming response generation ---
def generate_litellm_stream_response(response_generator: Generator, endpoint: str, client_model_name: str, flask_request: Request) -> Generator[bytes, None, None]:
    """Generator function to yield Litellm SSE formatted chunks and log usage if available (usually isn't)."""
    final_usage_data = None; start_time = time.time(); chunks_yielded = 0; usage_found = False
    logging.info(f"{endpoint}: Litellm stream starting for {client_model_name}")
    try:
        for chunk in response_generator:
            try: chunk_dict = chunk.dict()
            except: logging.error("Litellm stream chunk format error"); continue
            # Check for usage, but don't expect it until maybe the end
            if chunk_dict.get('usage'): final_usage_data = chunk_dict['usage']; usage_found = True; logging.debug(f"Litellm usage chunk found: {final_usage_data}")
            try: yield f"data: {json.dumps(chunk_dict)}\n\n".encode('utf-8'); chunks_yielded+=1
            except: logging.error("Stream yield error"); break
        yield "data: [DONE]\n\n".encode('utf-8')
        logging.info(f"{endpoint}: Litellm stream finished for {client_model_name}. Chunks: {chunks_yielded}. Time: {time.time()-start_time:.2f}s")
    except Exception as e: logging.error(f"{endpoint}: Litellm stream error: {e}", exc_info=True); #...(yield error chunk)...
    finally:
        # --- Always call log_token_usage in finally block ---
        logging.debug(f"[Litellm Stream Finally] Final Usage Data: {final_usage_data}, Usage Found Flag: {usage_found}")
        # Calculate total tokens if components exist
        if final_usage_data and 'total_tokens' not in final_usage_data:
             final_usage_data["total_tokens"] = final_usage_data.get("prompt_tokens", 0) + final_usage_data.get("completion_tokens", 0)
        log_token_usage(flask_request, client_model_name, final_usage_data if final_usage_data else {}) # Pass empty dict if None


# --- Main Execution Block ---
if __name__ == '__main__':
    args = parse_arguments()
    logging.info(f"--- Starting SAP AI Core Universal LLM Proxy v8 (Hybrid + Guaranteed Log Attempt) ---")
    logging.info(f"Loading configuration from: {args.config}")
    try: config = load_config(args.config)
    except: logging.critical("Exiting: Config loading failure."); exit(1)
    # ...(Load config values and validate)...
    service_key_json_path=config.get('service_key_json'); aicore_deployments_config=config.get('deployment_models',{}); secret_authentication_tokens=config.get('secret_authentication_tokens',[]); resource_group=config.get('resource_group',''); host=config.get('host','0.0.0.0'); port=config.get('port',3001); crit_err=False
    if not aicore_deployments_config: logging.critical("Config Error: 'deployment_models' empty."); crit_err=True
    if not service_key_json_path: logging.critical("Config Error: 'service_key_json' missing."); crit_err=True
    if not resource_group: logging.critical("Config Error: 'resource_group' missing."); crit_err=True
    if not isinstance(secret_authentication_tokens,list): logging.critical("Config Error: 'secret_authentication_tokens' not list."); crit_err=True
    if not secret_authentication_tokens: logging.warning("PROXY SECURITY: Client auth disabled.")
    aicore_deployment_urls = {} # Validate URLs
    for model, urls in aicore_deployments_config.items():
        if not isinstance(urls, list) or not urls: logging.critical(f"Config Error: URLs for '{model}' invalid."); crit_err=True; continue
        for i, url in enumerate(urls):
             if not isinstance(url,str) or not url.startswith(('http://','https://')): logging.critical(f"Config Error: Invalid URL '{url}' for model '{model}'."); crit_err=True
        if not crit_err: aicore_deployment_urls[model] = urls
    if crit_err: logging.critical("Exiting due to critical config errors."); exit(1)
    try: service_key = load_config(service_key_json_path); assert isinstance(service_key,dict) and all(k in service_key for k in ['clientid','clientsecret','url'])
    except Exception: logging.critical(f"Exiting: Failed loading/validating service key."); exit(1)
    logging.info(f"Proxy configured for SAP AI Core models: {list(aicore_deployment_urls.keys())}")
    # --- Start Server ---
    logging.info(f"Starting Server on http://{host}:{port}")
    try: from waitress import serve; logging.info("Using Waitress."); serve(app, host=host, port=port, threads=10)
    except ImportError: logging.warning("Waitress not found. Using Flask dev server."); app.run(host=host, port=port, debug=False)
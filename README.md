# sap-ai-core LLM Proxy Server

This project establishes a proxy server to interface with SAP AI Core services, it transform the SAP AI Core LLM API to Open AI Compatible API, no matter it's gpt-4o or claude sonnet 4.

So it is compatible with any application that supports the OpenAI API, so you can use the SAP AI Core in other Applications, e.g. 
- [Cursor IDE](https://www.cursor.com/) 
- Cherry Studio
- Cline
- Lobe Chat
- Claude Code (via the Claude Code Router)
- ChatWise
- Or [Chitchat](https://github.com/pjq/ChitChat/)
- Or [ChatCat](https://github.com/pjq/ChatCat/) 

**Important Reminder**: It is crucial to follow the documentation precisely to ensure the successful deployment of the LLM model. Please refer to the official SAP AI Core documentation for detailed instructions and guidelines.
- https://developers.sap.com/tutorials/ai-core-generative-ai.html

Once the LLM model is deployed, obtain the URL and update it in the config.json file: `deployment_models`.


## Quick Start
```shell
python proxy_server.py --config config.json
```

### Debug Mode
For detailed logging and troubleshooting, you can enable debug mode:
```shell
python proxy_server.py --config config.json --debug
```

After you run the proxy server, you will get
- API BaseUrl: http://127.0.0.1:3001/v1
- API key will be one of secret_authentication_tokens. 
- Model ID: models you configured in the `deployment_models`

You can check the models list
- http://127.0.0.1:3001/v1/models

e.g.
```json
{
  "data": [
    {
      "created": 1750833737,
      "id": "4-sonnet",
      "object": "model",
      "owned_by": "sap-ai-core"
    },
    {
      "created": 1750833737,
      "id": "anthropic/claude-4-sonnet",
      "object": "model",
      "owned_by": "sap-ai-core"
    }
  ],
  "object": "list"
}
```

### OpenAI Embeddings API
The proxy server now supports OpenAI-compatible embeddings API:
- Endpoint: http://127.0.0.1:3001/v1/embeddings
- Compatible with OpenAI embeddings request format
- Transforms requests for SAP AI Core compatibility

## Overview
`sap-ai-core-llm-proxy` is a Python-based project that includes functionalities for token management, forwarding requests to the SAP AI Core API, and handling responses. The project uses Flask to implement the proxy server.

Now it supports the following LLM models
- OpenAI: gpt-4o, gpt-4.1, gpt-o3-mini, gpt-o3, gpt-o4-mini
- Claude: 3.5-sonnet, 3.7-sonnet, 4-sonnet
- Google Gemini: gemini-2.5-pro

## Features
- **Token Management**: Fetch and cache tokens for authentication.
- **Proxy Server**: Forward requests to the AI API with token management.
- **Load Balance**: Support the load balancing across multiple subAccounts and deployments.
- **Multi-subAccount Support**: Distribute requests across multiple SAP AI Core subAccounts.
- **Model Management**: List available models and handle model-specific requests.
- **OpenAI Embeddings API**: Support for text embedding functionality through the `/v1/embeddings` endpoint.
- **Debug Mode**: Enhanced logging capabilities with `--debug` command line flag for detailed troubleshooting.

## Prerequisites
- Python 3.x
- Flask
- Requests library

## Installation
1. Clone the repository:
    ```sh
    git clone git@github.com:pjq/sap-ai-core-llm-proxy.git
    cd sap-ai-core-llm-proxy
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration
1. Copy the example configuration file to create your own configuration file:
    ```sh
    cp config.json.example config.json
    ```

2. Edit `config.json` to include your specific details. The file supports multi-account configurations for different model types:

   ### Multi-Account Configuration 
   ```json
   {
       "subAccounts": {
           "subAccount1": {
               "resource_group": "default",
               "service_key_json": "demokey1.json",
               "deployment_models": {
                   "gpt-4o": [
                       "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_1>"
                   ],
                   "gpt-4.1": [
                       "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_1b>"
                   ],
                   "3.5-sonnet": [
                       "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_2>"
                   ]
               }
           },
           "subAccount2": {
               "resource_group": "default",
               "service_key_json": "demokey2.json",
               "deployment_models": {
                   "gpt-4o": [
                       "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_3>"
                   ],
                   "3.7-sonnet": [
                       "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_4>"
                   ],
                   "4-sonnet": [
                       "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_5>"
                   ]
               }
           }
       },
       "secret_authentication_tokens": ["<hidden_key_1>", "<hidden_key_2>"],
       "port": 3001,
       "host": "127.0.0.1"
   }
   ```

3. Get the service key files (e.g., `demokey.json`) with the following structure from the SAP AI Core Guidelines for each subAccount:
    ```json
    {
      "serviceurls": {
        "AI_API_URL": "https://api.ai.********.********.********.********.********.com"
      },
      "appname": "your_appname",
      "clientid": "your_client_id",
      "clientsecret": "your_client_secret",
      "identityzone": "your_identityzone",
      "identityzoneid": "your_identityzoneid",
      "url": "your_auth_url"
    }
    ```

4. [Optional] Place your SSL certificates (`cert.pem` and `key.pem`) in the project root directory if you want to start the local server with HTTPS.

## Multi-subAccount Load Balancing

The proxy now supports distributing requests across multiple subAccounts:

1. **Cross-subAccount Load Balancing**: Requests for a specific model are distributed across all subAccounts that have that model deployed.

2. **Within-subAccount Load Balancing**: For each subAccount, if multiple deployment URLs are configured for a model, requests are distributed among them.

3. **Automatic Failover**: If a subAccount or specific deployment is unavailable, the system will automatically try another.

4. **Model Availability**: The proxy consolidates all available models across all subAccounts, allowing you to use any model that's deployed in any subAccount.

5. **Token Management**: Each subAccount maintains its own authentication token with independent refresh cycles.

## Running the Proxy Server

### Running the Proxy Server over HTTP
Start the proxy server using the following command:
```sh
python proxy_server.py
```
The server will run on `http://127.0.0.1:3001`.

### Anthropic Claude Messages API Compatibility
The proxy server provides full compatibility with the Anthropic Claude Messages API through the `/v1/messages` endpoint. This allows you to use any application that supports the Claude Messages API directly with SAP AI Core.
- **Endpoint**: `http://127.0.0.1:3001/v1/messages`

### Supported Features
- **Non-streaming requests**: Standard request/response format
- **Streaming requests**: Server-sent events (SSE) with `"stream": true`
- **Multi-model support**: Works with Claude, GPT, and Gemini models deployed in SAP AI Core
- **Tool use**: Support for function calling and tool usage
- **System messages**: Support for system prompts
- **Multi-turn conversations**: Full conversation history support

### Anthropic Claude Integration with SAP AI Core
The project is use the official SAP AI SDK (`sap-ai-sdk-gen`) for Anthropic Claude integration. This method provides better compatibility and follows SAP's official guidelines.
- https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_reference/README_sphynx.html
#### Configuration
1. Create the configuration directory and file:
```shell
mkdir -p ~/.aicore
```

2. Create `~/.aicore/config.json` with your SAP AI Core credentials:
```json
{
  "AICORE_AUTH_URL": "https://*****.authentication.sap.hana.ondemand.com",
  "AICORE_CLIENT_ID": "*****",
  "AICORE_CLIENT_SECRET": "*****",
  "AICORE_RESOURCE_GROUP": "*****",
  "AICORE_BASE_URL": "https://api.ai.*****.cfapps.sap.hana.ondemand.com/v2"
}
```

Replace the `*****` placeholders with your actual SAP AI Core service credentials:
- `AICORE_AUTH_URL`: Your SAP AI Core authentication URL
- `AICORE_CLIENT_ID`: Your client ID from the service key
- `AICORE_CLIENT_SECRET`: Your client secret from the service key
- `AICORE_RESOURCE_GROUP`: Your resource group (typically "default")
- `AICORE_BASE_URL`: Your SAP AI Core API base URL

### Compatible Applications
Any application that supports the Anthropic Claude Messages API can now work with SAP AI Core through this proxy, including:
- Claude Code
- Claude SDK
- Anthropic API clients
- Custom applications using the Messages API format

### Claude Code
You need to set the enviroment variables before run the claude code.
```shell
export ANTHROPIC_AUTH_TOKEN=your_secret_key
export ANTHROPIC_BASE_URL=http://127.0.0.1:3001
export ANTHROPIC_MODEL=anthropic--claude-4-sonnet
```

Then run the claude code
```shell
claude
```

## Running the Proxy Server over HTTPS
To run the proxy server over HTTPS, you need to generate SSL certificates. You can use the following command to generate a self-signed certificate and key:

```sh
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

This will generate `cert.pem` and `key.pem` files. Place these files in the project root directory. Then, start the proxy server using the following command:
```sh
python proxy_server.py
```
Ensure that your `proxy_server.py` includes the following line to enable HTTPS:
```python
if __name__ == '__main__':
    logging.info("Starting proxy server...")
    app.run(host='127.0.0.1', port=8443, debug=True, ssl_context=('cert.pem', 'key.pem'))
```
The server will run on `https://127.0.0.1:8443`.

### Sending a Demo Request
You can send a demo request to the proxy server using the `proxy_server_demo_request.py` script:
```sh
python proxy_server_demo_request.py
```

## Running the Local Chat Application

To start the local chat application using `chat.py`, use the following command:
```shell
python3 chat.py 
python3 chat.py --model gpt-4o 
```
Example
```shell
python3 chat.py 
Starting chat with model: gpt-4o. Type 'exit' to end.
You: Hello who are you
Assistant: Hello! I'm an AI language model created by OpenAI. I'm here to help you with a wide range of questions and tasks. How can I assist you today?
You: 
```

## Cursor(AI IDE) Integration with SAP AI Core
You can run the proxy_server in your public server, then you can update the base_url in the Cursor model settings.
**Now ONLY gpt-4o supported**
Check the details
- https://forum.cursor.com/t/custom-api-keys-fail-with-the-model-does-not-work-with-your-current-plan-or-api-key/97422

## Cline Integration with SAP AI Core
You can integrate the SAP AI Core with Cline
Choose the API Provider -> OpenAI API Compatible
- Base URL: http://127.0.0.1:3001/v1
- API key: will be one of secret_authentication_tokens. 
- Model ID: models you configured in the `deployment_models`, e.g. 4-sonnet

Note: Cline is already official support SAP AI Core.

### Alternative: Claude Code Integration via Proxy

You can also use the proxy server approach with Claude Code Router:
- https://github.com/musistudio/claude-code-router

```shell
npm install -g @anthropic-ai/claude-code
npm install -g @musistudio/claude-code-router
```
Then start Claude Code
```shell
ccr code
```

Here is the config example

```shell
cat ~/.claude-code-router/config.json
```

```JSON
{
  "OPENAI_API_KEY": "your secret key",
  "OPENAI_BASE_URL": "http://127.0.0.1:3001/v1",
  "OPENAI_MODEL": "3.7-sonnet",
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "http://127.0.0.1:3001/v1",
      "api_key": "your secret key",
      "models": [
        "gpt-4o",
	    "3.7-sonnet",
	    "4-sonnet"
      ]
    }
  ],
  "Router": {
    "background": "gpt-4o",
    "think": "deepseek,deepseek-reasoner",
    "longContext": "openrouter,3.7-sonnet"
  }
}
```

## Cherry Studio Integration
Add Provider->Provider Type -> OpenAI

- API Key: will be one of secret_authentication_tokens. 
- API Host: http://127.0.0.1:3001
- Add Models: models you configured in the `deployment_models` 

### Claude Integration
It seems the Cursor IDE will block the request if the model contains claude, so we need to rename it to the name don't contains claude
- claud
- sonnet

Now I am using `3.7-sonnet`

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please contact [pengjianqing@gmail.com].

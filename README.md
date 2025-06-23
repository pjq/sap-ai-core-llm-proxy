# sap-ai-core LLM Proxy Server

This project establishes a proxy server to interface with SAP AI Core services, specifically tailored for handling Large Language Model (LLM) requests.

It is compatible with any application that supports the OpenAI API, so you can use it in other Applications, e.g. [Cursor IDE](https://www.cursor.com/) or [Chitchat](https://github.com/pjq/ChitChat/).

**Important Reminder**: It is crucial to follow the documentation precisely to ensure the successful deployment of the LLM model. Please refer to the official SAP AI Core documentation for detailed instructions and guidelines.
- https://developers.sap.com/tutorials/ai-core-generative-ai.html

Once the LLM model is deployed, obtain the URL and update it in the config.json file: `deployment_models`.

## Quick Start
```shell
python proxy_server.py --config config.json
```
After you run the proxy server, you will get
- API BaseUrl: http://127.0.0.1:3001/v1
- API key will be one of secret_authentication_tokens. 
- Model ID: models you configured in the `deployment_models`

## Overview
`sap-ai-core-llm-proxy` is a Python-based project that includes functionalities for token management, forwarding requests to the SAP AI Core API, and handling responses. The project uses Flask to implement the proxy server.

Now it supports the following LLM models
- OpenAI: gpt-4o, gpt-4.1, gpt-o3-mini
- Claude: 3.5-sonnet, 3.7-sonnet, 4-sonnet 

## Features
- **Token Management**: Fetch and cache tokens for authentication.
- **Proxy Server**: Forward requests to the AI API with token management.
- **Load Balance**: Support the load balancing across multiple subAccounts and deployments.
- **Multi-subAccount Support**: Distribute requests across multiple SAP AI Core subAccounts.
- **Model Management**: List available models and handle model-specific requests.

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

2. Edit `config.json` to include your specific details. The file supports both multi-account configurations:

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

### Running the Proxy Server over HTTPS
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

## Cursor(AI IDE) Integration
You can run the proxy_server in your public server, then you can update the base_url in the Cursor model settings

## CLINE Integration
Choose the API Provider -> OpenAI API Compatible
- Base URL: http://127.0.0.1:3001/v1
- API key: will be one of secret_authentication_tokens. 
- Model ID: models you configured in the `deployment_models`

## Cherry Studio Integration
Add Provider->Provider Type -> OpenAI

- API Key: will be one of secret_authentication_tokens. 
- API Host: http://127.0.0.1:3001
- Add Models: models you configured in the `deployment_models` 

### Claude Integration
It seems the IDE will block the request if the model contains claude, so we need to rename it to the name don't contains claude
- claud
- sonnet
Now I am using `3.5-sonnet`

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please contact [pengjianqing@gmail.com].
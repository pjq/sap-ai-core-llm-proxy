# sap-ai-core LLM Proxy Server

This project sets up a proxy server to interact with SAP AI Core services, specifically designed for handling Large Language Model (LLM) requests.

## Overview
`sap-ai-core-llm-proxy` is a Python-based project that includes functionalities for token management, forwarding requests to the SAP AI Core API, and handling responses. The project uses Flask to implement the proxy server.

## Features
- **Token Management**: Fetch and cache tokens for authentication.
- **Proxy Server**: Forward requests to the AI API with token management.
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

2. Edit `config.json` to include your specific details. The file should have the following structure:
    ```json
    {
        "service_key_json": "demokey.json",
        "deployment_models": {
            "gpt-4o": "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_1>",
            "gpt-4": "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_2>",
            "gpt-4-32k": "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<hidden_id_3>"
        },
        "secret_authentication_tokens": ["<hidden_key_1>", "<hidden_key_2>", "<hidden_key_3>", "<hidden_key_4>"],
        "resource_group": "default"
    }
    ```

3. Get a configuration file `demokey.json` with the following structure from the SAP AI Core Guidelines.
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

4. [Optional] Place your SSL certificates (`cert.pem` and `key.pem`) in the project root directory if you want to start the local server with HTTPS:
    ```python
    app.run(host='127.0.0.1', port=443, debug=True, ssl_context=('cert.pem', 'key.pem'))
    ```

## Usage

### Running the Proxy Server
Start the proxy server using the following command:
```sh
python proxy_server.py
```
The server will run on `http://127.0.0.1:5000`.

### Sending a Demo Request
You can send a demo request to the proxy server using the `proxy_server_demo_request.py` script:
```sh
python proxy_server_demo_request.py
```

## Cursor(AI IDE) Integration
You can run the proxy_server in your public server, then you can update the base_url in the Cursor model settings


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please contact [pengjianqing@gmail.com].

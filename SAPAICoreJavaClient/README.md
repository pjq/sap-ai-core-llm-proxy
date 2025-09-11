# SAP AI Core Java Client

This is a completely configuration-driven Java client library for connecting to SAP AI Core services. It provides simple methods to interact with OpenAI, Claude, and Gemini models through SAP AI Core.

## Features

- **Fully Configuration-Driven**: No hardcoded URLs, deployment IDs, or service keys in the code
- **Automatic Model Routing**: Automatically routes requests to the correct subaccount based on model name
- **Token Caching**: Automatically caches authentication tokens to avoid repeated token requests
- **Multiple Model Support**: Supports OpenAI, Claude, and Gemini models
- **Cross-Account Support**: Handles different service keys for different AI models
- **Simple API**: Easy-to-use methods for each AI service  
- **Thread-Safe**: Uses proper synchronization for token management
- **Flexible Configuration**: Uses deployment URLs and model mappings from configuration file

## Prerequisites

- Java 8 or higher
- Gradle
- A `config.json` file with your SAP AI Core configuration

## Dependencies

This project uses:
- OkHttp for HTTP client functionality
- Gson for JSON parsing
- JUnit 5 for testing

## Configuration

### Required: config.json File

The client requires a `config.json` file with the following structure:

```json
{
    "subAccounts": {
        "subAccount1": {
            "resource_group": "default",
            "service_key_json": "service_key_1.json",
            "deployment_models": {
                "gpt-4o": [
                    "https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-1"
                ],
                "gpt-4.1": [
                    "https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-2"
                ],
                "anthropic/claude-4-sonnet": [
                    "https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-3"
                ]
            }
        },
        "subAccount2": {
            "resource_group": "default", 
            "service_key_json": "service_key_2.json",
            "deployment_models": {
                "gemini-2.5-pro": [
                    "https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-4"
                ],
                "anthropic/claude-4-sonnet": [
                    "https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-5"
                ]
            }
        }
    }
}
```

### Service Key Files

Each subaccount references a service key JSON file. **Service key file paths are resolved relative to the config.json directory.**

For example, if your config.json is at `../config.json` and references `"service_key_json": "service_key.json"`, the client will look for the service key at `../service_key.json`.

```json
{
  "serviceurls": {
    "AI_API_URL": "https://api.ai.example-region.cloud.sap"
  },
  "clientid": "your-client-id",
  "clientsecret": "your-client-secret", 
  "identityzoneid": "your-identity-zone-id",
  "url": "https://your-auth-url.authentication.example-region.cloud.sap"
}
```

## Current Status

✅ **Fully Configuration-Driven**: No hardcoded URLs, keys, or deployment IDs in the code
✅ **All AI Models Functional**: OpenAI, Claude, and Gemini working with configuration-based routing
✅ **Multi-SubAccount Support**: Automatic model-to-subaccount mapping
✅ **Dynamic Model Discovery**: Convenience methods automatically find available models

## Usage

### Basic Usage

```java
import me.pjq.SAPAICoreClient;
import java.io.IOException;

public class Example {
    public static void main(String[] args) {
        try {
            // Initialize with config.json - completely configuration-driven
            SAPAICoreClient client = new SAPAICoreClient("config.json");
            
            // Use specific model names with custom temperature
            String gptResponse = client.postMessage("gpt-4o", "Hello, how are you?", 0.3);
            System.out.println("GPT-4o: " + gptResponse);
            
            String claudeResponse = client.postMessage("anthropic/claude-4-sonnet", "Hello!", 0.8);
            System.out.println("Claude: " + claudeResponse);
            
            String geminiResponse = client.postMessage("gemini-2.5-pro", "Hello!", 0.5);
            System.out.println("Gemini: " + geminiResponse);
            
            // Or use default temperature (0.7)
            String defaultTempResponse = client.postMessage("gpt-4o", "Hello with default temperature!");
            System.out.println("Default temp: " + defaultTempResponse);
            
            client.close();
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

### Using Convenience Methods

```java
try {
    SAPAICoreClient client = new SAPAICoreClient("config.json");
    
    // Convenience methods with custom temperature
    String openaiResponse = client.postMessageOpenAI("Hello, how are you?", 0.3);
    String claudeResponse = client.postMessageClaude("What is AI?", 0.8);
    String geminiResponse = client.postMessageGemini("Explain quantum computing", 0.5);
    
    // Or use default temperature (0.7)
    String defaultOpenAI = client.postMessageOpenAI("Hello with default temperature!");
    String defaultClaude = client.postMessageClaude("Hello with default temperature!");
    String defaultGemini = client.postMessageGemini("Hello with default temperature!");
    
    client.close();
    
} catch (IOException e) {
    e.printStackTrace();
}
```

## Constructor

### `SAPAICoreClient(String configPath)`
Initialize with a config.json file path. The client loads all configuration from the file including:
- Subaccount definitions
- Service key file paths (resolved relative to config.json directory)
- Model-to-deployment URL mappings
- Resource group settings

**Path Resolution**: Service key files specified in `service_key_json` are resolved relative to the directory containing the config.json file. For example:
- Config at: `../config.json`
- Service key reference: `"service_key_json": "service_key.json"`
- Actual path used: `../service_key.json`

## API Methods

### `postMessage(String model, String message, double temperature)` - **Primary Method**
Sends a message to any supported model with custom temperature. The client automatically:
- Finds the correct subaccount for the model
- Uses the appropriate API endpoint and format
- Handles authentication with the correct service key

**Parameters**:
- `model`: Model name (e.g., "gpt-4o", "claude-3.5-sonnet", "gemini-2.5-pro")
- `message`: The message to send
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = very creative)

### `postMessage(String model, String message)` - **Convenience Overload**
Same as above but uses default temperature of 0.7.

Example models (based on your configuration):
- `"gpt-4o"`, `"gpt-4.1"`, `"gpt-5"`, etc. (OpenAI models)
- `"anthropic/claude-4-sonnet"`, `"anthropic/claude-3.7-sonnet"`, etc. (Claude models)  
- `"gemini-2.5-pro"` (Gemini models)

### Convenience Methods

#### `postMessageOpenAI(String message, double temperature)`
Automatically finds the first OpenAI/GPT model in your configuration and sends the message with specified temperature.

#### `postMessageOpenAI(String message)`
Same as above but uses default temperature of 0.7.

#### `postMessageClaude(String message, double temperature)`
Automatically finds the first Claude model in your configuration and sends the message with specified temperature.

#### `postMessageClaude(String message)`
Same as above but uses default temperature of 0.7.

#### `postMessageGemini(String message, double temperature)`
Automatically finds the first Gemini model in your configuration and sends the message with specified temperature.

#### `postMessageGemini(String message)`
Same as above but uses default temperature of 0.7.

#### `close()`
Cleans up HTTP client resources. Should be called when done using the client.

## Configuration Benefits

### Completely Configuration-Driven
- **No Hardcoded Values**: All URLs, deployment IDs, and model mappings come from config.json
- **Environment Flexibility**: Easy to switch between development, staging, and production
- **Model Management**: Add/remove models without code changes
- **Load Balancing**: Support multiple deployment URLs per model

### Multiple Subaccounts
- **Cross-Account Support**: Different models can use different SAP AI Core subaccounts
- **Automatic Routing**: Client automatically determines which subaccount to use for each model
- **Token Management**: Independent token caching per subaccount

## Building and Running

### Build Commands

```bash
# Clean and build the project
./gradlew clean build

# Build without tests
./gradlew build -x test

# Run tests only
./gradlew test

# Build fat JAR (executable with all dependencies)
./gradlew shadowJar

# Build library JAR (for use as dependency, without Main class)
./gradlew shadowLibJar

# Build both JARs
./gradlew shadowJar shadowLibJar

# Generate distributions
./gradlew distZip distTar
```

#### Generated JAR Files

After running `./gradlew shadowJar shadowLibJar`, you'll find these files in `build/libs/`:

- **`sap-ai-core-client.jar`** - Executable fat JAR (~3.4MB) with all dependencies and Main class
- **`sap-ai-core-client-lib.jar`** - Library JAR (~3.4MB) with all dependencies, no Main class (for use as dependency)
- **`SAPAICoreJavaClient-1.0-SNAPSHOT.jar`** - Thin JAR (~14KB) with only your code (requires classpath)

### Running the Example

```bash
# Run the example application (demo mode - tests multiple models)
./gradlew run

# Or run the JAR directly
java -jar build/libs/sap-ai-core-client.jar
```

### Command Line Interface

The JAR supports various command-line options for easy testing:

```bash
# Show help
java -jar build/libs/sap-ai-core-client.jar --help

# Test specific model with custom message
java -jar build/libs/sap-ai-core-client.jar --model gpt-4o --message "What is AI?"

# Use custom config file
java -jar build/libs/sap-ai-core-client.jar --config ./my-config.json --model anthropic/claude-4-sonnet --message "Hello!"

# Set custom temperature (0.0-1.0)
java -jar build/libs/sap-ai-core-client.jar -m gemini-2.5-pro -msg "Explain quantum computing" -t 0.3

# List available models (from config)
java -jar build/libs/sap-ai-core-client.jar --list-models

# Debug mode for troubleshooting
java -jar build/libs/sap-ai-core-client.jar --model gpt-4o --message "Test" --debug
```

#### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--help` | `-h` | Show help message | - |
| `--config <file>` | `-c` | Config file path | `../config.json` |
| `--model <name>` | `-m` | Model name | - |
| `--message <text>` | `-msg` | Message to send | - |
| `--temperature <num>` | `-t` | Temperature (0.0-1.0) | `0.7` |
| `--list-models` | `-l` | List available models | - |
| `--debug` | - | Show debug information | - |

#### Expected Output

**Demo Mode (default):**
```
SAP AI Core Java Client - Demo Mode
===================================
Config: ../config.json

--- Testing GPT-4o ---
Request: Hello, how are you today?
Response: Hello! I'm here and ready to assist you. How can I help you today?

--- Testing Claude 4-Sonnet ---
Request: Hello, how are you today?
Response: Hello! I'm doing well, thank you for asking. I'm here and ready to help with whatever you'd like to discuss or work on. How are you doing today?

--- Testing Gemini 2.5 Pro ---
Request: Hello, how are you today?
Response: Hello! I'm doing great, thank you for asking. As an AI, I'm always ready to help.
```

**Single Test Mode:**
```bash
$ java -jar sap-ai-core-client.jar --model gpt-4o --message "What is the capital of France?" --temperature 0.3

SAP AI Core Java Client - Single Test Mode
==========================================
Config: ../config.json
Model: gpt-4o
Message: What is the capital of France?
Temperature: 0.3

Response:
The capital of France is Paris.
```

## Using the JAR Libraries

### Executable JAR (sap-ai-core-client.jar)

The executable JAR includes all dependencies and can be run directly:

```bash
# Run the example application
java -jar build/libs/sap-ai-core-client.jar

# Run with specific JVM options
java -Xmx1g -Dfile.encoding=UTF-8 -jar build/libs/sap-ai-core-client.jar

# Copy and run from anywhere (ensure config.json is accessible)
cp build/libs/sap-ai-core-client.jar /path/to/your/project/
cd /path/to/your/project/
java -jar sap-ai-core-client.jar
```

### Library JAR (sap-ai-core-client-lib.jar)

Use this JAR as a dependency in your own Java projects:

```bash
# Compile your project with the library
javac -cp "build/libs/sap-ai-core-client-lib.jar" YourApp.java

# Run your project with the library
java -cp "build/libs/sap-ai-core-client-lib.jar:." YourApp
```

#### Maven Usage

To use in a Maven project, install to local repository:

```bash
mvn install:install-file \
  -Dfile=build/libs/sap-ai-core-client-lib.jar \
  -DgroupId=me.pjq \
  -DartifactId=sap-ai-core-client \
  -Dversion=1.0-SNAPSHOT \
  -Dpackaging=jar
```

Then add to your `pom.xml`:

```xml
<dependency>
    <groupId>me.pjq</groupId>
    <artifactId>sap-ai-core-client</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

#### Gradle Usage

Copy the JAR to your project and add to `build.gradle`:

```gradle
dependencies {
    implementation files('libs/sap-ai-core-client-lib.jar')
}
```

### Alternative Run Methods

```bash
# Run the executable fat JAR directly
java -jar build/libs/sap-ai-core-client.jar

# Run with custom config file path (modify Main.java to accept arguments)
java -jar build/libs/sap-ai-core-client.jar /path/to/config.json

# Run from distribution
./build/distributions/SAPAICoreJavaClient-1.0-SNAPSHOT/bin/SAPAICoreJavaClient

# Run with custom JVM options
java -Xmx512m -Dfile.encoding=UTF-8 -jar build/libs/sap-ai-core-client.jar

# Use the library JAR in another Java project (add to classpath)
java -cp "build/libs/sap-ai-core-client-lib.jar:your-app.jar" com.yourcompany.YourMainClass
```

### Development Commands

```bash
# Continuous build (rebuilds on file changes)
./gradlew build --continuous

# Run with debug information
./gradlew run --debug

# Check for dependency updates
./gradlew dependencyUpdates

# Generate project report
./gradlew projectReport
```

## Error Handling

The client provides clear error messages for configuration issues:

- `Configuration file not found: <path>` - config.json file is missing
- `Model not found in configuration: <model>. Available models: <list>` - Requested model not configured
- `No deployment URL found for model: <model>` - Model exists but has no deployment URLs
- `SubAccount not found: <name>` - Invalid subaccount reference
- `No OpenAI/GPT models configured. Available models: <list>` - Convenience method can't find suitable model

## Token Management

The client automatically handles OAuth token management:
- Fetches tokens when needed per subaccount
- Caches tokens until expiry (with 5-minute buffer)
- Thread-safe token refresh across multiple subaccounts
- Automatic retry on token expiration

## Configuration Examples

### Single Subaccount Setup
```json
{
    "subAccounts": {
        "main": {
            "resource_group": "default",
            "service_key_json": "service_key.json",
            "deployment_models": {
                "gpt-4o": ["https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-1"],
                "claude-3.5": ["https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-2"],
                "gemini-2.5-pro": ["https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-3"]
            }
        }
    }
}
```

### Multi-Subaccount Setup with Load Balancing
```json
{
    "subAccounts": {
        "production": {
            "resource_group": "default",
            "service_key_json": "prod_key.json",
            "deployment_models": {
                "gpt-4o": [
                    "https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-1",
                    "https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-2"
                ]
            }
        },
        "experimental": {
            "resource_group": "experimental",
            "service_key_json": "exp_key.json",
            "deployment_models": {
                "gpt-5": ["https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-3"],
                "gemini-pro": ["https://api.ai.example-region.cloud.sap/v2/inference/deployments/deployment-id-4"]
            }
        }
    }
}
```

## Notes

- **Configuration Required**: The client requires a properly formatted config.json file
- **No Hardcoded Values**: All deployment URLs and model mappings must be in the configuration
- **Thread-Safe**: Safe for concurrent requests across different models/subaccounts
- **Model Auto-Discovery**: Convenience methods automatically find available models from configuration
- **No Streaming Support**: Uses non-streaming endpoints only
- **Resource Groups**: Configurable per subaccount (defaults to "default")
# SAP AI Core Now Supports Claude Code: A Game-Changer for Enterprise Development

**Breaking News**: Enterprise developers can finally use Claude Code with SAP AI Core! The `sap-ai-core-llm-proxy` now provides native Anthropic Messages API support, making this integration seamless and straightforward.
- http://127.0.0.1:3001/v1/messages

## The Breakthrough

Until now, Claude Code couldn't connect to SAP AI Core because it requires Anthropic's specific `/v1/messages` API format. The missing piece was a bridge that could translate between SAP AI Core's API and what Claude Code expects.

**That bridge now exists.**

## Quick Setup Guide

### 1. Get the Proxy
```bash
git clone https://github.com/pjq/sap-ai-core-llm-proxy.git
cd sap-ai-core-llm-proxy
pip install -r requirements.txt
```

### 2. Configure SAP AI Core SDK (Recommended Method)
The proxy is using sap-ai-sdk-gen Python SDK, you can read the document to get more configuration details.
- [SAP AI Core SDK Guideline](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_reference/README_sphynx.html) 

First, create the AI Core configuration directory:
```bash
mkdir -p ~/.aicore
```

Then create `~/.aicore/config.json` with your SAP AI Core credentials:
```json
{
  "AICORE_AUTH_URL": "https://*****.authentication.sap.hana.ondemand.com",
  "AICORE_CLIENT_ID": "*****",
  "AICORE_CLIENT_SECRET": "*****",
  "AICORE_RESOURCE_GROUP": "*****",
  "AICORE_BASE_URL": "https://api.ai.*****.cfapps.sap.hana.ondemand.com/v2"
}
```

Replace the `*****` placeholders with your actual SAP AI Core service credentials from your service key.

### 3. Configure the AI Core Proxy Server
Copy and edit the main configuration:
```bash
cp config.json.example config.json
```

Set up your SAP AI Core deployments:
```json
{
    "subAccounts": {
        "production": {
            "resource_group": "default",
            "service_key_json": "sap-ai-core-key.json",
            "deployment_models": {
                "anthropic--claude-4-sonnet": [
                    "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/your-deployment-id"
                ]
            }
        }
    },
    "secret_authentication_tokens": ["your-secret-token"],
    "port": 3001
}
```

And make sure you have already downloaded the service key json `sap-ai-core-key.json`

### 4. Start the Proxy
```bash
python proxy_server.py --config config.json --debug
```

### 5. Configure Claude Code Environment
```bash
export ANTHROPIC_AUTH_TOKEN=your-secret-token
export ANTHROPIC_BASE_URL=http://127.0.0.1:3001
export ANTHROPIC_MODEL=anthropic--claude-4-sonnet
```

### 6. Launch Claude Code
```bash
claude
```

That's it! Claude Code now uses your SAP AI Core Claude deployments through the official SAP AI SDK.

## Why This Configuration Matters

The `~/.aicore/config.json` file uses the **official SAP AI SDK** (`sap-ai-sdk-gen`) for Anthropic Claude integration. This approach:

- ✅ **Follows SAP's official guidelines**
- ✅ **Provides better compatibility** 
- ✅ **Ensures enterprise security standards**
- ✅ **Simplifies credential management**

## What This Means for Developers

### 🚀 **Enterprise AI Coding**
Use Claude 4 Sonnet's advanced coding capabilities through your company's SAP AI Core infrastructure.

### 🔒 **Data Sovereignty** 
Your code conversations never leave your enterprise environment.

### 💰 **Cost Control**
Leverage enterprise pricing instead of individual API subscriptions.

### 🔄 **Universal Compatibility**
The same proxy works with Cursor IDE, Cline, Cherry Studio, and other popular development tools.

## Why This Works

The proxy implements the exact `/v1/messages` endpoint that Claude Code expects:

- **Native Anthropic API**: Full compatibility with Claude's message format
- **Streaming Support**: Real-time responses with proper SSE formatting  
- **Tool Use**: Function calling and advanced features work seamlessly
- **Multi-turn Conversations**: Complete conversation history support

## Real-World Benefits

### For Individual Developers
- Access Claude 4 Sonnet through enterprise infrastructure
- No personal API costs
- Compliance with company data policies

### For Development Teams  
- Standardized AI coding assistance across the organization
- Centralized usage tracking and management
- Consistent model versions and capabilities

### For Enterprise Architects
- Single platform for all AI development tools
- Full audit trail and governance
- Integration with existing SAP ecosystem

## Beyond Claude Code

The same proxy supports multiple popular development tools:

- **Cursor IDE** - AI-powered code editor
- **Cline** - VS Code extension for AI assistance  
- **Cherry Studio** - AI chat interface
- **Lobe Chat** - Conversational AI platform
- **Custom Applications** - Any tool using OpenAI or Anthropic APIs

## The Bottom Line

What was impossible before is now a simple setup process. Enterprise developers no longer have to choose between:

❌ **SAP AI Core's enterprise benefits** OR **Claude Code's superior assistance**

✅ **Now you can have both!**

## Get Started

1. **Check out the project**: [sap-ai-core-llm-proxy on GitHub](https://github.com/pjq/sap-ai-core-llm-proxy)
2. **Follow the setup guide** above, especially the `~/.aicore/config.json` configuration
3. **Start coding** with enterprise-grade AI assistance

The future of enterprise AI development is here. Ready to revolutionize your workflow?

---

*💡 **Pro Tip**: The proxy supports load balancing across multiple SAP AI Core deployments for high availability and better performance.*

*🔗 **Learn More**: Check out the [SAP AI Core Guidelines](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_reference/README_sphynx.html) for advanced configuration options.*

# Token Usage Logging Guide

## Overview

The proxy server logs all API requests to `logs/token_usage.log` in a structured JSON format for easy analysis and monitoring.

## Log Format

Each log entry is a JSON object with the following fields:

```json
{
  "timestamp": "2026-01-27T17:30:45.123456",
  "request_id": "1769504445123-4567",
  "user": "Bearer sk-or-v1-3eeb88...",
  "ip": "127.0.0.1",
  "model": "gpt-4o",
  "subaccount": "SFMobileSH",
  "tokens": {
    "prompt": 150,
    "completion": 45,
    "total": 195
  },
  "streaming": false,
  "duration_ms": 1234,
  "status": "success"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 formatted timestamp |
| `request_id` | string | Unique identifier for tracking individual requests |
| `user` | string | Authentication token (truncated for security) |
| `ip` | string | Client IP address |
| `model` | string | Model name used for the request |
| `subaccount` | string | SubAccount that handled the request |
| `tokens.prompt` | integer | Number of prompt/input tokens |
| `tokens.completion` | integer | Number of completion/output tokens |
| `tokens.total` | integer | Total tokens used |
| `streaming` | boolean | Whether this was a streaming request |
| `duration_ms` | integer | Request duration in milliseconds |
| `status` | string | `success` or `error` |
| `error` | string | Error message (only present if status is error) |

## Analyzing Logs

Use the included `analyze_token_logs.py` script to analyze token usage:

### Basic Usage

```bash
# Analyze all logs
python3 analyze_token_logs.py

# Filter by model
python3 analyze_token_logs.py --model gpt-4o

# Filter by subaccount
python3 analyze_token_logs.py --subaccount SFMobileSH

# Custom log file
python3 analyze_token_logs.py --log-file /path/to/custom.log
```

### Example Output

```
================================================================================
TOKEN USAGE ANALYSIS REPORT
================================================================================

📊 OVERALL STATISTICS
Total API Calls:                    6,258
  Streaming calls:                  4,833
  Non-streaming calls:              1,425
Successful calls:                   6,258
Error calls:                            0
Streaming with 0 tokens:            3,047

💰 TOKEN USAGE
Total tokens consumed:          3,349,553
Average tokens per call:            535.2

⏱️  PERFORMANCE
Average request duration:            1,234 ms

📱 BY MODEL (Top 10)
Model                             Calls       Tokens   Avg/Call
--------------------------------------------------------------------------------
gpt-4o                            2,519        5,884          2
3.7-sonnet                        1,800      976,250        542
4-sonnet                            794      120,883        152

🏢 BY SUBACCOUNT
SubAccount                        Calls       Tokens   Avg/Call
--------------------------------------------------------------------------------
erica                               576    2,264,971      3,932
SFMobileSH                          552       52,625         95
jack                                449      163,610        364

👤 BY USER (Top 10)
User                                   Calls       Tokens   Avg/Call
--------------------------------------------------------------------------------
Bearer d7024118d5025...                2,005    1,812,085        904
Bearer sk-or-v1-3eeb...                1,226    1,514,034      1,235

📅 BY DATE (Last 7 days)
Date                    Calls       Tokens   Avg/Call
--------------------------------------------------------------------------------
2026-01-27                 15          450         30
2026-01-26                  1            0          0
```

## Common Use Cases

### 1. Monitor Daily Usage

```bash
# Run daily via cron to track usage trends
0 0 * * * cd /path/to/proxy && python3 analyze_token_logs.py > daily_report.txt
```

### 2. Track Costs Per SubAccount

Use the "BY SUBACCOUNT" section to see which subaccounts are using the most tokens.

### 3. Identify Performance Issues

Check the "PERFORMANCE" section for average request duration. High values may indicate:
- Network latency issues
- Backend service slowness
- Large prompt/completion sizes

### 4. Debug Streaming Issues

The script identifies "Streaming with 0 tokens" which indicates:
- Requests where token counting failed
- Empty responses
- Errors in streaming response handling

### 5. Audit API Usage by User

Track which API keys are using the most resources:
```bash
python3 analyze_token_logs.py | grep -A 20 "BY USER"
```

## Advanced Analysis with jq

Since logs are in JSON format, you can use `jq` for custom queries:

```bash
# Get all errors
cat logs/token_usage.log | grep '"status":"error"' | jq .

# Calculate total tokens for a specific model today
cat logs/token_usage.log | grep "$(date +%Y-%m-%d)" | grep '"model":"gpt-4o"' | jq '.tokens.total' | awk '{sum+=$1} END {print sum}'

# Find slowest requests (> 5 seconds)
cat logs/token_usage.log | jq 'select(.duration_ms > 5000)'

# Get unique users
cat logs/token_usage.log | jq -r '.user' | sort -u

# Average tokens by model
cat logs/token_usage.log | jq -r '[.model, .tokens.total] | @tsv' | \
  awk '{sum[$1]+=$2; count[$1]++} END {for (model in sum) print model, sum[model]/count[model]}'
```

## Log Rotation

The log file can grow large over time. Consider setting up log rotation:

### Using logrotate (Linux)

Create `/etc/logrotate.d/sap-ai-proxy`:

```
/path/to/proxy/logs/token_usage.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 user group
    postrotate
        # Optional: Generate daily report before rotation
        cd /path/to/proxy && python3 analyze_token_logs.py > logs/daily_report_$(date +%Y%m%d).txt
    endscript
}
```

### Manual Rotation

```bash
# Rotate log file manually
mv logs/token_usage.log logs/token_usage_$(date +%Y%m%d).log
gzip logs/token_usage_$(date +%Y%m%d).log
touch logs/token_usage.log
```

## Troubleshooting

### Issue: Streaming requests show 0 tokens

**Cause**: Token counting for streaming responses may fail if:
- The backend doesn't return token usage metadata
- Response parsing errors occur
- Stream is interrupted

**Solution**:
1. Check backend logs for errors
2. Verify model supports token usage in streaming mode
3. Update token extraction logic if needed

### Issue: High average duration

**Cause**: Slow request processing

**Investigation**:
```bash
# Find slowest requests
cat logs/token_usage.log | jq 'select(.duration_ms > 10000) | {request_id, model, duration_ms}'
```

### Issue: Many error status entries

**Cause**: Requests are failing

**Investigation**:
```bash
# Get error details
cat logs/token_usage.log | grep '"status":"error"' | jq '{timestamp, model, error}'
```

## Migration from Old Format

The old log format was plain text:
```
2025-12-01 18:20:18,266 - User: Bearer d7024118d5025..., IP: 10.128.39.148, Model: gpt-4o, SubAccount: SFMobileSH, PromptTokens: 516, CompletionTokens: 3, TotalTokens: 519
```

The new format is JSON:
```json
{"timestamp": "2025-12-01T18:20:18.266", "request_id": "...", "user": "Bearer d7024118d5025...", "ip": "10.128.39.148", "model": "gpt-4o", "subaccount": "SFMobileSH", "tokens": {"prompt": 516, "completion": 3, "total": 519}, "streaming": false, "duration_ms": 1234, "status": "success"}
```

The `analyze_token_logs.py` script supports **both formats** for backwards compatibility.

## Best Practices

1. **Regular Analysis**: Run the analyzer weekly to track trends
2. **Set Alerts**: Monitor for unusual spikes in token usage
3. **Archive Old Logs**: Keep compressed archives for historical analysis
4. **Track Costs**: Map token counts to costs based on your pricing
5. **Performance Monitoring**: Watch duration_ms to catch slowdowns early
6. **Error Tracking**: Monitor error rates by model and subaccount

## Cost Estimation

Add cost calculations based on your pricing:

```python
# Example: Calculate costs (adjust prices for your models)
PRICING = {
    "gpt-4o": {"prompt": 0.005 / 1000, "completion": 0.015 / 1000},  # per token
    "claude-4.5-sonnet": {"prompt": 0.003 / 1000, "completion": 0.015 / 1000},
    # ... add more models
}

# Use with analyzer script to calculate costs per model/subaccount
```

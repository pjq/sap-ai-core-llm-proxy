# Proxy Server Unresponsiveness Fixes

## Date: 2026-01-27

## Problem Identified

The proxy server was becoming unresponsive after running for some time due to:

1. **Connection Pool Exhaustion**: 242 established connections were observed, exceeding the configured pool limits (10 pools × 20 connections = 200 max)
2. **Thread Starvation**: Waitress was configured with only 40 threads, insufficient for handling many concurrent streaming requests
3. **No Connection Cleanup**: Idle connections were never cleaned up, accumulating over time
4. **No Health Check**: Difficult to verify server health without making authenticated requests

## Fixes Applied

### 1. Increased Connection Pool Limits (proxy_server.py:127-128)
- `pool_connections`: 10 → 50 (5x increase)
- `pool_maxsize`: 20 → 100 (5x increase)
- Total capacity: 200 → 5,000 connections

### 2. Enhanced Waitress Server Configuration (proxy_server.py:3037-3048)
- `threads`: 40 → 100 (2.5x increase for concurrent requests)
- `connection_limit`: 1000 → 2000 (2x increase)
- `cleanup_interval`: 30 → 15 seconds (faster cleanup)
- `channel_timeout`: 120 → 300 seconds (better for long streaming)
- Added `recv_bytes` and `send_bytes`: 65536 (improved buffer sizes)
- Enabled `asyncore_use_poll`: True (better scalability)

### 3. Added Health Check Endpoint (proxy_server.py:1853-1859)
- New endpoint: `GET /health`
- No authentication required
- Returns: `{"status": "ok", "timestamp": <unix_time>, "server": "sap-ai-core-llm-proxy"}`
- Useful for monitoring and quick server status checks

### 4. Periodic Connection Pool Maintenance (proxy_server.py:2953-2978)
- Background thread runs every 5 minutes
- Clears idle connections from the pool
- Prevents connection accumulation over time
- Runs as daemon thread (won't block shutdown)

### 5. Improved Connection Cleanup (proxy_server.py:147-156)
- Honor client's `Connection: close` header
- Better connection lifecycle management

## Usage

### Restart the Server
```bash
./restart_proxy.sh
```

### Check Server Health
```bash
curl http://127.0.0.1:3001/health
```

### Monitor Connection Count
```bash
# See active connections
netstat -an | grep 3001 | grep ESTABLISHED | wc -l

# See all connections by state
netstat -an | grep 3001
```

### View Logs
```bash
# Real-time logs
tail -f logs/proxy.log

# Token usage logs
tail -f logs/token_usage.log
```

## Expected Improvements

1. **No More Connection Reset**: The `/v1/models` endpoint should respond immediately
2. **Better Concurrency**: Can now handle 100+ concurrent streaming requests
3. **Automatic Cleanup**: Connections are cleaned up every 5 minutes
4. **Monitoring**: Use `/health` endpoint to verify server is responsive
5. **Longer Uptime**: Server should remain responsive for extended periods

## Testing

After restart, verify:
```bash
# 1. Health check
curl http://127.0.0.1:3001/health

# 2. Models endpoint
curl http://127.0.0.1:3001/v1/models

# 3. Check connection count (should be low after restart)
netstat -an | grep 3001 | grep ESTABLISHED | wc -l
```

## Configuration Changes Summary

| Setting | Before | After | Reason |
|---------|--------|-------|--------|
| pool_connections | 10 | 50 | Handle more concurrent backends |
| pool_maxsize | 20 | 100 | More connections per backend |
| waitress threads | 40 | 100 | Handle more concurrent requests |
| connection_limit | 1000 | 2000 | Accept more incoming connections |
| cleanup_interval | 30s | 15s | Faster idle connection cleanup |
| channel_timeout | 120s | 300s | Better for long streaming responses |

## Monitoring Recommendations

1. Watch connection counts regularly:
   ```bash
   watch -n 5 'netstat -an | grep 3001 | grep ESTABLISHED | wc -l'
   ```

2. Set up alerts if connections exceed 1500

3. Monitor the health endpoint from monitoring tools

4. Review logs for connection pool warnings

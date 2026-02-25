"""
Integration Guide for Enhanced Load Balancer and Token Manager

This document explains how to integrate the new enhanced modules into proxy_server.py
"""

## Overview

The new modules provide:
1. **enhanced_load_balancer.py**: Smart load balancing with health tracking and circuit breakers
2. **enhanced_token_manager.py**: Proactive token refresh with retry logic

## Integration Steps

### Step 1: Import the New Modules

Add these imports at the top of `proxy_server.py`:

```python
from enhanced_load_balancer import get_load_balancer, EnhancedLoadBalancer
from enhanced_token_manager import get_token_manager, EnhancedTokenManager
```

### Step 2: Initialize After Config Loading

After loading the proxy configuration, initialize the enhanced components:

```python
# After proxy_config.initialize()
load_balancer = get_load_balancer(proxy_config)
token_manager = get_token_manager(_http_session)

# Register all subaccounts with token manager
for subaccount_name, subaccount in proxy_config.subaccounts.items():
    if subaccount.service_key:
        token_manager.register_subaccount(subaccount_name, subaccount.service_key)
```

### Step 3: Replace Token Fetching

Replace calls to `fetch_token(subaccount_name)` with:

```python
token = token_manager.get_token(subaccount_name)
if not token:
    return jsonify({"error": "Failed to obtain authentication token"}), 500
```

### Step 4: Replace Load Balancing

Replace calls to `load_balance_url(model)` with:

```python
subaccount_name = load_balancer.select_subaccount(model)
if not subaccount_name:
    raise ValueError(f"No healthy subaccount available for model '{model}'")

selected_url = load_balancer.select_url(subaccount_name, model)
if not selected_url:
    raise ValueError(f"No URL found for model '{model}' in subaccount '{subaccount_name}'")

resource_group = proxy_config.subaccounts[subaccount_name].resource_group
```

### Step 5: Add Success/Failure Tracking

Wrap request handling with tracking:

```python
import time

start_time = time.time()
try:
    # ... make the request ...
    response = _http_session.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
    response_time_ms = (time.time() - start_time) * 1000
    
    load_balancer.record_success(subaccount_name, response_time_ms)
    
except Exception as e:
    load_balancer.record_failure(subaccount_name)
    raise
```

### Step 6: Add Health Check Endpoint (Optional)

Add a new endpoint to monitor health:

```python
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with load balancer and token status"""
    lb_stats = load_balancer.get_all_stats() if load_balancer else []
    token_stats = token_manager.get_all_token_info() if token_manager else []
    
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "load_balancer": lb_stats,
        "token_manager": token_stats
    })
```

### Step 7: Add Stats Endpoint (Optional)

Add endpoint for monitoring:

```python
@app.route('/stats', methods=['GET'])
def get_stats():
    """Get detailed statistics"""
    return jsonify({
        "load_balancer": load_balancer.get_all_stats() if load_balancer else [],
        "token_manager": token_manager.get_all_token_info() if token_manager else []
    })
```

## Configuration Options

### Load Balancer

```python
# Customize circuit breaker settings
for cb in load_balancer.circuit_breakers.values():
    cb.failure_threshold = 5  # failures before opening circuit
    cb.recovery_timeout = 60  # seconds before trying again
    cb.half_open_requests = 3  # successes needed to close circuit
```

### Token Manager

```python
# Adjust proactive refresh timing (already set to 80% of expiry)
# Or change background refresh interval
token_manager.start_background_refresh(interval=30)  # check every 30 seconds
```

## Migration Strategy

### Phase 1: Parallel Run (Recommended)
1. Keep existing `fetch_token` and `load_balance_url` functions
2. Add new enhanced versions alongside
3. Add feature flag to switch between them
4. Monitor and compare behavior

### Phase 2: Gradual Rollout
1. Enable enhanced load balancer for non-critical models first
2. Monitor error rates and latency
3. Gradually expand to all models

### Phase 3: Full Migration
1. Remove old functions
2. Clean up deprecated code
3. Update documentation

## Testing

### Unit Tests
```python
def test_load_balancer_health_tracking():
    lb = EnhancedLoadBalancer(proxy_config)
    
    # Record some failures
    lb.record_failure("subaccount1")
    lb.record_failure("subaccount1")
    
    # Check health status
    stats = lb.get_stats("subaccount1")
    assert stats["consecutive_failures"] == 2
```

### Integration Tests
```python
def test_token_refresh():
    token_manager = EnhancedTokenManager()
    token_manager.register_subaccount("test", service_key)
    
    token = token_manager.get_token("test")
    assert token is not None
```

## Rollback Plan

If issues occur:
1. Revert to old `fetch_token` and `load_balance_url` functions
2. Keep new modules for future use
3. Debug issues in isolation
4. Re-enable when fixed

## Monitoring

Key metrics to watch:
- Error rate per subaccount (should decrease)
- Token refresh failures (should be rare)
- Circuit breaker state changes (should be infrequent)
- Response time distribution (should improve or stay same)
- Request throughput (should improve under load)

## Benefits

After integration:
- ✅ Better fault tolerance with circuit breakers
- ✅ Proactive token refresh prevents expiry-related failures
- ✅ Health-aware load balancing avoids problematic subaccounts
- ✅ Detailed statistics for monitoring and debugging
- ✅ Better error messages and logging

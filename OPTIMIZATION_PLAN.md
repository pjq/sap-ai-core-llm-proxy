# Optimization Plan for SAP AI Core LLM Proxy

## Current Issues Identified

### 1. Load Balancing (Simple Round-Robin)
- **Issue**: Current implementation uses basic round-robin without considering:
  - Subaccount rate limits or quotas
  - Response times/latency
  - Error rates per subaccount
  - Health status of deployments
- **Impact**: Requests may be routed to slow or failing subaccounts

### 2. Connection Pool Management
- **Issue**: Single global `_http_session` shared across all subaccounts
  - No isolation between different SAP AI Core endpoints
  - Pool exhaustion can affect all requests
  - No per-subaccount connection limits
- **Impact**: One slow/failing subaccount can impact others

### 3. Token Refresh Strategy
- **Issue**: Tokens are refreshed only when expired (reactive)
  - No proactive refresh before expiry
  - Race conditions possible under high concurrency
  - No retry logic for token refresh failures
- **Impact**: Request failures during token refresh, potential race conditions

### 4. Error Handling & Resilience
- **Issue**: Limited error handling strategies
  - No circuit breaker pattern
  - No exponential backoff with jitter
  - Limited fallback options when all subaccounts fail
- **Impact**: Cascading failures, poor user experience during outages

### 5. Performance Bottlenecks
- **Issue**: Synchronous request handling
  - Blocking I/O for all external calls
  - No request batching for embeddings
  - Streaming could be optimized
- **Impact**: Limited throughput under high load

### 6. Code Quality
- **Issue**: Large monolithic file (3000+ lines)
  - Limited type hints
  - Mixed concerns (token management, load balancing, request handling)
  - No unit tests
- **Impact**: Hard to maintain, test, and extend

## Proposed Optimizations

### Phase 1: Quick Wins (Low Risk, High Impact)

#### 1.1 Improved Load Balancing
- Add weighted round-robin based on configurable weights per subaccount
- Track error rates per subaccount and temporarily deprioritize failing ones
- Add configurable health check interval

#### 1.2 Better Token Management
- Proactive token refresh (refresh at 80% of expiry time)
- Add token refresh queue to prevent race conditions
- Implement retry logic with exponential backoff for token refresh

#### 1.3 Enhanced Error Handling
- Add circuit breaker pattern for subaccounts
- Implement exponential backoff with jitter for retries
- Better fallback chain when primary subaccounts fail

### Phase 2: Structural Improvements (Medium Risk)

#### 2.1 Connection Pool Optimization
- Create per-subaccount HTTP sessions for better isolation
- Add connection pool monitoring and metrics
- Implement pool size auto-adjustment based on load

#### 2.2 Code Refactoring
- Split `proxy_server.py` into modules:
  - `token_manager.py`: Token fetching and caching
  - `load_balancer.py`: Load balancing logic
  - `request_handler.py`: Request processing
  - `models/`: Model-specific converters
- Add comprehensive type hints
- Add docstrings to all public functions

#### 2.3 Performance Optimization
- Add async support for I/O operations using `aiohttp`
- Implement request batching for embeddings
- Optimize streaming response handling

### Phase 3: Advanced Features (Higher Risk)

#### 3.1 Monitoring & Observability
- Add Prometheus metrics endpoint
- Request tracing with correlation IDs
- Detailed latency tracking per subaccount

#### 3.2 Advanced Load Balancing
- Latency-based routing
- Quota-aware routing (track API limits)
- Predictive scaling based on traffic patterns

#### 3.3 Testing
- Unit tests for core logic
- Integration tests with mock SAP AI Core
- Load testing suite

## Implementation Priority

1. **Immediate** (This Branch):
   - Improved load balancing with error rate tracking
   - Proactive token refresh
   - Circuit breaker for subaccounts
   - Better error messages and logging

2. **Next Iteration**:
   - Per-subaccount HTTP sessions
   - Code refactoring into modules
   - Type hints and documentation

3. **Future**:
   - Async support
   - Monitoring dashboard
   - Advanced load balancing algorithms

## Success Metrics

- **Reduced Error Rate**: Target < 0.1% request failures
- **Improved Latency**: P95 latency < 2 seconds
- **Better Availability**: 99.9% uptime
- **Code Quality**: > 80% test coverage

## Risk Mitigation

- All changes are backward compatible
- Feature flags for new load balancing logic
- Gradual rollout with monitoring
- Easy rollback to previous version

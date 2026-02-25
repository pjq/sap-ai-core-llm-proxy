"""
Enhanced Load Balancer for SAP AI Core LLM Proxy

This module provides improved load balancing with:
- Weighted round-robin support
- Error rate tracking
- Circuit breaker pattern
- Health status monitoring
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SubAccountStats:
    """Statistics for a subaccount"""
    total_requests: int = 0
    failed_requests: int = 0
    successful_requests: int = 0
    last_error_time: float = 0
    last_success_time: float = 0
    consecutive_failures: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate over recent requests"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def is_healthy(self) -> bool:
        """Check if subaccount is healthy based on error rate and consecutive failures"""
        # Consider unhealthy if:
        # - More than 5 consecutive failures
        # - Error rate > 50% with at least 10 requests
        if self.consecutive_failures >= 5:
            return False
        if self.total_requests >= 10 and self.error_rate > 0.5:
            return False
        return True


@dataclass
class CircuitBreaker:
    """Circuit breaker for subaccount protection"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_requests: int = 3
    
    state: str = "closed"  # closed, open, half-open
    failure_count: int = 0
    last_failure_time: float = 0
    half_open_successes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_success(self):
        """Record a successful request"""
        with self.lock:
            if self.state == "half-open":
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_requests:
                    logging.info("Circuit breaker closed after successful recovery")
                    self.state = "closed"
                    self.failure_count = 0
                    self.half_open_successes = 0
            elif self.state == "closed":
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record a failed request"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == "half-open":
                logging.warning("Circuit breaker opened during half-open state")
                self.state = "open"
                self.half_open_successes = 0
            elif self.state == "closed" and self.failure_count >= self.failure_threshold:
                logging.warning(f"Circuit breaker opened after {self.failure_count} failures")
                self.state = "open"
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        with self.lock:
            if self.state == "closed":
                return True
            
            if self.state == "open":
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    logging.info("Circuit breaker entering half-open state")
                    self.state = "half-open"
                    self.half_open_successes = 0
                    return True
                return False
            
            # half-open state
            return True
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        with self.lock:
            return self.state


class EnhancedLoadBalancer:
    """Enhanced load balancer with health tracking and circuit breakers"""
    
    def __init__(self, proxy_config):
        self.proxy_config = proxy_config
        self.stats: Dict[str, SubAccountStats] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.counters: Dict[str, int] = {}
        self.lock = threading.Lock()
        
        # Initialize stats and circuit breakers for all subaccounts
        for subaccount_name in proxy_config.subaccounts.keys():
            self.stats[subaccount_name] = SubAccountStats()
            self.circuit_breakers[subaccount_name] = CircuitBreaker()
        
        logging.info(f"EnhancedLoadBalancer initialized with {len(self.stats)} subaccounts")
    
    def get_healthy_subaccounts(self, model_name: str) -> List[str]:
        """Get list of healthy subaccounts that have the model"""
        if model_name not in self.proxy_config.model_to_subaccounts:
            return []
        
        all_subaccounts = self.proxy_config.model_to_subaccounts[model_name]
        healthy = []
        
        for subaccount_name in all_subaccounts:
            circuit_breaker = self.circuit_breakers.get(subaccount_name)
            stats = self.stats.get(subaccount_name)
            
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                logging.debug(f"SubAccount '{subaccount_name}' circuit breaker is open")
                continue
            
            # Check health stats
            if stats and not stats.is_healthy:
                logging.debug(f"SubAccount '{subaccount_name}' is unhealthy (error_rate={stats.error_rate:.2f}, consecutive_failures={stats.consecutive_failures})")
                continue
            
            healthy.append(subaccount_name)
        
        return healthy
    
    def select_subaccount(self, model_name: str) -> Optional[str]:
        """Select a subaccount using weighted round-robin with health checks"""
        healthy_subaccounts = self.get_healthy_subaccounts(model_name)
        
        if not healthy_subaccounts:
            logging.error(f"No healthy subaccounts available for model '{model_name}'")
            return None
        
        # Simple round-robin among healthy subaccounts
        with self.lock:
            if model_name not in self.counters:
                self.counters[model_name] = 0
            
            counter_key = f"lb:{model_name}"
            if counter_key not in self.counters:
                self.counters[counter_key] = 0
            
            # Select based on counter
            index = self.counters[counter_key] % len(healthy_subaccounts)
            selected = healthy_subaccounts[index]
            
            # Increment counters
            self.counters[model_name] += 1
            self.counters[counter_key] += 1
        
        logging.info(f"LoadBalancer selected subaccount '{selected}' for model '{model_name}' (healthy: {len(healthy_subaccounts)}/{len(self.proxy_config.model_to_subaccounts.get(model_name, []))})")
        return selected
    
    def select_url(self, subaccount_name: str, model_name: str) -> Optional[str]:
        """Select a URL from the subaccount's model list"""
        subaccount = self.proxy_config.subaccounts.get(subaccount_name)
        if not subaccount:
            return None
        
        url_list = subaccount.normalized_models.get(model_name, [])
        if not url_list:
            return None
        
        with self.lock:
            counter_key = f"url:{subaccount_name}:{model_name}"
            if counter_key not in self.counters:
                self.counters[counter_key] = 0
            
            index = self.counters[counter_key] % len(url_list)
            selected_url = url_list[index]
            
            self.counters[counter_key] += 1
        
        return selected_url
    
    def record_success(self, subaccount_name: str, response_time_ms: float = 0):
        """Record a successful request"""
        if subaccount_name not in self.stats:
            return
        
        stats = self.stats[subaccount_name]
        with self.lock:
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.consecutive_failures = 0
            stats.last_success_time = time.time()
            if response_time_ms > 0:
                stats.response_times.append(response_time_ms)
        
        # Update circuit breaker
        if subaccount_name in self.circuit_breakers:
            self.circuit_breakers[subaccount_name].record_success()
    
    def record_failure(self, subaccount_name: str):
        """Record a failed request"""
        if subaccount_name not in self.stats:
            return
        
        stats = self.stats[subaccount_name]
        with self.lock:
            stats.total_requests += 1
            stats.failed_requests += 1
            stats.consecutive_failures += 1
            stats.last_error_time = time.time()
        
        # Update circuit breaker
        if subaccount_name in self.circuit_breakers:
            self.circuit_breakers[subaccount_name].record_failure()
    
    def get_stats(self, subaccount_name: str) -> Optional[Dict]:
        """Get statistics for a subaccount"""
        if subaccount_name not in self.stats:
            return None
        
        stats = self.stats[subaccount_name]
        circuit_breaker = self.circuit_breakers.get(subaccount_name)
        
        return {
            "subaccount": subaccount_name,
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "error_rate": stats.error_rate,
            "avg_response_time_ms": stats.avg_response_time,
            "consecutive_failures": stats.consecutive_failures,
            "is_healthy": stats.is_healthy,
            "circuit_breaker_state": circuit_breaker.get_state() if circuit_breaker else "unknown"
        }
    
    def get_all_stats(self) -> List[Dict]:
        """Get statistics for all subaccounts"""
        return [self.get_stats(name) for name in self.stats.keys() if self.get_stats(name)]
    
    def reset_stats(self, subaccount_name: Optional[str] = None):
        """Reset statistics (optionally for a specific subaccount)"""
        if subaccount_name:
            if subaccount_name in self.stats:
                with self.lock:
                    self.stats[subaccount_name] = SubAccountStats()
            if subaccount_name in self.circuit_breakers:
                self.circuit_breakers[subaccount_name] = CircuitBreaker()
        else:
            # Reset all
            for name in self.stats.keys():
                self.stats[name] = SubAccountStats()
                self.circuit_breakers[name] = CircuitBreaker()
            with self.lock:
                self.counters.clear()


# Global load balancer instance (will be initialized after proxy_config)
_load_balancer: Optional[EnhancedLoadBalancer] = None


def get_load_balancer(proxy_config=None):
    """Get or create the global load balancer instance"""
    global _load_balancer
    if _load_balancer is None and proxy_config:
        _load_balancer = EnhancedLoadBalancer(proxy_config)
    return _load_balancer


def reset_load_balancer():
    """Reset the global load balancer (for testing/reconfiguration)"""
    global _load_balancer
    _load_balancer = None

"""
Enhanced Token Manager for SAP AI Core LLM Proxy

This module provides improved token management with:
- Proactive token refresh (before expiry)
- Token refresh queue to prevent race conditions
- Retry logic with exponential backoff
- Better error handling
"""

import time
import threading
import logging
import base64
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class TokenInfo:
    """Token information with metadata"""
    token: Optional[str] = None
    expiry: float = 0
    refresh_time: float = 0  # When to proactively refresh
    lock: threading.Lock = field(default_factory=threading.Lock)
    refresh_in_progress: bool = False
    last_refresh_attempt: float = 0
    consecutive_failures: int = 0
    
    def is_valid(self) -> bool:
        """Check if token is still valid"""
        return self.token is not None and time.time() < self.expiry
    
    def needs_refresh(self) -> bool:
        """Check if token needs proactive refresh"""
        return time.time() >= self.refresh_time and not self.refresh_in_progress
    
    def should_retry(self) -> bool:
        """Check if we should retry token refresh after failure"""
        if self.consecutive_failures == 0:
            return True
        
        # Exponential backoff: 2^failures seconds, max 5 minutes
        backoff = min(2 ** self.consecutive_failures, 300)
        return time.time() - self.last_refresh_attempt >= backoff


class EnhancedTokenManager:
    """Enhanced token manager with proactive refresh and retry logic"""
    
    def __init__(self, http_session: requests.Session = None):
        self.tokens: Dict[str, TokenInfo] = {}
        self.lock = threading.Lock()
        self.http_session = http_session or self._create_session()
        self.refresh_thread: Optional[threading.Thread] = None
        self.stop_refresh_thread = threading.Event()
        
        logging.info("EnhancedTokenManager initialized")
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def start_background_refresh(self, interval: int = 30):
        """Start background thread for proactive token refresh"""
        if self.refresh_thread and self.refresh_thread.is_alive():
            return
        
        self.stop_refresh_thread.clear()
        self.refresh_thread = threading.Thread(
            target=self._background_refresh_loop,
            args=(interval,),
            daemon=True
        )
        self.refresh_thread.start()
        logging.info(f"Started background token refresh thread (interval: {interval}s)")
    
    def stop_background_refresh(self):
        """Stop background refresh thread"""
        self.stop_refresh_thread.set()
        if self.refresh_thread:
            self.refresh_thread.join(timeout=5)
            logging.info("Stopped background token refresh thread")
    
    def _background_refresh_loop(self, interval: int):
        """Background loop to check and refresh tokens"""
        while not self.stop_refresh_thread.is_set():
            try:
                self._check_and_refresh_all_tokens()
            except Exception as e:
                logging.error(f"Error in background token refresh: {e}")
            
            self.stop_refresh_thread.wait(interval)
    
    def _check_and_refresh_all_tokens(self):
        """Check all tokens and refresh those that need it"""
        with self.lock:
            tokens_to_refresh = [
                (name, info) for name, info in self.tokens.items()
                if info.needs_refresh() and info.should_retry()
            ]
        
        for subaccount_name, token_info in tokens_to_refresh:
            logging.info(f"Proactively refreshing token for subaccount '{subaccount_name}'")
            try:
                self.refresh_token(subaccount_name)
            except Exception as e:
                logging.error(f"Failed to refresh token for '{subaccount_name}': {e}")
    
    def register_subaccount(self, subaccount_name: str, service_key):
        """Register a subaccount for token management"""
        with self.lock:
            if subaccount_name not in self.tokens:
                self.tokens[subaccount_name] = TokenInfo()
            
            # Store service key reference (will be used during refresh)
            token_info = self.tokens[subaccount_name]
            token_info.service_key = service_key
    
    def get_token(self, subaccount_name: str) -> Optional[str]:
        """Get token for subaccount, refreshing if necessary"""
        with self.lock:
            if subaccount_name not in self.tokens:
                logging.error(f"Token not registered for subaccount '{subaccount_name}'")
                return None
            
            token_info = self.tokens[subaccount_name]
        
        # Check if we have a valid token
        if token_info.is_valid():
            logging.debug(f"Using cached token for subaccount '{subaccount_name}'")
            return token_info.token
        
        # Token is expired or missing, need to refresh
        logging.info(f"Token expired/missing for subaccount '{subaccount_name}', refreshing...")
        return self.refresh_token(subaccount_name)
    
    def refresh_token(self, subaccount_name: str) -> Optional[str]:
        """Refresh token for subaccount with retry logic"""
        with self.lock:
            if subaccount_name not in self.tokens:
                raise ValueError(f"Subaccount '{subaccount_name}' not registered")
            
            token_info = self.tokens[subaccount_name]
            
            # Check if refresh is already in progress
            if token_info.refresh_in_progress:
                logging.debug(f"Token refresh already in progress for '{subaccount_name}'")
                # Wait for refresh to complete (with timeout)
                token_info.lock.wait(timeout=10)
                return token_info.token
            
            # Mark refresh as in progress
            token_info.refresh_in_progress = True
            service_key = getattr(token_info, 'service_key', None)
        
        if not service_key:
            logging.error(f"No service key available for subaccount '{subaccount_name}'")
            with self.lock:
                token_info.refresh_in_progress = False
                token_info.lock.notify_all()
            return None
        
        try:
            # Build auth header
            auth_string = f"{service_key.clientid}:{service_key.clientsecret}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            
            token_url = f"{service_key.url}/oauth/token?grant_type=client_credentials"
            headers = {"Authorization": f"Basic {encoded_auth}"}
            
            logging.info(f"Fetching new token for subaccount '{subaccount_name}' from {service_key.url}")
            
            response = self.http_session.post(token_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            token_data = response.json()
            new_token = token_data.get('access_token')
            
            if not new_token:
                raise ValueError("Fetched token is empty")
            
            # Calculate expiry (with buffer for proactive refresh)
            expires_in = int(token_data.get('expires_in', 14400))
            now = time.time()
            
            # Set refresh time to 80% of expiry (proactive refresh)
            refresh_buffer = int(expires_in * 0.2)
            actual_expiry = now + expires_in - 300  # 5-minute safety buffer
            refresh_time = now + expires_in - refresh_buffer
            
            with self.lock:
                token_info.token = new_token
                token_info.expiry = actual_expiry
                token_info.refresh_time = refresh_time
                token_info.refresh_in_progress = False
                token_info.consecutive_failures = 0
                token_info.last_refresh_attempt = now
                token_info.lock.notify_all()
            
            logging.info(f"Token refreshed successfully for '{subaccount_name}' (expires in {expires_in}s, will refresh at {refresh_time - now:.0f}s)")
            return new_token
            
        except requests.exceptions.Timeout as err:
            logging.error(f"Timeout fetching token for '{subaccount_name}': {err}")
            self._handle_refresh_failure(subaccount_name, err)
            raise TimeoutError(f"Timeout fetching token for '{subaccount_name}'") from err
            
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP error fetching token for '{subaccount_name}': {err.response.status_code}")
            self._handle_refresh_failure(subaccount_name, err)
            raise ConnectionError(f"HTTP {err.response.status_code} fetching token for '{subaccount_name}'") from err
            
        except Exception as err:
            logging.error(f"Unexpected error fetching token for '{subaccount_name}': {err}", exc_info=True)
            self._handle_refresh_failure(subaccount_name, err)
            raise RuntimeError(f"Error fetching token for '{subaccount_name}': {err}") from err
        
        finally:
            with self.lock:
                token_info.refresh_in_progress = False
                token_info.lock.notify_all()
    
    def _handle_refresh_failure(self, subaccount_name: str, error: Exception):
        """Handle token refresh failure"""
        with self.lock:
            if subaccount_name in self.tokens:
                token_info = self.tokens[subaccount_name]
                token_info.consecutive_failures += 1
                token_info.last_refresh_attempt = time.time()
                token_info.refresh_in_progress = False
                token_info.lock.notify_all()
                
                logging.warning(f"Token refresh failed for '{subaccount_name}' (attempt {token_info.consecutive_failures})")
    
    def get_token_info(self, subaccount_name: str) -> Optional[Dict]:
        """Get token information for monitoring"""
        with self.lock:
            if subaccount_name not in self.tokens:
                return None
            
            token_info = self.tokens[subaccount_name]
            now = time.time()
            
            return {
                "subaccount": subaccount_name,
                "has_token": token_info.token is not None,
                "expires_in_seconds": max(0, token_info.expiry - now),
                "refresh_in_seconds": max(0, token_info.refresh_time - now),
                "needs_refresh": token_info.needs_refresh(),
                "consecutive_failures": token_info.consecutive_failures,
                "refresh_in_progress": token_info.refresh_in_progress
            }
    
    def get_all_token_info(self) -> List[Dict]:
        """Get token information for all subaccounts"""
        return [self.get_token_info(name) for name in self.tokens.keys() if self.get_token_info(name)]
    
    def invalidate_token(self, subaccount_name: str):
        """Invalidate cached token (force refresh on next use)"""
        with self.lock:
            if subaccount_name in self.tokens:
                self.tokens[subaccount_name].token = None
                self.tokens[subaccount_name].expiry = 0
                logging.info(f"Invalidated token for subaccount '{subaccount_name}'")
    
    def close(self):
        """Cleanup resources"""
        self.stop_background_refresh()
        self.http_session.close()


# Global token manager instance
_token_manager: Optional[EnhancedTokenManager] = None


def get_token_manager(http_session=None) -> EnhancedTokenManager:
    """Get or create the global token manager instance"""
    global _token_manager
    if _token_manager is None:
        _token_manager = EnhancedTokenManager(http_session)
        _token_manager.start_background_refresh(interval=30)
    return _token_manager


def reset_token_manager():
    """Reset the global token manager (for testing)"""
    global _token_manager
    if _token_manager:
        _token_manager.close()
    _token_manager = None

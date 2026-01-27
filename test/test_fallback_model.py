#!/usr/bin/env python3
"""
Test script to verify that the load_balance_url function correctly returns the fallback model name
"""

import sys
sys.path.append('.')

from proxy_server import load_balance_url, proxy_config, ProxyConfig, SubAccountConfig

def test_fallback_model():
    """Test that load_balance_url returns the correct fallback model name"""
    
    # Mock configuration for testing
    proxy_config.subaccounts = {
        "test_subaccount": SubAccountConfig(
            name="test_subaccount",
            resource_group="default",
            service_key_json="test_key.json",
            deployment_models={"4-sonnet": ["http://test-url.com"]},
            normalized_models={"4-sonnet": ["http://test-url.com"]}
        )
    }
    
    proxy_config.model_to_subaccounts = {
        "4-sonnet": ["test_subaccount"]
    }
    
    print("Testing load_balance_url with fallback functionality...")
    
    # Test 1: Request a model that doesn't exist but has a fallback
    print("\nTest 1: Requesting 'claude-3-5-haiku-20241022' (should fallback to '4-sonnet')")
    try:
        url, subaccount, resource_group, final_model = load_balance_url("claude-3-5-haiku-20241022")
        print(f"✅ Success!")
        print(f"  - Original model: claude-3-5-haiku-20241022")
        print(f"  - Final model: {final_model}")
        print(f"  - URL: {url}")
        print(f"  - SubAccount: {subaccount}")
        print(f"  - Resource Group: {resource_group}")
        
        if final_model == "4-sonnet":
            print("✅ Fallback model correctly returned!")
        else:
            print(f"❌ Expected fallback model '4-sonnet', got '{final_model}'")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Request a model that exists (no fallback needed)
    print("\nTest 2: Requesting '4-sonnet' (should use original model)")
    try:
        url, subaccount, resource_group, final_model = load_balance_url("4-sonnet")
        print(f"✅ Success!")
        print(f"  - Original model: 4-sonnet")
        print(f"  - Final model: {final_model}")
        print(f"  - URL: {url}")
        print(f"  - SubAccount: {subaccount}")
        print(f"  - Resource Group: {resource_group}")
        
        if final_model == "4-sonnet":
            print("✅ Original model correctly returned!")
        else:
            print(f"❌ Expected original model '4-sonnet', got '{final_model}'")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Request a model that doesn't exist and has no fallback
    print("\nTest 3: Requesting 'non-existent-model' (should fail)")
    try:
        url, subaccount, resource_group, final_model = load_balance_url("non-existent-model")
        print(f"❌ Should have failed but got: {final_model}")
    except Exception as e:
        print(f"✅ Correctly failed: {e}")

if __name__ == "__main__":
    test_fallback_model()

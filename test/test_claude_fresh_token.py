#!/usr/bin/env python3
"""
Test script that fetches a fresh token and tests the Claude endpoint
This script uses the SFMobileSH service key to get a fresh token
"""

import json
import requests
import base64
import time

def load_service_key():
    """Load service key from SFMobileSH_defaultKey.json"""
    try:
        with open('SFMobileSH_defaultKey.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ SFMobileSH_defaultKey.json not found!")
        print("Available service key files:")
        import os
        for file in os.listdir('.'):
            if file.endswith('_service_key.json') or file.endswith('defaultKey.json'):
                print(f"  - {file}")
        return None

def fetch_fresh_token(service_key):
    """Fetch a fresh token using the service key"""
    try:
        auth_string = f"{service_key['clientid']}:{service_key['clientsecret']}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        token_url = f"{service_key['url']}/oauth/token?grant_type=client_credentials"
        headers = {"Authorization": f"Basic {encoded_auth}"}
        
        print(f"Fetching token from: {token_url}")
        response = requests.post(token_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        token_data = response.json()
        access_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in', 3600)
        
        print(f"✅ Token fetched successfully! Expires in {expires_in} seconds")
        return access_token, service_key['identityzoneid']
        
    except Exception as e:
        print(f"❌ Failed to fetch token: {e}")
        return None, None

def test_claude_endpoints(token, tenant_id):
    """Test both converse and converse-stream endpoints"""
    base_url = "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/daa9d6476f84ef9e"
    
    headers = {
        "AI-Resource-Group": "default",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "AI-Tenant-Id": tenant_id
    }
    
    # Test payload - start with minimal
    test_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Hello! Please respond with exactly: 'Hi there, this is Claude on SAP AI Core!'"
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 100,
            "temperature": 0.0
        }
    }
    
    # Test 1: Non-streaming endpoint
    print("\\n" + "="*60)
    print("TEST 1: /converse (non-streaming)")
    print("="*60)
    
    try:
        url = f"{base_url}/converse"
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(test_payload, indent=2)}")
        
        response = requests.post(url, headers=headers, json=test_payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS!")
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print("❌ FAILED!")
            try:
                error = response.json()
                print(f"Error: {json.dumps(error, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Streaming endpoint
    print("\\n" + "="*60)
    print("TEST 2: /converse-stream (streaming)")
    print("="*60)
    
    try:
        url = f"{base_url}/converse-stream"
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(test_payload, indent=2)}")
        
        response = requests.post(url, headers=headers, json=test_payload, stream=True, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS!")
            print("Stream chunks:")
            chunk_count = 0
            for chunk in response.iter_lines(decode_unicode=True):
                if chunk:
                    chunk_count += 1
                    print(f"  Chunk {chunk_count}: {chunk}")
                    if chunk_count >= 10:  # Limit output
                        print("  ... (stopping after 10 chunks)")
                        break
        else:
            print("❌ FAILED!")
            try:
                error = response.json()
                print(f"Error: {json.dumps(error, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 3: Test with the original failing payload format
    print("\\n" + "="*60)
    print("TEST 3: Original failing payload format")
    print("="*60)
    
    # This is the format that was failing from the logs
    failing_payload = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 512,
        "temperature": 0,
        "system": [
            {
                "type": "text",
                "text": "You are an expert at analyzing git history."
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "Test message"
            }
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }
    
    try:
        url = f"{base_url}/converse-stream"
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(failing_payload, indent=2)}")
        
        # Add the original headers that were failing
        failing_headers = headers.copy()
        failing_headers.update({
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "fine-grained-tool-streaming-2025-05-14"
        })
        
        response = requests.post(url, headers=failing_headers, json=failing_payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! (This shouldn't have failed before)")
        else:
            print("❌ EXPECTED FAILURE - Let's see the error:")
            try:
                error = response.json()
                print(f"Error: {json.dumps(error, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    """Main function"""
    print("SAP AI Core Claude Endpoint Test with Fresh Token")
    print("="*60)
    
    # Load service key
    service_key = load_service_key()
    if not service_key:
        return
    
    print(f"Service key loaded for: {service_key.get('identityzoneid', 'unknown')}")
    
    # Fetch fresh token
    token, tenant_id = fetch_fresh_token(service_key)
    if not token:
        return
    
    # Test endpoints
    test_claude_endpoints(token, tenant_id)
    
    print("\\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()

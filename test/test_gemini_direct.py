#!/usr/bin/env python3

import json
import requests
import base64

def get_token(service_key_file):
    """Get OAuth token from SAP AI Core"""
    with open(service_key_file, 'r') as f:
        service_key = json.load(f)
    
    # Create basic auth header
    auth_string = f"{service_key['clientid']}:{service_key['clientsecret']}"
    auth_header = base64.b64encode(auth_string.encode()).decode()
    
    # Get token
    token_url = f"{service_key['url']}/oauth/token?grant_type=client_credentials"
    headers = {
        'Authorization': f'Basic {auth_header}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    response = requests.post(token_url, headers=headers, data='')
    response.raise_for_status()
    
    token_data = response.json()
    return token_data['access_token'], service_key['identityzoneid']

def test_gemini_endpoint():
    """Test the Gemini endpoint directly"""
    try:
        # Get token using erica service key
        token, tenant_id = get_token('erica_service_key.json')
        print("✅ Token obtained successfully with erica service key")
        
        # Test endpoint
        endpoint = "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/dbb4244d6b73a887/models/gemini-2.5-pro:generateContent"
        
        payload = {
            "contents": {
                "role": "user",
                "parts": {"text": "Hello"}
            },
            "generation_config": {
                "maxOutputTokens": 100,
                "temperature": 0.7
            },
            "safety_settings": {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                "threshold": "BLOCK_LOW_AND_ABOVE"
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
            'AI-Resource-Group': 'default',
            'AI-Tenant-Id': tenant_id
        }
        
        print(f"🔗 Testing endpoint: {endpoint}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(endpoint, headers=headers, json=payload)
        
        print(f"📊 Status code: {response.status_code}")
        print(f"📝 Response headers: {dict(response.headers)}")
        print(f"💬 Response body: {response.text}")
        
        if response.status_code == 200:
            print("✅ Gemini request successful!")
        else:
            print(f"❌ Gemini request failed with {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_gemini_endpoint()
#!/usr/bin/env python3
"""
Simple test script for SAP AI Core Claude endpoint (non-streaming)
This tests the /converse endpoint instead of /converse-stream
"""

import json
import requests
import time

# Configuration from the error logs
TEST_CONFIG = {
    "base_url": "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/daa9d6476f84ef9e",
    "resource_group": "default",
    "tenant_id": "8c9ba495-98ae-4c52-8d2b-81129101519e",
    # Note: This token will likely be expired, you'll need to get a fresh one
    "token": "eyJhbGciOiJSUzI1NiIsImprdSI6Imh0dHBzOi8vc2Ztb2JpbGVzaC5hdXRoZW50aWNhdGlvbi5ldTEyLmhhbmEub25kZW1hbmQuY29tL3Rva2VuX2tleXMiLCJraWQiOiJkZWZhdWx0LWp3dC1rZXktNzQ5OGM0N2Y5MyIsInR5cCI6IkpXVCIsImppZCI6ICJCc0R3UndOVHR4U2xGcklIWllXMVBEZTU2TUNmNFh4NGhyL0R4V2xWNFJVPSJ9.eyJqdGkiOiJjOGRjMGZkN2VkZmE0ZmJiOGUxMDQ2YThhYWUwYTNjNyIsImV4dF9hdHRyIjp7ImVuaGFuY2VyIjoiWFNVQUEiLCJzdWJhY2NvdW50aWQiOiI4YzliYTQ5NS05OGFlLTRjNTItOGQyYi04MTEyOTEwMTUxOWUiLCJ6ZG4iOiJzZm1vYmlsZXNoIiwic2VydmljZWluc3RhbmNlaWQiOiI5MWJkMDc0NC00NDg5LTQwNzgtOGYxZS00YjNhOWFkMGM2ZWUifSwic3ViIjoic2ItOTFiZDA3NDQtNDQ4OS00MDc4LThmMWUtNGIzYTlhZDBjNmVlIWIxMDc3NDg2fHhzdWFhX3N0ZCFiMzE4MDYxIiwiYXV0aG9yaXRpZXMiOlsieHN1YWFfc3RkIWIzMTgwNjEubG9ncy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuZG9ja2VycmVnaXN0cnlzZWNyZXQuY3JlZGVudGlhbHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuYXJ0aWZhY3RzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0aW9uc2NoZWR1bGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5leGVjdXRpb25zLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlcG9zaXRvcmllcy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLmFwcGxpY2F0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNlY3JldHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0YWJsZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLmRhdGFzZXRzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEubWV0YS5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEucmVwb3NpdG9yaWVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MucHJvbXB0VGVtcGxhdGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5kb2NrZXJyZWdpc3RyeXNlY3JldC5jcmVkZW50aWFscy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEubm9kZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5hcnRpZmFjdHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5hcHBsaWNhdGlvbnMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VjcmV0cy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmRlcGxveW1lbnRzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5ub2Rlcy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VydmljZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuY29uZmlndXJhdGlvbnMucmVhZCIsInVhYS5yZXNvdXJjZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5jb25maWd1cmF0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLmRlcGxveW1lbnRzLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5kZXBsb3ltZW50cy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zY2hlZHVsZXMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5kYXRhc2V0cy5kb3dubG9hZCIsInhzdWFhX3N0ZCFiMzE4MDYxLmtwaXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZGVwbG95bWVudHMucHJlZGljdCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5tZXRyaWNzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLm1ldHJpY3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5wcm9tcHRUZW1wbGF0ZXMud3JpdGUiXSwic2NvcGUiOlsieHN1YWFfc3RkIWIzMTgwNjEubG9ncy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuZG9ja2VycmVnaXN0cnlzZWNyZXQuY3JlZGVudGlhbHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuYXJ0aWZhY3RzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0aW9uc2NoZWR1bGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5leGVjdXRpb25zLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlcG9zaXRvcmllcy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLmFwcGxpY2F0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNlY3JldHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0YWJsZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLmRhdGFzZXRzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEubWV0YS5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEucmVwb3NpdG9yaWVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MucHJvbXB0VGVtcGxhdGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5kb2NrZXJyZWdpc3RyeXNlY3JldC5jcmVkZW50aWFscy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEubm9kZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5hcnRpZmFjdHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5hcHBsaWNhdGlvbnMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VjcmV0cy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmRlcGxveW1lbnRzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5ub2Rlcy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VydmljZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuY29uZmlndXJhdGlvbnMucmVhZCIsInVhYS5yZXNvdXJjZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5jb25maWd1cmF0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLmRlcGxveW1lbnRzLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5kZXBsb3ltZW50cy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zY2hlZHVsZXMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5kYXRhc2V0cy5kb3dubG9hZCIsInhzdWFhX3N0ZCFiMzE4MDYxLmtwaXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZGVwbG95bWVudHMucHJlZGljdCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5tZXRyaWNzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLm1ldHJpY3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5wcm9tcHRUZW1wbGF0ZXMud3JpdGUiXSwiY2xpZW50X2lkIjoic2ItOTFiZDA3NDQtNDQ4OS00MDc4LThmMWUtNGIzYTlhZDBjNmVlIWIxMDc3NDg2fHhzdWFhX3N0ZCFiMzE4MDYxIiwiY2lkIjoic2ItOTFiZDA3NDQtNDQ4OS00MDc4LThmMWUtNGIzYTlhZDBjNmVlIWIxMDc3NDg2fHhzdWFhX3N0ZCFiMzE4MDYxIiwiYXpwIjoic2ItOTFiZDA3NDQtNDQ4OS00MDc4LThmMWUtNGIzYTlhZDBjNmVlIWIxMDc3NDg2fHhzdWFhX3N0ZCFiMzE4MDYxIiwiZ3JhbnRfdHlwZSI6ImNsaWVudF9jcmVkZW50aWFscyIsInJldl9zaWciOiI4ZDg3ODMyZCIsImlhdCI6MTc1NTIzNTM2OSwiZXhwIjoxNzU1Mjc4NTY5LCJpc3MiOiJodHRwczovL3NmbW9iaWxlc2guYXV0aGVudGljYXRpb24uZXUxMi5oYW5hLm9uZGVtYW5kLmNvbS9vYXV0aC90b2tlbiIsInppZCI6IjhjOWJhNDk1LTk4YWUtNGM1Mi04ZDJiLTgxMTI5MTAxNTE5ZSIsImF1ZCI6WyJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuY29uZmlndXJhdGlvbnMiLCJ4c3VhYV9zdGQhYjMxODA2MS5rcGlzIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLnByb21wdFRlbXBsYXRlcyIsInNiLTkxYmQwNzQ0LTQ0ODktNDA3OC04ZjFlLTRiM2E5YWQwYzZlZSFiMTA3NzQ4Nnx4c3VhYV9zdGQhYjMxODA2MSIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzIiwieHN1YWFfc3RkIWIzMTgwNjEubWV0YSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlcG9zaXRvcmllcyIsInhzdWFhX3N0ZCFiMzE4MDYxLmRhdGFzZXRzIiwidWFhIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmRlcGxveW1lbnRzIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmV4ZWN1dGlvbnMiLCJ4c3VhYV9zdGQhYjMxODA2MS5zZWNyZXRzIiwieHN1YWFfc3RkIWIzMTgwNjEuZGVwbG95bWVudHMubG9ncyIsInhzdWFhX3N0ZCFiMzE4MDYxLmxvZ3MiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MiLCJ4c3VhYWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmV4ZWN1dGFibGVzIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VydmljZXMiLCJ4c3VhYV9zdGQhYjMxODA2MS5ub2RlcyIsInhzdWFhX3N0ZCFiMzE4MDYxLmFwcGxpY2F0aW9ucyIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zY2hlZHVsZXMiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MubWV0cmljcyIsInhzdWFhX3N0ZCFiMzE4MDYxLmRvY2tlcnJlZ2lzdHJ5c2VjcmV0LmNyZWRlbnRpYWxzIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmFydGlmYWN0cyIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAiLCJ4c3VhYV9zdGQhYjMxODA2MS5leGVjdXRpb25zLmxvZ3MiXX0.Yon6zWEc7iPjkXVcl364go6KTeSKDta0ZvYIFjBQN7MNorGdFSafgWLkNIGOdDxWtwHqZ5THShARS2OYpbrHapzTAUr-1FwQeSH97g1zu-0CHkbIjRoFzt8alAMhHAikYGnCFX5ouPhZj8u3jv6kBQF6EGLuybISTiNiL3tlQO5aHcONbbrxj34jxB-CDwcFt7YNF5kUTNJSYI5uNuyhgz39iZutwFYbgSz7sj61rufaZqV0DZBnglVaFHRbB1YiMtqzAK-46m85puwfwA8GqoXkNDyYdE-y5ovOxwIzn2KmJpKNuZG6Eh4eAOHSl9A16AJbtoEhwaXAaBTCq_COCA"
}

def test_converse_endpoint():
    """Test the non-streaming /converse endpoint"""
    url = f"{TEST_CONFIG['base_url']}/converse"
    
    headers = {
        "AI-Resource-Group": TEST_CONFIG["resource_group"],
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEST_CONFIG['token']}",
        "AI-Tenant-Id": TEST_CONFIG["tenant_id"]
    }
    
    # Minimal valid payload for Claude Converse API
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Hello! Please respond with just 'Hi there!'"
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 50
        }
    }
    
    print("Testing /converse endpoint (non-streaming)")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_json = response.json()
            print("✅ SUCCESS!")
            print(f"Response: {json.dumps(response_json, indent=2)}")
        else:
            print("❌ FAILED!")
            try:
                error_json = response.json()
                print(f"Error: {json.dumps(error_json, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"Exception: {e}")

def test_converse_stream_endpoint():
    """Test the streaming /converse-stream endpoint"""
    url = f"{TEST_CONFIG['base_url']}/converse-stream"
    
    headers = {
        "AI-Resource-Group": TEST_CONFIG["resource_group"],
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEST_CONFIG['token']}",
        "AI-Tenant-Id": TEST_CONFIG["tenant_id"]
    }
    
    # Minimal valid payload for Claude Converse API
    payload = {
        "messages": [
            {
                "role": "user", 
                "content": [
                    {
                        "text": "Hello! Please respond with just 'Hi there!'"
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 50
        }
    }
    
    print("\nTesting /converse-stream endpoint (streaming)")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ SUCCESS!")
            print("Stream chunks:")
            for i, chunk in enumerate(response.iter_lines(decode_unicode=True)):
                if chunk and i < 10:  # Show first 10 chunks
                    print(f"  Chunk {i+1}: {chunk}")
                elif i >= 10:
                    print("  ... (showing first 10 chunks)")
                    break
        else:
            print("❌ FAILED!")
            try:
                error_json = response.json()
                print(f"Error: {json.dumps(error_json, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"Exception: {e}")

def test_with_original_failing_payload():
    """Test with the exact payload that failed"""
    url = f"{TEST_CONFIG['base_url']}/converse-stream"
    
    headers = {
        "AI-Resource-Group": TEST_CONFIG["resource_group"],
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {TEST_CONFIG['token']}",
        "AI-Tenant-Id": TEST_CONFIG["tenant_id"],
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "fine-grained-tool-streaming-2025-05-14"
    }
    
    # The exact failing payload from the logs
    payload = {
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
                "content": "Files modified by user:\n  46 \n  33 proxy_server.py"
            }
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }
    
    print("\nTesting with original failing payload")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ SUCCESS!")
        else:
            print("❌ FAILED!")
            try:
                error_json = response.json()
                print(f"Error: {json.dumps(error_json, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    print("SAP AI Core Claude Endpoint Test")
    print("=" * 50)
    
    # Test both endpoints
    test_converse_endpoint()
    test_converse_stream_endpoint() 
    test_with_original_failing_payload()
    
    print("\n" + "=" * 50)
    print("Test complete!")

#!/usr/bin/env python3
"""
Test script for SAP AI Core Claude endpoint
Based on the error logs from proxy_server.py
"""

import json
import requests
import time
from typing import Dict, Any

# Configuration from the error logs
TEST_CONFIG = {
    "url": "https://api.ai.intprod-eu12.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/daa9d6476f84ef9e/converse-stream",
    "resource_group": "default",
    "tenant_id": "8c9ba495-98ae-4c52-8d2b-81129101519e",
    # Note: Token would need to be refreshed in real usage
    "token": "eyJhbGciOiJSUzI1NiIsImprdSI6Imh0dHBzOi8vc2Ztb2JpbGVzaC5hdXRoZW50aWNhdGlvbi5ldTEyLmhhbmEub25kZW1hbmQuY29tL3Rva2VuX2tleXMiLCJraWQiOiJkZWZhdWx0LWp3dC1rZXktNzQ5OGM0N2Y5MyIsInR5cCI6IkpXVCIsImppZCI6ICJCc0R3UndOVHR4U2xGcklIWllXMVBEZTU2TUNmNFh4NGhyL0R4V2xWNFJVPSJ9.eyJqdGkiOiJjOGRjMGZkN2VkZmE0ZmJiOGUxMDQ2YThhYWUwYTNjNyIsImV4dF9hdHRyIjp7ImVuaGFuY2VyIjoiWFNVQUEiLCJzdWJhY2NvdW50aWQiOiI4YzliYTQ5NS05OGFlLTRjNTItOGQyYi04MTEyOTEwMTUxOWUiLCJ6ZG4iOiJzZm1vYmlsZXNoIiwic2VydmljZWluc3RhbmNlaWQiOiI5MWJkMDc0NC00NDg5LTQwNzgtOGYxZS00YjNhOWFkMGM2ZWUifSwic3ViIjoic2ItOTFiZDA3NDQtNDQ4OS00MDc4LThmMWUtNGIzYTlhZDBjNmVlIWIxMDc3NDg2fHhzdWFhX3N0ZCFiMzE4MDYxIiwiYXV0aG9yaXRpZXMiOlsieHN1YWFfc3RkIWIzMTgwNjEubG9ncy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuZG9ja2VycmVnaXN0cnlzZWNyZXQuY3JlZGVudGlhbHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuYXJ0aWZhY3RzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0aW9uc2NoZWR1bGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5leGVjdXRpb25zLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlcG9zaXRvcmllcy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLmFwcGxpY2F0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNlY3JldHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0YWJsZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLmRhdGFzZXRzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEubWV0YS5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEucmVwb3NpdG9yaWVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MucHJvbXB0VGVtcGxhdGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5kb2NrZXJyZWdpc3RyeXNlY3JldC5jcmVkZW50aWFscy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEubm9kZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5hcnRpZmFjdHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5hcHBsaWNhdGlvbnMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VjcmV0cy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmRlcGxveW1lbnRzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5ub2Rlcy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VydmljZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuY29uZmlndXJhdGlvbnMucmVhZCIsInVhYS5yZXNvdXJjZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5jb25maWd1cmF0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLmRlcGxveW1lbnRzLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5kZXBsb3ltZW50cy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zY2hlZHVsZXMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5kYXRhc2V0cy5kb3dubG9hZCIsInhzdWFhX3N0ZCFiMzE4MDYxLmtwaXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZGVwbG95bWVudHMucHJlZGljdCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5tZXRyaWNzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLm1ldHJpY3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5wcm9tcHRUZW1wbGF0ZXMud3JpdGUiXSwic2NvcGUiOlsieHN1YWFfc3RkIWIzMTgwNjEubG9ncy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuZG9ja2VycmVnaXN0cnlzZWNyZXQuY3JlZGVudGlhbHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuYXJ0aWZhY3RzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0aW9uc2NoZWR1bGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5leGVjdXRpb25zLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlcG9zaXRvcmllcy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLmFwcGxpY2F0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNlY3JldHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0YWJsZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLmRhdGFzZXRzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEubWV0YS5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEucmVwb3NpdG9yaWVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MucHJvbXB0VGVtcGxhdGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5kb2NrZXJyZWdpc3RyeXNlY3JldC5jcmVkZW50aWFscy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEubm9kZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5hcnRpZmFjdHMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5hcHBsaWNhdGlvbnMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VjcmV0cy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmRlcGxveW1lbnRzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5ub2Rlcy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5yZWFkIiwieHN1YWFfc3RkIWIzMTgwNjEuc2VydmljZXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuY29uZmlndXJhdGlvbnMucmVhZCIsInVhYS5yZXNvdXJjZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlc291cmNlZ3JvdXAucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5jb25maWd1cmF0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLmRlcGxveW1lbnRzLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5kZXBsb3ltZW50cy53cml0ZSIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5leGVjdXRpb25zY2hlZHVsZXMud3JpdGUiLCJ4c3VhYV9zdGQhYjMxODA2MS5kYXRhc2V0cy5kb3dubG9hZCIsInhzdWFhX3N0ZCFiMzE4MDYxLmtwaXMucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLnJlYWQiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZGVwbG95bWVudHMucHJlZGljdCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5tZXRyaWNzLndyaXRlIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLm1ldHJpY3MucmVhZCIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5wcm9tcHRUZW1wbGF0ZXMud3JpdGUiXSwiY2xpZW50X2lkIjoic2ItOTFiZDA3NDQtNDQ4OS00MDc4LThmMWUtNGIzYTlhZDBjNmVlIWIxMDc3NDg2fHhzdWFhX3N0ZCFiMzE4MDYxIiwiY2lkIjoic2ItOTFiZDA3NDQtNDQ4OS00MDc4LThmMWUtNGIzYTlhZDBjNmVlIWIxMDc3NDg2fHhzdWFhX3N0ZCFiMzE4MDYxIiwiYXpwIjoic2ItOTFiZDA3NDQtNDQ4OS00MDc4LThmMWUtNGIzYTlhZDBjNmVlIWIxMDc3NDg2fHhzdWFhX3N0ZCFiMzE4MDYxIiwiZ3JhbnRfdHlwZSI6ImNsaWVudF9jcmVkZW50aWFscyIsInJldl9zaWciOiI4ZDg3ODMyZCIsImlhdCI6MTc1NTIzNTM2OSwiZXhwIjoxNzU1Mjc4NTY5LCJpc3MiOiJodHRwczovL3NmbW9iaWxlc2guYXV0aGVudGljYXRpb24uZXUxMi5oYW5hLm9uZGVtYW5kLmNvbS9vYXV0aC90b2tlbiIsInppZCI6IjhjOWJhNDk1LTk4YWUtNGM1Mi04ZDJiLTgxMTI5MTAxNTE5ZSIsImF1ZCI6WyJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuY29uZmlndXJhdGlvbnMiLCJ4c3VhYV9zdGQhYjMxODA2MS5rcGlzIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLnByb21wdFRlbXBsYXRlcyIsInNiLTkxYmQwNzQ0LTQ0ODktNDA3OC04ZjFlLTRiM2E5YWQwYzZlZSFiMTA3NzQ4Nnx4c3VhYV9zdGQhYjMxODA2MSIsInhzdWFhX3N0ZCFiMzE4MDYxLm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzIiwieHN1YWFfc3RkIWIzMTgwNjEubWV0YSIsInhzdWFhX3N0ZCFiMzE4MDYxLnJlcG9zaXRvcmllcyIsInhzdWFhX3N0ZCFiMzE4MDYxLmRhdGFzZXRzIiwidWFhIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmRlcGxveW1lbnRzIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmV4ZWN1dGlvbnMiLCJ4c3VhYV9zdGQhYjMxODA2MS5zZWNyZXRzIiwieHN1YWFfc3RkIWIzMTgwNjEuZGVwbG95bWVudHMubG9ncyIsInhzdWFhX3N0ZCFiMzE4MDYxLmxvZ3MiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuZXhlY3V0YWJsZXMiLCJ4c3VhYV9zdGQhYjMxODA2MS5zZXJ2aWNlcyIsInhzdWFhX3N0ZCFiMzE4MDYxLm5vZGVzIiwieHN1YWFfc3RkIWIzMTgwNjEuYXBwbGljYXRpb25zIiwieHN1YWFfc3RkIWIzMTgwNjEuc2NlbmFyaW9zLmV4ZWN1dGlvbnNjaGVkdWxlcyIsInhzdWFhX3N0ZCFiMzE4MDYxLnNjZW5hcmlvcy5tZXRyaWNzIiwieHN1YWFfc3RkIWIzMTgwNjEuZG9ja2VycmVnaXN0cnlzZWNyZXQuY3JlZGVudGlhbHMiLCJ4c3VhYV9zdGQhYjMxODA2MS5zY2VuYXJpb3MuYXJ0aWZhY3RzIiwieHN1YWFfc3RkIWIzMTgwNjEucmVzb3VyY2Vncm91cCIsInhzdWFhX3N0ZCFiMzE4MDYxLmV4ZWN1dGlvbnMubG9ncyJdfQ.Yon6zWEc7iPjkXVcl364go6KTeSKDta0ZvYIFjBQN7MNorGdFSafgWLkNIGOdDxWtwHqZ5THShARS2OYpbrHapzTAUr-1FwQeSH97g1zu-0CHkbIjRoFzt8alAMhHAikYGnCFX5ouPhZj8u3jv6kBQF6EGLuybISTiNiL3tlQO5aHcONbbrxj34jxB-CDwcFt7YNF5kUTNJSYI5uNuyhgz39iZutwFYbgSz7sj61rufaZqV0DZBnglVaFHRbB1YiMtqzAK-46m85puwfwA8GqoXkNDyYdE-y5ovOxwIzn2KmJpKNuZG6Eh4eAOHSl9A16AJbtoEhwaXAaBTCq_COCA"
}

def create_failing_payload():
    """Create the exact payload that failed in the logs"""
    return {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 512,
        "temperature": 0,
        "system": [
            {
                "type": "text",
                "text": "You are an expert at analyzing git history. Given a list of files and their modification counts, return exactly five filenames that are frequently modified and represent core application logic (not auto-generated files, dependencies, or configuration). Make sure filenames are diverse, not all in the same folder, and are a mix of user and other users. Return only the filenames' basenames (without the path) separated by newlines with no explanation."
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "Files modified by user:\n  46 \n  33 proxy_server.py\n  15 README.md\n   7 proxy_server_demo_request.py\n   7 config.json.example\n   3 proxy_server_litellm.py\n   2 requirements.txt\n   2 .gitignore\n   1 chat.py\n   1 .github/workflows/ai-code-review.yml\n   1 .github/scripts/ai_review.py\n"
            }
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }

def create_corrected_payload_v1():
    """Create a corrected payload based on Claude Converse API spec"""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert at analyzing git history. Given a list of files and their modification counts, return exactly five filenames that are frequently modified and represent core application logic (not auto-generated files, dependencies, or configuration). Make sure filenames are diverse, not all in the same folder, and are a mix of user and other users. Return only the filenames' basenames (without the path) separated by newlines with no explanation.\n\nFiles modified by user:\n  46 \n  33 proxy_server.py\n  15 README.md\n   7 proxy_server_demo_request.py\n   7 config.json.example\n   3 proxy_server_litellm.py\n   2 requirements.txt\n   2 .gitignore\n   1 chat.py\n   1 .github/workflows/ai-code-review.yml\n   1 .github/scripts/ai_review.py\n"
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.0
        }
    }

def create_corrected_payload_v2():
    """Create another corrected payload variant"""
    return {
        "messages": [
            {
                "role": "user", 
                "content": [
                    {
                        "text": "You are an expert at analyzing git history. Given a list of files and their modification counts, return exactly five filenames that are frequently modified and represent core application logic (not auto-generated files, dependencies, or configuration). Make sure filenames are diverse, not all in the same folder, and are a mix of user and other users. Return only the filenames' basenames (without the path) separated by newlines with no explanation.\n\nFiles modified by user:\n  46 \n  33 proxy_server.py\n  15 README.md\n   7 proxy_server_demo_request.py\n   7 config.json.example\n   3 proxy_server_litellm.py\n   2 requirements.txt\n   2 .gitignore\n   1 chat.py\n   1 .github/workflows/ai-code-review.yml\n   1 .github/scripts/ai_review.py\n"
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.0
        }
    }

def create_minimal_payload():
    """Create a minimal test payload"""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Hello, how are you?"
                    }
                ]
            }
        ]
    }

def test_request(payload: Dict[str, Any], test_name: str, stream: bool = True):
    """Test a request payload against the SAP AI Core endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")
    
    headers = {
        "AI-Resource-Group": TEST_CONFIG["resource_group"],
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEST_CONFIG['token']}",
        "AI-Tenant-Id": TEST_CONFIG["tenant_id"],
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "fine-grained-tool-streaming-2025-05-14"
    }
    
    print(f"Request URL: {TEST_CONFIG['url']}")
    print(f"Request Headers: {json.dumps({k: v[:50] + '...' if k == 'Authorization' else v for k, v in headers.items()}, indent=2)}")
    print(f"Request Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            TEST_CONFIG["url"],
            headers=headers,
            json=payload,
            stream=stream,
            timeout=30
        )
        
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ SUCCESS!")
            if stream:
                print("Stream response chunks:")
                for i, chunk in enumerate(response.iter_lines(decode_unicode=True)):
                    if chunk and i < 5:  # Show first 5 chunks
                        print(f"  Chunk {i+1}: {chunk[:200]}...")
                    elif i >= 5:
                        print(f"  ... (showing first 5 chunks of many)")
                        break
            else:
                response_json = response.json()
                print(f"Response: {json.dumps(response_json, indent=2)[:500]}...")
        else:
            print("❌ FAILED!")
            try:
                error_response = response.json()
                print(f"Error Response: {json.dumps(error_response, indent=2)}")
            except:
                print(f"Error Response (text): {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST EXCEPTION: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")

def main():
    """Run all test cases"""
    print("Testing SAP AI Core Claude endpoint")
    print(f"Endpoint: {TEST_CONFIG['url']}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test cases
    test_cases = [
        (create_minimal_payload(), "Minimal test payload"),
        (create_corrected_payload_v1(), "Corrected payload v1 (with type field)"),
        (create_corrected_payload_v2(), "Corrected payload v2 (without type field)"),
        (create_failing_payload(), "Original failing payload"),
    ]
    
    for payload, name in test_cases:
        test_request(payload, name)
        time.sleep(1)  # Brief pause between requests
    
    print(f"\n{'='*60}")
    print("Test Summary Complete")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

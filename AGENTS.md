<claude-mem-context>
# Memory Context

# [sap-ai-core-llm-proxy] recent context, 2026-05-20 12:17pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (20,246t read) | 525,013t work | 96% savings

### May 19, 2026
902 5:35p 🔵 proxy_server.py — Default Model Location Identified at Line 2120
903 " 🔵 proxy_server.py — Two Hardcoded "gpt-4o" Default Locations at Lines 2120 and 2125
904 5:36p ⚖️ proxy_server.py — Default Model Target Set to Claude Opus 4.6
905 " ✅ proxy_server.py — Default and Fallback Model Changed to gpt-5.4 (Not Claude Opus 4.6)
907 5:37p 🔵 proxy_server.py /v1/messages — Separate Fallback Logic Uses Dynamic Claude/Sonnet Priority Search
908 5:38p 🟣 sfmobile sh — claude-opus-4-7 Added via SAP AI Core EU12 Deployment
909 5:39p ✅ README.md — Supported Models List and Default Fallbacks Updated
910 " ✅ Commit and Push — sap-ai-core-llm-proxy Default Model Updates
912 5:45p ✅ OpenAI Codex CLI Updated to Latest Version
913 " 🔵 OpenAI Codex CLI Version Confirmed — 0.1.2505191031
S1031 Update OpenAI Codex CLI + guidance on using SAP AI Core models with Codex (May 19 at 5:48 PM)
S1032 OpenAI Codex CLI wire_api "chat" deprecated — fix config.toml or upgrade proxy to support responses API (May 19 at 5:48 PM)
914 5:52p 🔵 OpenAI Codex CLI — wire_api "chat" No Longer Supported
S1044 Add /v1/responses endpoint to SAP AI Core LLM proxy to fix broken OpenAI Codex CLI after wire_api="chat" deprecation (May 19 at 5:52 PM)
916 5:53p 🔵 sap-ai-core-llm-proxy — Full Route and Function Map of proxy_server.py
918 " 🔵 sap-ai-core-llm-proxy — Deep Architecture: Request Routing, Token Auth, and Streaming Pipeline
919 " 🔵 OpenAI Responses API — Full Request/Response Schema for /v1/responses
920 5:54p 🔵 Codex CLI wire_api "chat" Removed — Requires "responses" — Proxy Lacks /v1/responses
922 5:55p 🔵 Codex CLI Responses API — Exact Request/Response Schema for Proxy Implementation
923 " 🔵 Responses API Streaming SSE — Exact Wire Format for response.output_text.delta
925 5:57p 🔵 OpenAI Responses API — Confirmed Schema Details for Proxy Implementation
926 " 🔵 Responses API Migration Guide — messages Array Is Compatible with input Field
927 " 🔵 Responses API Streaming — Complete SSE Event Sequence for Plain Text Response
928 5:58p 🔵 OpenAI Codex CLI — Confirmed Use of Responses API with previous_response_id for Multi-Turn
929 " 🔵 proxy_server.py — /v1/chat/completions Route Structure Confirmed for /v1/responses Patterning
930 " ⚖️ proxy_server.py — /v1/responses Implementation Plan Finalized
S1053 Fix OpenAI Codex CLI broken after update — wire_api="chat" deprecated, requires "responses" — implement /v1/responses in sap-ai-core-llm-proxy (May 19 at 5:58 PM)
931 5:59p 🔵 proxy_server.py — uuid and hashlib Not Imported, Must Be Added for /v1/responses
932 6:00p ✅ proxy_server.py — import uuid Added to Support Responses API ID Generation
935 6:02p 🟣 proxy_server.py — /v1/responses Endpoint Fully Implemented and Verified
937 6:03p 🟣 proxy_server.py — /v1/responses Endpoint Live and Returning Valid Responses
938 " 🟣 proxy_server.py — /v1/responses Streaming SSE Fully Verified End-to-End
939 " ✅ sap-ai-core-llm-proxy — CLAUDE.md Created with Full Architecture Documentation
940 " 🟣 sap-ai-core-llm-proxy — /v1/responses Feature Branch Created and Staged for Commit
942 6:04p 🟣 sap-ai-core-llm-proxy — /v1/responses Committed to feature/responses-api Branch
943 " 🟣 sap-ai-core-llm-proxy — feature/responses-api Branch Pushed to GitHub
S1057 ~/.codex/config.toml — wire_api Updated from "chat" to "responses" (May 19 at 6:04 PM)
944 6:05p 🔵 ~/.codex/config.toml — Still Uses wire_api="chat" and Points to Remote Proxy URL
945 6:06p ✅ ~/.codex/config.toml — wire_api Updated from "chat" to "responses"
S1058 ~/.codex/config.toml — base_url Switched to Localhost for Local Testing (May 19 at 6:06 PM)
946 6:07p ✅ ~/.codex/config.toml — base_url Switched to Localhost for Local Testing
S1066 Fix /v1/responses endpoint 400 Bad Request error when OpenAI Codex CLI sends requests through sap-ai-core-llm-proxy (May 19 at 6:07 PM)
947 6:08p 🔵 proxy_server.py — /v1/responses Endpoint Returns 400 Bad Request from SAP AI Core
948 " 🔵 proxy_server.py — /v1/responses Endpoint Works Locally but Fails Against SAP AI Core EU12
950 6:09p 🔵 proxy_server.py — generate_responses_streaming Passes Raw Responses API Body to handle_default_request Without Translation
951 " 🔴 proxy_server.py — /v1/responses Endpoint Refactored: Input Translation + Unsupported Field Stripping + Better Error Handling
952 6:10p 🔴 proxy_server.py — /v1/responses Non-Streaming Path Also Fixed: Payload Cleaning + Explicit Error Handling
S1068 Restart proxy server and test Codex CLI end-to-end with /v1/responses fix — now pointing at localhost (May 19 at 6:10 PM)
S1071 proxy_server.py — /v1/responses 400 Error Fix Verified End-to-End: output_text Conversion Working (May 19 at 6:11 PM)
954 6:13p 🔵 Codex CLI — Continuous Reconnecting Observed in Debug Logs
955 6:14p 🔵 Codex CLI Reconnecting — Root Cause: "unhandled incoming priority event" Warnings
959 " 🔵 sap-ai-core-llm-proxy — Working State: proxy_server.py Has Uncommitted Changes
956 " 🔵 proxy_server.py — No Backend Errors in Log; Only "unhandled incoming priority event" Warnings
957 " 🔵 proxy_server.py — Backend 400 Error: Orphaned tool_call_id in Conversation History
958 6:15p 🔵 proxy_server.py — Orphaned tool_call Is "exec_command" + Matching function_call_output Exists in Same Log
960 " 🔵 proxy_server.py — Tool Call ID Mismatch: "id" vs "call_id" Field Causes Orphaned Tool Calls
961 " 🔴 proxy_server.py — Fixed tool_call_id Mismatch in convert_responses_input_to_messages()
S1072 proxy_server.py — Fixed tool_call_id Mismatch in convert_responses_input_to_messages() (May 19 at 6:15 PM)
962 6:16p 🔵 proxy_server.py Codebase — Structure Explored
963 " 🔵 Codex CLI Stopped — Likely /v1/responses Endpoint Issue

Access 525k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>
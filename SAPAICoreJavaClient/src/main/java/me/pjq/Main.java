package me.pjq;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        try {
            System.out.println("Testing SAP AI Core Java Client");
            System.out.println("================================");
            
            testConfigJsonMode();
            
        } catch (IOException e) {
            System.err.println("Failed to initialize SAP AI Core client: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testConfigJsonMode() throws IOException {
        System.out.println("\n=== Testing with config.json (Configuration-Driven Mode) ===");
        
        SAPAICoreClient client = new SAPAICoreClient("../config.json");
        
        String testMessage = "Hello, how are you today?";
        
        try {
            System.out.println("\n--- Testing GPT-4o via config.json ---");
            String gptResponse = client.postMessage("gpt-4o", testMessage);
            System.out.println("Request: " + testMessage);
            System.out.println("Response: " + gptResponse);
        } catch (IOException e) {
            System.err.println("GPT-4o request failed: " + e.getMessage());
        }
        
        try {
            System.out.println("\n--- Testing Claude 4-Sonnet via config.json ---");
            String claudeResponse = client.postMessage("anthropic/claude-4-sonnet", testMessage);
            System.out.println("Request: " + testMessage);
            System.out.println("Response: " + claudeResponse);
        } catch (IOException e) {
            System.err.println("Claude request failed: " + e.getMessage());
        }
        
        try {
            System.out.println("\n--- Testing Gemini 2.5 Pro via config.json ---");
            String geminiResponse = client.postMessage("gemini-2.5-pro", testMessage);
            System.out.println("Request: " + testMessage);
            System.out.println("Response: " + geminiResponse);
        } catch (IOException e) {
            System.err.println("Gemini request failed: " + e.getMessage());
        }
        
        try {
            System.out.println("\n--- Testing convenience method postMessageOpenAI() ---");
            String openaiResponse = client.postMessageOpenAI(testMessage);
            System.out.println("Request: " + testMessage);
            System.out.println("Response: " + openaiResponse);
        } catch (IOException e) {
            System.err.println("OpenAI convenience method failed: " + e.getMessage());
        }
        
        try {
            System.out.println("\n--- Testing convenience method postMessageClaude() ---");
            String claudeResponse = client.postMessageClaude("What is the capital of France?");
            System.out.println("Request: What is the capital of France?");
            System.out.println("Response: " + claudeResponse);
        } catch (IOException e) {
            System.err.println("Claude convenience method failed: " + e.getMessage());
        }
        
        try {
            System.out.println("\n--- Testing convenience method postMessageGemini() ---");
            String geminiResponse = client.postMessageGemini("What is the capital of Japan?");
            System.out.println("Request: What is the capital of Japan?");
            System.out.println("Response: " + geminiResponse);
        } catch (IOException e) {
            System.err.println("Gemini convenience method failed: " + e.getMessage());
        }
        
        client.close();
    }
}
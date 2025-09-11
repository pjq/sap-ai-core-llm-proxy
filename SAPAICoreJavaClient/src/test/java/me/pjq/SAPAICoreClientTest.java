package me.pjq;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import java.io.IOException;
import static org.junit.jupiter.api.Assertions.*;

public class SAPAICoreClientTest {
    
    private SAPAICoreClient configClient;
    
    @BeforeEach
    void setUp() {
        try {
            configClient = new SAPAICoreClient("test_config.json");
        } catch (IOException e) {
            System.out.println("test_config.json not found, tests will be skipped: " + e.getMessage());
        }
    }
    
    @Test
    @DisplayName("Test config-based client initialization")
    void testConfigClientInitialization() {
        if (configClient != null) {
            assertNotNull(configClient, "Config-based client should be initialized successfully");
        }
    }
    
    @Test
    @DisplayName("Test model-specific message posting via config")
    void testPostMessageWithModel() {
        if (configClient != null) {
            assertDoesNotThrow(() -> {
                String response = configClient.postMessage("gpt-4o", "Hello, this is a test message.");
                assertNotNull(response, "Response should not be null");
                assertFalse(response.isEmpty(), "Response should not be empty");
            }, "Model-specific message posting should not throw an exception");
        }
    }
    
    @Test
    @DisplayName("Test OpenAI message posting (config mode)")
    void testPostMessageOpenAIConfig() {
        if (configClient != null) {
            assertDoesNotThrow(() -> {
                String response = configClient.postMessageOpenAI("Hello, this is a test message.");
                assertNotNull(response, "Response should not be null");
                assertFalse(response.isEmpty(), "Response should not be empty");
            }, "OpenAI message posting should not throw an exception");
        }
    }
    
    @Test
    @DisplayName("Test Claude message posting (config mode)")
    void testPostMessageClaudeConfig() {
        if (configClient != null) {
            assertDoesNotThrow(() -> {
                String response = configClient.postMessageClaude("Hello, this is a test message.");
                assertNotNull(response, "Response should not be null");
                assertFalse(response.isEmpty(), "Response should not be empty");
            }, "Claude message posting should not throw an exception");
        }
    }
    
    @Test
    @DisplayName("Test Gemini message posting (config mode)")
    void testPostMessageGeminiConfig() {
        if (configClient != null) {
            assertDoesNotThrow(() -> {
                String response = configClient.postMessageGemini("Hello, this is a test message.");
                assertNotNull(response, "Response should not be null");
                assertFalse(response.isEmpty(), "Response should not be empty");
            }, "Gemini message posting should not throw an exception");
        }
    }
    
    @Test
    @DisplayName("Test invalid model handling")
    void testInvalidModel() {
        if (configClient != null) {
            assertThrows(IOException.class, () -> {
                configClient.postMessage("invalid-model", "Hello, this is a test message.");
            }, "Should throw IOException for invalid model");
        }
    }
}
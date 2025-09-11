package me.pjq;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import okhttp3.*;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.util.Base64;

public class SAPAICoreClient {
    private static final String API_VERSION_OPENAI = "2023-05-15";
    private static final String API_VERSION_CLAUDE = "2023-05-15";
    private static final String API_VERSION_GEMINI = "2023-05-15";
    
    private final Config config;
    private final OkHttpClient httpClient;
    private final Gson gson;
    private final Map<String, ReentrantLock> tokenLocks = new HashMap<>();
    private final Map<String, String> cachedTokens = new HashMap<>();
    private final Map<String, Long> tokenExpiries = new HashMap<>();
    
    public static class ServiceKey {
        public String clientid;
        public String clientsecret;
        public String url;
        public String identityzoneid;
        public ServiceUrls serviceurls;
        
        public static class ServiceUrls {
            public String AI_API_URL;
        }
    }
    
    public static class SubAccountConfig {
        public String resource_group;
        public String service_key_json;
        public Map<String, List<String>> deployment_models;
    }
    
    public static class Config {
        public Map<String, SubAccountConfig> subAccounts;
        public List<String> secret_authentication_tokens;
        public int port;
        public String host;
        public String hostLocal;
        
        private Map<String, ServiceKey> loadedServiceKeys = new HashMap<>();
        private String configDirectory;
        
        public void setConfigDirectory(String configDirectory) {
            this.configDirectory = configDirectory;
        }
        
        public ServiceKey getServiceKey(String subAccountName) throws IOException {
            if (!loadedServiceKeys.containsKey(subAccountName)) {
                SubAccountConfig subAccount = subAccounts.get(subAccountName);
                if (subAccount == null) {
                    throw new IOException("SubAccount not found: " + subAccountName);
                }
                
                // Resolve service key path relative to config directory
                String serviceKeyPath;
                if (configDirectory != null && !configDirectory.isEmpty()) {
                    serviceKeyPath = configDirectory + "/" + subAccount.service_key_json;
                } else {
                    serviceKeyPath = subAccount.service_key_json;
                }
                
                try (FileReader reader = new FileReader(new File(serviceKeyPath))) {
                    Gson gson = new Gson();
                    ServiceKey key = gson.fromJson(reader, ServiceKey.class);
                    loadedServiceKeys.put(subAccountName, key);
                }
            }
            return loadedServiceKeys.get(subAccountName);
        }
        
        public String getDeploymentUrl(String model) {
            for (Map.Entry<String, SubAccountConfig> entry : subAccounts.entrySet()) {
                SubAccountConfig subAccount = entry.getValue();
                if (subAccount.deployment_models.containsKey(model)) {
                    List<String> urls = subAccount.deployment_models.get(model);
                    if (!urls.isEmpty()) {
                        return urls.get(0);
                    }
                }
            }
            return null;
        }
        
        public String getSubAccountForModel(String model) {
            for (Map.Entry<String, SubAccountConfig> entry : subAccounts.entrySet()) {
                SubAccountConfig subAccount = entry.getValue();
                if (subAccount.deployment_models.containsKey(model)) {
                    List<String> urls = subAccount.deployment_models.get(model);
                    if (!urls.isEmpty()) {
                        return entry.getKey();
                    }
                }
            }
            return null;
        }
    }
    
    public SAPAICoreClient(String configPath) throws IOException {
        this.httpClient = new OkHttpClient.Builder()
                .readTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
                .writeTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
                .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
                .build();
        this.gson = new Gson();
        this.config = loadConfig(configPath);
    }
    
    private Config loadConfig(String configPath) throws IOException {
        File configFile = new File(configPath);
        
        if (!configFile.exists()) {
            throw new IOException("Configuration file not found: " + configPath);
        }
        
        try (FileReader reader = new FileReader(configFile)) {
            String content = new String(java.nio.file.Files.readAllBytes(configFile.toPath()));
            
            if (content.contains("subAccounts")) {
                Config config = gson.fromJson(content, Config.class);
                
                // Extract directory path from config file path
                String configDirectory = configFile.getParent();
                if (configDirectory == null) {
                    configDirectory = "."; // Current directory if no parent
                }
                config.setConfigDirectory(configDirectory);
                
                return config;
            } else {
                throw new IOException("Single service key files are no longer supported. Please use config.json format with subAccounts structure.");
            }
        }
    }
    
    private String getTokenForSubAccount(String subAccountName) throws IOException {
        tokenLocks.computeIfAbsent(subAccountName, k -> new ReentrantLock());
        ReentrantLock lock = tokenLocks.get(subAccountName);
        
        lock.lock();
        try {
            long currentTime = System.currentTimeMillis() / 1000;
            
            String tokenKey = subAccountName + "_token";
            String expiryKey = subAccountName + "_expiry";
            
            String cachedToken = cachedTokens.get(tokenKey);
            Long expiry = tokenExpiries.get(expiryKey);
            
            if (cachedToken != null && expiry != null && currentTime < expiry) {
                return cachedToken;
            }
            
            ServiceKey serviceKey = config.getServiceKey(subAccountName);
            String authString = serviceKey.clientid + ":" + serviceKey.clientsecret;
            String encodedAuth = Base64.getEncoder().encodeToString(authString.getBytes());
            
            String tokenUrl = serviceKey.url + "/oauth/token?grant_type=client_credentials";
            
            Request request = new Request.Builder()
                    .url(tokenUrl)
                    .addHeader("Authorization", "Basic " + encodedAuth)
                    .post(RequestBody.create("", MediaType.parse("application/x-www-form-urlencoded")))
                    .build();
                    
            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new IOException("Token request failed for " + subAccountName + ": " + response.code() + " " + response.message());
                }
                
                String responseBody = response.body().string();
                JsonObject tokenData = JsonParser.parseString(responseBody).getAsJsonObject();
                
                String newToken = tokenData.get("access_token").getAsString();
                int expiresIn = tokenData.get("expires_in").getAsInt();
                long newExpiry = currentTime + expiresIn - 300;
                
                cachedTokens.put(tokenKey, newToken);
                tokenExpiries.put(expiryKey, newExpiry);
                
                return newToken;
            }
        } finally {
            lock.unlock();
        }
    }
    
    public String postMessage(String model, String message, double temperature) throws IOException {
        String subAccount = config.getSubAccountForModel(model);
        if (subAccount == null) {
            throw new IOException("Model not found in configuration: " + model + ". Available models: " + getAvailableModels());
        }
        
        String deploymentUrl = config.getDeploymentUrl(model);
        if (deploymentUrl == null) {
            throw new IOException("No deployment URL found for model: " + model);
        }
        
        if (model.startsWith("gpt-") || model.contains("openai")) {
            return postMessageOpenAI(subAccount, deploymentUrl, message, temperature);
        } else if (model.contains("claude") || model.contains("anthropic")) {
            return postMessageClaude(subAccount, deploymentUrl, message, temperature);
        } else if (model.contains("gemini")) {
            return postMessageGemini(subAccount, deploymentUrl, message, temperature);
        } else {
            throw new IOException("Unknown model type: " + model + ". Cannot determine API format.");
        }
    }
    
    public String postMessage(String model, String message) throws IOException {
        return postMessage(model, message, 0.7); // Default temperature
    }
    
    public String postMessageOpenAI(String message, double temperature) throws IOException {
        String model = findFirstModelByType("gpt");
        if (model == null) {
            throw new IOException("No OpenAI/GPT models configured. Available models: " + getAvailableModels());
        }
        return postMessage(model, message, temperature);
    }
    
    public String postMessageOpenAI(String message) throws IOException {
        return postMessageOpenAI(message, 0.7); // Default temperature
    }
    
    public String postMessageClaude(String message, double temperature) throws IOException {
        String model = findFirstModelByType("claude");
        if (model == null) {
            throw new IOException("No Claude models configured. Available models: " + getAvailableModels());
        }
        return postMessage(model, message, temperature);
    }
    
    public String postMessageClaude(String message) throws IOException {
        return postMessageClaude(message, 0.7); // Default temperature
    }
    
    public String postMessageGemini(String message, double temperature) throws IOException {
        String model = findFirstModelByType("gemini");
        if (model == null) {
            throw new IOException("No Gemini models configured. Available models: " + getAvailableModels());
        }
        return postMessage(model, message, temperature);
    }
    
    public String postMessageGemini(String message) throws IOException {
        return postMessageGemini(message, 0.7); // Default temperature
    }
    
    private String findFirstModelByType(String type) {
        for (SubAccountConfig subAccount : config.subAccounts.values()) {
            for (String model : subAccount.deployment_models.keySet()) {
                if (model.toLowerCase().contains(type.toLowerCase())) {
                    return model;
                }
            }
        }
        return null;
    }
    
    private String getAvailableModels() {
        Set<String> models = new HashSet<>();
        for (SubAccountConfig subAccount : config.subAccounts.values()) {
            models.addAll(subAccount.deployment_models.keySet());
        }
        return String.join(", ", models);
    }
    
    private String postMessageOpenAI(String subAccount, String deploymentUrl, String message, double temperature) throws IOException {
        String token = getTokenForSubAccount(subAccount);
        String endpoint = deploymentUrl + "/chat/completions?api-version=" + API_VERSION_OPENAI;
        
        JsonObject payload = new JsonObject();
        payload.addProperty("model", "gpt-4o");
        payload.addProperty("max_tokens", 4096);
        payload.addProperty("temperature", temperature);
        payload.addProperty("stream", false);
        
        JsonObject messageObj = new JsonObject();
        messageObj.addProperty("role", "user");
        messageObj.addProperty("content", message);
        
        JsonObject[] messages = {messageObj};
        payload.add("messages", gson.toJsonTree(messages));
        
        ServiceKey serviceKey = config.getServiceKey(subAccount);
        
        Request request = new Request.Builder()
                .url(endpoint)
                .addHeader("Authorization", "Bearer " + token)
                .addHeader("AI-Resource-Group", "default")
                .addHeader("AI-Tenant-Id", serviceKey.identityzoneid)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(gson.toJson(payload), MediaType.parse("application/json")))
                .build();
                
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("OpenAI request failed: " + response.code() + " " + response.message());
            }
            
            String responseBody = response.body().string();
            JsonObject responseJson = JsonParser.parseString(responseBody).getAsJsonObject();
            
            return responseJson.getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString();
        }
    }
    
    private String postMessageClaude(String subAccount, String deploymentUrl, String message, double temperature) throws IOException {
        String token = getTokenForSubAccount(subAccount);
        String endpoint = deploymentUrl + "/converse";
        
        JsonObject payload = new JsonObject();
        payload.addProperty("maxTokens", 4096);
        payload.addProperty("temperature", temperature);
        
        JsonObject messageContent = new JsonObject();
        messageContent.addProperty("text", message);
        
        JsonObject[] contentArray = {messageContent};
        
        JsonObject messageObj = new JsonObject();
        messageObj.addProperty("role", "user");
        messageObj.add("content", gson.toJsonTree(contentArray));
        
        JsonObject[] messages = {messageObj};
        payload.add("messages", gson.toJsonTree(messages));
        
        JsonObject inferenceConfig = new JsonObject();
        inferenceConfig.addProperty("maxTokens", 4096);
        inferenceConfig.addProperty("temperature", temperature);
        payload.add("inferenceConfig", inferenceConfig);
        
        ServiceKey serviceKey = config.getServiceKey(subAccount);
        
        Request request = new Request.Builder()
                .url(endpoint)
                .addHeader("Authorization", "Bearer " + token)
                .addHeader("AI-Resource-Group", "default")
                .addHeader("AI-Tenant-Id", serviceKey.identityzoneid)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(gson.toJson(payload), MediaType.parse("application/json")))
                .build();
                
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Claude request failed: " + response.code() + " " + response.message());
            }
            
            String responseBody = response.body().string();
            JsonObject responseJson = JsonParser.parseString(responseBody).getAsJsonObject();
            
            return responseJson.getAsJsonObject("output")
                    .getAsJsonObject("message")
                    .getAsJsonArray("content")
                    .get(0).getAsJsonObject()
                    .get("text").getAsString();
        }
    }
    
    private String postMessageGemini(String subAccount, String deploymentUrl, String message, double temperature) throws IOException {
        String token = getTokenForSubAccount(subAccount);
        String endpoint = deploymentUrl + "/models/gemini-2.5-pro:generateContent";
        
        JsonObject payload = new JsonObject();
        
        JsonObject parts = new JsonObject();
        parts.addProperty("text", message);
        
        JsonObject content = new JsonObject();
        content.addProperty("role", "user");
        content.add("parts", parts);
        
        payload.add("contents", content);
        
        JsonObject generationConfig = new JsonObject();
        generationConfig.addProperty("maxOutputTokens", 4096);
        generationConfig.addProperty("temperature", temperature);
        payload.add("generation_config", generationConfig);
        
        JsonObject safetySettings = new JsonObject();
        safetySettings.addProperty("category", "HARM_CATEGORY_SEXUALLY_EXPLICIT");
        safetySettings.addProperty("threshold", "BLOCK_LOW_AND_ABOVE");
        payload.add("safety_settings", safetySettings);
        
        ServiceKey serviceKey = config.getServiceKey(subAccount);
        
        Request request = new Request.Builder()
                .url(endpoint)
                .addHeader("Authorization", "Bearer " + token)
                .addHeader("AI-Resource-Group", "default")
                .addHeader("AI-Tenant-Id", serviceKey.identityzoneid)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(gson.toJson(payload), MediaType.parse("application/json")))
                .build();
                
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String errorBody = response.body() != null ? response.body().string() : "No response body";
                throw new IOException("Gemini request failed: " + response.code() + " " + response.message() + " - " + errorBody);
            }
            
            String responseBody = response.body().string();
            JsonObject responseJson = JsonParser.parseString(responseBody).getAsJsonObject();
            
            return responseJson.getAsJsonArray("candidates")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("content")
                    .getAsJsonArray("parts")
                    .get(0).getAsJsonObject()
                    .get("text").getAsString();
        }
    }
    
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
    }
}
package me.pjq;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Main {
    private static final String DEFAULT_CONFIG = "./config.json";
    private static final String DEFAULT_MESSAGE = "Hello, how are you today?";
    
    public static void main(String[] args) {
        try {
            // Parse command line arguments
            CommandLineArgs cmdArgs = parseArguments(args);
            
            if (cmdArgs.showHelp) {
                printUsage();
                return;
            }
            
            if (cmdArgs.listModels) {
                listAvailableModels(cmdArgs.configFile);
                return;
            }
            
            if (cmdArgs.model != null && cmdArgs.message != null) {
                // Single model test mode
                runSingleTest(cmdArgs.configFile, cmdArgs.model, cmdArgs.message, cmdArgs.temperature);
            } else {
                // Default demo mode
                runDemoMode(cmdArgs.configFile);
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            if (Arrays.asList(args).contains("--debug")) {
                e.printStackTrace();
            }
            System.exit(1);
        }
    }
    
    private static void runSingleTest(String configFile, String model, String message, double temperature) throws IOException {
        System.out.println("SAP AI Core Java Client - Single Test Mode");
        System.out.println("==========================================");
        System.out.println("Config: " + configFile);
        System.out.println("Model: " + model);
        System.out.println("Message: " + message);
        System.out.println("Temperature: " + temperature);
        System.out.println();
        
        SAPAICoreClient client = new SAPAICoreClient(configFile);
        
        try {
            String response = client.postMessage(model, message, temperature);
            System.out.println("Response:");
            System.out.println(response);
        } finally {
            client.close();
        }
    }
    
    private static void listAvailableModels(String configFile) throws IOException {
        System.out.println("Available Models in " + configFile + ":");
        System.out.println("=====================================");
        
        SAPAICoreClient client = new SAPAICoreClient(configFile);
        try {
            // This would require adding a method to SAPAICoreClient to list models
            System.out.println("Note: Model listing requires config.json inspection");
            System.out.println("Common models: gpt-4o, gpt-4.1, anthropic/claude-4-sonnet, gemini-2.5-pro");
        } finally {
            client.close();
        }
    }
    
    private static void runDemoMode(String configFile) throws IOException {
        System.out.println("SAP AI Core Java Client - Demo Mode");
        System.out.println("===================================");
        System.out.println("Config: " + configFile);
        System.out.println();
        
        SAPAICoreClient client = new SAPAICoreClient(configFile);
        
        String testMessage = DEFAULT_MESSAGE;
        
        try {
            System.out.println("--- Testing GPT-4o ---");
            String gptResponse = client.postMessage("gpt-4o", testMessage);
            System.out.println("Request: " + testMessage);
            System.out.println("Response: " + gptResponse);
            System.out.println();
        } catch (IOException e) {
            System.err.println("GPT-4o request failed: " + e.getMessage());
        }
        
        try {
            System.out.println("--- Testing Claude 4-Sonnet ---");
            String claudeResponse = client.postMessage("anthropic/claude-4-sonnet", testMessage);
            System.out.println("Request: " + testMessage);
            System.out.println("Response: " + claudeResponse);
            System.out.println();
        } catch (IOException e) {
            System.err.println("Claude request failed: " + e.getMessage());
        }
        
        try {
            System.out.println("--- Testing Gemini 2.5 Pro ---");
            String geminiResponse = client.postMessage("gemini-2.5-pro", testMessage);
            System.out.println("Request: " + testMessage);
            System.out.println("Response: " + geminiResponse);
            System.out.println();
        } catch (IOException e) {
            System.err.println("Gemini request failed: " + e.getMessage());
        }
        
        client.close();
    }
    
    private static CommandLineArgs parseArguments(String[] args) {
        CommandLineArgs cmdArgs = new CommandLineArgs();
        
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--help":
                case "-h":
                    cmdArgs.showHelp = true;
                    break;
                case "--config":
                case "-c":
                    if (i + 1 < args.length) {
                        cmdArgs.configFile = args[++i];
                    } else {
                        throw new IllegalArgumentException("--config requires a file path");
                    }
                    break;
                case "--model":
                case "-m":
                    if (i + 1 < args.length) {
                        cmdArgs.model = args[++i];
                    } else {
                        throw new IllegalArgumentException("--model requires a model name");
                    }
                    break;
                case "--message":
                case "-msg":
                    if (i + 1 < args.length) {
                        cmdArgs.message = args[++i];
                    } else {
                        throw new IllegalArgumentException("--message requires a message text");
                    }
                    break;
                case "--temperature":
                case "-t":
                    if (i + 1 < args.length) {
                        try {
                            cmdArgs.temperature = Double.parseDouble(args[++i]);
                            if (cmdArgs.temperature < 0.0 || cmdArgs.temperature > 1.0) {
                                throw new IllegalArgumentException("Temperature must be between 0.0 and 1.0");
                            }
                        } catch (NumberFormatException e) {
                            throw new IllegalArgumentException("Invalid temperature value: " + args[i]);
                        }
                    } else {
                        throw new IllegalArgumentException("--temperature requires a numeric value");
                    }
                    break;
                case "--list-models":
                case "-l":
                    cmdArgs.listModels = true;
                    break;
                case "--debug":
                    cmdArgs.debug = true;
                    break;
                default:
                    if (args[i].startsWith("-")) {
                        throw new IllegalArgumentException("Unknown option: " + args[i]);
                    }
                    // Ignore non-option arguments
                    break;
            }
        }
        
        return cmdArgs;
    }
    
    private static void printUsage() {
        System.out.println("SAP AI Core Java Client");
        System.out.println("Usage: java -jar sap-ai-core-client.jar [OPTIONS]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  -h, --help              Show this help message");
        System.out.println("  -c, --config <file>     Config file path (default: ./config.json)");
        System.out.println("  -m, --model <name>      Model name (e.g., gpt-4o, anthropic/claude-4-sonnet)");
        System.out.println("  -msg, --message <text>  Message to send to the model");
        System.out.println("  -t, --temperature <num> Temperature (0.0-1.0, default: 0.7)");
        System.out.println("  -l, --list-models       List available models from config");
        System.out.println("  --debug                 Show debug information on errors");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  # Run demo mode with default config");
        System.out.println("  java -jar sap-ai-core-client.jar");
        System.out.println();
        System.out.println("  # Test specific model with custom message");
        System.out.println("  java -jar sap-ai-core-client.jar --model gpt-4o --message \"Hello, world!\"");
        System.out.println();
        System.out.println("  # Use custom config and temperature");
        System.out.println("  java -jar sap-ai-core-client.jar -c ./my-config.json -m anthropic/claude-4-sonnet -msg \"Explain AI\" -t 0.3");
        System.out.println();
        System.out.println("  # List available models");
        System.out.println("  java -jar sap-ai-core-client.jar --list-models");
    }
    
    private static class CommandLineArgs {
        String configFile = DEFAULT_CONFIG;
        String model = null;
        String message = null;
        double temperature = 0.7;
        boolean showHelp = false;
        boolean listModels = false;
        boolean debug = false;
    }
}
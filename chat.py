import openai
import click
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        logging.debug(f"Configuration loaded successfully from {file_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration from {file_path}: {e}")
        raise

config = load_config('config.json')
openai.api_key = config['secret_authentication_tokens'][0]
openai.base_url = "http://127.0.0.1:3001/v1/"

@click.command()
@click.option('--model', default="gpt-4o", help='OpenAI model to use (e.g., gpt-3.5-turbo, gpt-4).')
def chat_cli(model):
    """Simple CLI chat application using OpenAI API."""
    if not openai.api_key:
        click.echo(
            "Error: OpenAI API key not found. "
            "Please set the OPENAI_API_KEY environment variable or hardcode it in the script."
        )
        return

    click.echo(f"Starting chat with model: {model}. Type 'exit' to end.")
    logging.debug(f"Chat session started with model: {model}")

    messages = [] # Initialize message history for context

    while True:
        user_input = click.prompt("You")
        if user_input.lower() == 'exit':
            logging.debug("User exited the chat.")
            break

        messages.append({"role": "user", "content": user_input}) # Add user message to history
        logging.debug(f"User input added to messages: {user_input}")

        try:
            logging.debug(f"Sending request to OpenAI API with messages: {messages}")
            response = openai.chat.completions.create(
                model=model,
                messages=messages # Send message history to maintain conversation context
            )
            assistant_reply = response.choices[0].message.content
            click.echo(f"Assistant: {assistant_reply}")
            messages.append({"role": "assistant", "content": assistant_reply}) # Add assistant message to history
            logging.debug(f"Assistant reply added to messages: {assistant_reply}")

        except openai.APIError as e:
            logging.error(f"OpenAI API error: {e}")
            click.echo(f"OpenAI API error: {e}")
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error occurred: {e}")
            click.echo(f"Connection error occurred: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            click.echo(f"An error occurred: {e}")

if __name__ == '__main__':
    chat_cli()
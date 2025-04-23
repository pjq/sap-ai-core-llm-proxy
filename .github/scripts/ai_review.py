import os
import sys
import re
import yaml
import json
import logging
from github import Github
from github.GithubException import GithubException
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting AI review script")

# Setup GitHub client
github_token = os.environ["GITHUB_TOKEN"]
g = Github(github_token)
repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])
logger.info(f"Initialized GitHub client for repo: {os.environ['GITHUB_REPOSITORY']}")

# Load configuration
config = {
    "prompt": "You are an expert code reviewer. Focus on code quality, security issues, and performance improvements.",
    "model": "gpt-4o",
    "max_tokens": 4096,
    "temperature": 0.7,
    "comment_tag": "AI-REVIEW",
    "baseUrl": "",
    "apiKey": ""
}

# Try to load user config from .github/ai-review-config.yml if it exists
try:
    logger.info("Attempting to load custom configuration")
    user_config = yaml.safe_load(open('.github/ai-review-config.yml', 'r'))
    if user_config and isinstance(user_config, dict):
        config.update(user_config)
        logger.info("Custom configuration loaded successfully")
except Exception as e:
    logger.warning(f"No custom configuration found or error loading it: {str(e)}. Using defaults.")

# Setup API configuration
# Priority: 1. Environment variable, 2. Config file, 3. Default
api_key = os.environ.get("OPENAI_API_KEY", "") or config.get("apiKey", "")
base_url = os.environ.get("OPENAI_API_BASE", "") or config.get("baseUrl", "")

if not api_key:
    logger.error("No API key provided. Set OPENAI_API_KEY in GitHub secrets or apiKey in config.")
    sys.exit(1)

logger.info(f"API configuration set up. Using base URL: {base_url or 'default OpenAI'}")

# Event-specific processing
event_name = os.environ.get("GITHUB_EVENT_NAME")
logger.info(f"Processing event type: {event_name}")

def get_client():
    logger.debug("Initializing OpenAI client")
    client_params = {"api_key": api_key}
    if base_url:
        client_params["base_url"] = base_url
    return openai.OpenAI(**client_params)

def summarize_pr(pr):
    """Generate a summary of the PR"""
    logger.info(f"Generating summary for PR #{pr.number}: {pr.title}")
    client = get_client()
    
    title = pr.title
    body = pr.body or ""
    
    try:
        diff_text = open('pr_diff.txt', 'r').read()
        logger.info(f"Loaded diff file, size: {len(diff_text)} bytes")
        if len(diff_text) > 24000:  # Truncate if too large
            logger.warning("Diff too large, truncating to 24000 characters")
            diff_text = diff_text[:24000] + "\n[Diff truncated due to size]"
    except Exception as e:
        logger.error(f"Error loading diff file: {str(e)}")
        diff_text = "[Unable to load diff]"
    
    prompt = f"""
    {config['prompt']}
    
    Please provide a summary of the following pull request:
    
    Title: {title}
    Description: {body}
    
    Changes:
    ```
    {diff_text}
    ```
    
    Provide a concise summary that includes:
    1. The main purpose of this PR
    2. Key changes made
    3. Potential impact of these changes
    """
    
    logger.info(f"Sending request to {config['model']} for PR summary")
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        logger.info("Successfully received summary response")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def review_code(pr):
    """Perform a code review on the PR"""
    logger.info(f"Performing code review for PR #{pr.number}")
    client = get_client()
    
    try:
        diff_text = open('pr_diff.txt', 'r').read()
        logger.info(f"Loaded diff file for review, size: {len(diff_text)} bytes")
        if len(diff_text) > 24000:  # Truncate if too large
            logger.warning("Diff too large for review, truncating to 24000 characters")
            diff_text = diff_text[:24000] + "\n[Diff truncated due to size]"
    except Exception as e:
        logger.error(f"Error loading diff file for review: {str(e)}")
        diff_text = "[Unable to load diff]"
    
    prompt = f"""
    {config['prompt']}
    
    Please review the following code changes and provide feedback:
    
    ```
    {diff_text}
    ```
    
    Focus on:
    1. Code quality issues
    2. Potential bugs
    3. Security concerns
    4. Performance improvements
    5. Best practices
    
    Format your feedback as a markdown list with ### headers for different sections.
    """
    
    logger.info(f"Sending request to {config['model']} for code review")
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        logger.info("Successfully received code review response")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error performing code review: {str(e)}")
        return f"Error performing code review: {str(e)}"

def post_comment(pr, content, reply_to=None):
    """Post a comment on the PR"""
    tag = f"<!-- {config['comment_tag']} -->"
    full_comment = f"{tag}\n{content}"
    
    if reply_to:
        # Reply to an existing comment
        logger.info(f"Updating existing comment {reply_to}")
        try:
            comment = repo.get_issue_comment(int(reply_to))
            comment.edit(full_comment)
            logger.info("Comment updated successfully")
        except Exception as e:
            logger.error(f"Error updating comment: {str(e)}")
    else:
        # Create a new comment
        logger.info(f"Creating new comment on PR #{pr.number}")
        try:
            pr.create_issue_comment(full_comment)
            logger.info("Comment created successfully")
        except Exception as e:
            logger.error(f"Error creating comment: {str(e)}")

def handle_command(pr, command, comment_id):
    """Handle comment commands"""
    logger.info(f"Handling command: '{command}' from comment {comment_id}")
    client = get_client()
    
    # Parse command: /ai-review command [additional parameters]
    parts = command.split(' ', 2)
    action = parts[1] if len(parts) > 1 else "review"
    params = parts[2] if len(parts) > 2 else ""
    logger.info(f"Parsed command - action: '{action}', params: '{params}'")
    
    if action == "summarize":
        logger.info("Executing summarize command")
        summary = summarize_pr(pr)
        post_comment(pr, f"## PR Summary\n\n{summary}")
        
    elif action == "review":
        logger.info("Executing review command")
        review = review_code(pr)
        post_comment(pr, f"## AI Code Review\n\n{review}")
        
    elif action == "ask":
        logger.info(f"Executing ask command with question: {params}")
        try:
            diff_text = open('pr_diff.txt', 'r').read()
            if len(diff_text) > 20000:
                logger.warning("Diff too large for query, truncating")
                diff_text = diff_text[:20000]
        except Exception as e:
            logger.error(f"Error loading diff file for query: {str(e)}")
            diff_text = "[Unable to load diff]"
            
        prompt = f"""
        {config['prompt']}
        
        Regarding this code:
        ```
        {diff_text}
        ```
        
        The user is asking: {params}
        
        Please provide a helpful response.
        """
        
        logger.info("Sending ask request to model")
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )
            logger.info("Successfully received response to question")
            post_comment(pr, f"## Response to: {params}\n\n{response.choices[0].message.content}")
        except Exception as e:
            logger.error(f"Error processing ask command: {str(e)}")
            post_comment(pr, f"## Error\nThere was an error processing your question: {str(e)}")
        
    elif action == "clean":
        logger.info("Executing clean command")
        # Remove all AI review comments
        comments = pr.get_issue_comments()
        tag = f"<!-- {config['comment_tag']} -->"
        deleted_count = 0
        
        for comment in comments:
            if tag in comment.body:
                try:
                    comment.delete()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting comment {comment.id}: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} comments")
        post_comment(pr, f"AI review comments have been cleaned up. {deleted_count} comments were removed.")
        
    elif action == "help":
        logger.info("Executing help command")
        help_text = """
        ## AI Code Review Help
        
        Available commands:
        - `/ai-review summarize` - Generate a summary of the PR
        - `/ai-review review` - Perform a code review
        - `/ai-review ask [your question]` - Ask a question about the code
        - `/ai-review clean` - Remove all AI review comments
        - `/ai-review help` - Show this help message
        
        You can customize the behavior by adding a `.github/ai-review-config.yml` file to your repository.
        """
        post_comment(pr, help_text)
        
    else:
        logger.warning(f"Unknown command received: {action}")
        post_comment(pr, f"Unknown command: {action}. Try `/ai-review help` for a list of commands.")

def main():
    logger.info("Starting main execution flow")
    try:
        if event_name == "pull_request":
            pr_number = os.environ.get("PR_NUMBER") or os.environ.get("GITHUB_EVENT_PATH", "").split("/")[-2]
            logger.info(f"Processing pull request #{pr_number}")
            pr = repo.get_pull(int(pr_number))
            
            # For new PRs, provide both a summary and review
            logger.info("Generating PR summary")
            summary = summarize_pr(pr)
            post_comment(pr, f"## PR Summary\n\n{summary}")
            
            logger.info("Generating code review")
            review = review_code(pr)
            post_comment(pr, f"## AI Code Review\n\n{review}")
            
        elif event_name == "issue_comment":
            comment_body = os.environ.get("COMMENT_BODY", "")
            issue_number = int(os.environ["ISSUE_NUMBER"])
            comment_id = os.environ["COMMENT_ID"]
            logger.info(f"Processing comment on issue #{issue_number}, comment ID: {comment_id}")
            
            # Process commands that start with /ai-review
            if comment_body.startswith("/ai-review"):
                logger.info(f"AI review command detected: {comment_body}")
                pr = repo.get_pull(issue_number)
                handle_command(pr, comment_body, comment_id)
            else:
                logger.info("Comment is not an AI review command, ignoring")
        else:
            logger.warning(f"Unsupported event type: {event_name}")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
        logger.info("Script completed successfully")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
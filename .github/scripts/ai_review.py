import os
import sys
import re
import yaml
import json
from github import Github
from github.GithubException import GithubException
import openai

# Setup GitHub client
github_token = os.environ["GITHUB_TOKEN"]
g = Github(github_token)
repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])

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
    user_config = yaml.safe_load(open('.github/ai-review-config.yml', 'r'))
    if user_config and isinstance(user_config, dict):
        config.update(user_config)
except:
    print("No custom configuration found or error loading it. Using defaults.")

# Setup API configuration
# Priority: 1. Environment variable, 2. Config file, 3. Default
api_key = os.environ.get("OPENAI_API_KEY", "") or config.get("apiKey", "")
base_url = os.environ.get("OPENAI_API_BASE", "") or config.get("baseUrl", "")

if not api_key:
    print("Error: No API key provided. Set OPENAI_API_KEY in GitHub secrets or apiKey in config.")
    sys.exit(1)

# Event-specific processing
event_name = os.environ.get("GITHUB_EVENT_NAME")

def get_client():
    client_params = {"api_key": api_key}
    if base_url:
        client_params["base_url"] = base_url
    return openai.OpenAI(**client_params)

def summarize_pr(pr):
    """Generate a summary of the PR"""
    client = get_client()
    
    title = pr.title
    body = pr.body or ""
    diff_text = open('pr_diff.txt', 'r').read()
    if len(diff_text) > 24000:  # Truncate if too large
        diff_text = diff_text[:24000] + "\n[Diff truncated due to size]"
    
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
    
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"]
    )
    
    return response.choices[0].message.content

def review_code(pr):
    """Perform a code review on the PR"""
    client = get_client()
    
    diff_text = open('pr_diff.txt', 'r').read()
    if len(diff_text) > 24000:  # Truncate if too large
        diff_text = diff_text[:24000] + "\n[Diff truncated due to size]"
    
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
    
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"]
    )
    
    return response.choices[0].message.content

def post_comment(pr, content, reply_to=None):
    """Post a comment on the PR"""
    tag = f"<!-- {config['comment_tag']} -->"
    full_comment = f"{tag}\n{content}"
    
    if reply_to:
        # Reply to an existing comment
        comment = repo.get_issue_comment(int(reply_to))
        comment.edit(full_comment)
    else:
        # Create a new comment
        pr.create_issue_comment(full_comment)

def handle_command(pr, command, comment_id):
    """Handle comment commands"""
    client = get_client()
    
    # Parse command: /ai-review command [additional parameters]
    parts = command.split(' ', 2)
    action = parts[1] if len(parts) > 1 else "review"
    params = parts[2] if len(parts) > 2 else ""
    
    if action == "summarize":
        summary = summarize_pr(pr)
        post_comment(pr, f"## PR Summary\n\n{summary}")
        
    elif action == "review":
        review = review_code(pr)
        post_comment(pr, f"## AI Code Review\n\n{review}")
        
    elif action == "ask":
        diff_text = open('pr_diff.txt', 'r').read()
        prompt = f"""
        {config['prompt']}
        
        Regarding this code:
        ```
        {diff_text[:20000] if len(diff_text) > 20000 else diff_text}
        ```
        
        The user is asking: {params}
        
        Please provide a helpful response.
        """
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        
        post_comment(pr, f"## Response to: {params}\n\n{response.choices[0].message.content}")
        
    elif action == "clean":
        # Remove all AI review comments
        comments = pr.get_issue_comments()
        tag = f"<!-- {config['comment_tag']} -->"
        
        for comment in comments:
            if tag in comment.body:
                comment.delete()
        
        post_comment(pr, "AI review comments have been cleaned up.")
        
    elif action == "help":
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
        post_comment(pr, f"Unknown command: {action}. Try `/ai-review help` for a list of commands.")

def main():
    if event_name == "pull_request":
        pr_number = os.environ.get("PR_NUMBER") or os.environ.get("GITHUB_EVENT_PATH", "").split("/")[-2]
        pr = repo.get_pull(int(pr_number))
        
        # For new PRs, provide both a summary and review
        summary = summarize_pr(pr)
        post_comment(pr, f"## PR Summary\n\n{summary}")
        
        review = review_code(pr)
        post_comment(pr, f"## AI Code Review\n\n{review}")
        
    elif event_name == "issue_comment":
        comment_body = os.environ.get("COMMENT_BODY", "")
        issue_number = int(os.environ["ISSUE_NUMBER"])
        comment_id = os.environ["COMMENT_ID"]
        
        # Process commands that start with /ai-review
        if comment_body.startswith("/ai-review"):
            pr = repo.get_pull(issue_number)
            handle_command(pr, comment_body, comment_id)

if __name__ == "__main__":
    main()
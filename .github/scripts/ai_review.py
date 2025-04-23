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
    "apiKey": "",
    "command_prefix": "/"
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

def check_diff_file():
    """Check if diff file exists and has content"""
    try:
        if not os.path.exists('pr_diff.txt'):
            logger.error("pr_diff.txt file does not exist")
            return False
            
        file_size = os.path.getsize('pr_diff.txt')
        logger.info(f"pr_diff.txt file exists, size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("pr_diff.txt exists but is empty")
            return False
            
        with open('pr_diff.txt', 'r') as f:
            first_line = f.readline().strip()
            logger.info(f"First line of diff: {first_line[:50]}...")
            
        return True
    except Exception as e:
        logger.error(f"Error checking diff file: {str(e)}")
        return False

def summarize_pr(pr):
    """Generate a summary of the PR"""
    logger.info(f"Generating summary for PR #{pr.number}: {pr.title}")
    client = get_client()
    
    title = pr.title
    body = pr.body or ""
    
    # Check if diff file exists and has content
    if not check_diff_file():
        logger.warning("Using PR details only since diff file is problematic")
        diff_text = "[Diff unavailable]"
    else:
        try:
            with open('pr_diff.txt', 'r', encoding='utf-8') as f:
                diff_text = f.read()
            
            logger.info(f"Loaded diff file, size: {len(diff_text)} bytes")
            if len(diff_text) > 24000:  # Truncate if too large
                logger.warning("Diff too large, truncating to 24000 characters")
                diff_text = diff_text[:24000] + "\n[Diff truncated due to size]"
            elif len(diff_text) < 10:
                logger.warning(f"Diff file is suspiciously small ({len(diff_text)} bytes)")
                diff_text = f"[Warning: Diff file is very small: '{diff_text}']"
        except Exception as e:
            logger.error(f"Error loading diff file: {str(e)}")
            diff_text = f"[Unable to load diff: {str(e)}]"
    
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
    
    # Check if diff file exists and has content
    if not check_diff_file():
        error_message = "Unable to perform code review: The diff file is missing or empty. This might happen if the PR doesn't contain any changes or if there was an error fetching the diff."
        logger.error(error_message)
        return error_message
    
    try:
        with open('pr_diff.txt', 'r', encoding='utf-8') as f:
            diff_text = f.read()
        
        logger.info(f"Loaded diff file for review, size: {len(diff_text)} bytes")
        if len(diff_text) > 24000:  # Truncate if too large
            logger.warning("Diff too large for review, truncating to 24000 characters")
            diff_text = diff_text[:24000] + "\n[Diff truncated due to size]"
        elif len(diff_text) < 10:
            logger.warning(f"Diff file is suspiciously small ({len(diff_text)} bytes): '{diff_text}'")
            return f"Unable to perform code review: The diff appears to be empty or invalid. Diff content: '{diff_text}'"
    except Exception as e:
        logger.error(f"Error loading diff file for review: {str(e)}")
        return f"Error loading diff file for review: {str(e)}"
    
    prompt = f"""
    {config['prompt']}
    
    Please review the following code changes and provide detailed feedback.
    If you don't see any code to review, mention that explicitly.
    
    Changes for review:
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
        review_content = response.choices[0].message.content
        
        # Check if the response indicates an empty code block
        if "empty code block" in review_content.lower() or "no code to review" in review_content.lower():
            logger.warning("Model reported empty code block")
            # Debug the diff content
            logger.info(f"Diff preview (first 200 chars): {diff_text[:200]}")
            
            # Try an alternative approach - fetch directly from GitHub API
            try:
                logger.info("Attempting to fetch diff directly from GitHub API")
                files = pr.get_files()
                files_content = []
                
                for file in files:
                    files_content.append(f"File: {file.filename} (Status: {file.status})")
                    if file.patch:
                        files_content.append(file.patch)
                    else:
                        files_content.append("[No patch available]")
                    
                if files_content:
                    alternative_diff = "\n\n".join(files_content)
                    logger.info(f"Fetched {len(files_content)} files from API")
                    
                    alt_prompt = f"""
                    {config['prompt']}
                    
                    Please review the following code changes and provide feedback:
                    
                    ```
                    {alternative_diff[:24000]}
                    ```
                    
                    Format your feedback as a markdown list with ### headers for different sections.
                    """
                    
                    logger.info("Sending second request with API-fetched diff")
                    alt_response = client.chat.completions.create(
                        model=config["model"],
                        messages=[{"role": "user", "content": alt_prompt}],
                        max_tokens=config["max_tokens"],
                        temperature=config["temperature"]
                    )
                    
                    review_content = alt_response.choices[0].message.content
                    logger.info("Successfully received alternative code review")
                else:
                    logger.warning("No files found in PR through API")
                    review_content += "\n\nNo changes were found in this PR using the GitHub API either."
                    
            except Exception as e:
                logger.error(f"Error in alternative diff approach: {str(e)}")
                review_content += f"\n\nAttempted to fetch PR details directly from GitHub API but encountered an error: {str(e)}"
        
        return review_content
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

def post_inline_comments(pr, file_path, line_num, content):
    """Post a comment on a specific line in the PR"""
    tag = f"<!-- {config['comment_tag']} -->"
    full_comment = f"{tag}\n{content}"
    logger.info(f"Adding inline comment to {file_path}:{line_num}")
    
    try:
        # Get the PR for inline comments
        files = pr.get_files()
        for file in files:
            if file.filename == file_path:
                try:
                    # Create review comment using the correct method signature
                    pr.create_review_comment(
                        body=full_comment,
                        path=file_path,
                        position=line_num,  # Using line_num as position - may need adjustment
                    )
                    logger.info(f"Inline comment added successfully to {file_path}:{line_num}")
                    return True
                except Exception as e:
                    logger.error(f"Error adding inline comment: {str(e)}")
                    
                    # Fallback: try to create a review and add comments to it
                    try:
                        logger.info("Trying alternative approach with a review")
                        review = pr.create_review(
                            body="Code review comments",
                            comments=[{
                                'path': file_path, 
                                'position': line_num, 
                                'body': full_comment
                            }]
                        )
                        logger.info("Successfully added comment via review")
                        return True
                    except Exception as e2:
                        logger.error(f"Alternative approach also failed: {str(e2)}")
                        return False
                
        logger.warning(f"File {file_path} not found in PR files")
        return False
    except Exception as e:
        logger.error(f"Error adding inline comment: {str(e)}")
        return False

def review_code_with_inline_comments(pr):
    """Perform a code review with inline comments"""
    logger.info(f"Performing inline code review for PR #{pr.number}")
    client = get_client()
    
    # Get files changed in the PR
    try:
        logger.info("Fetching files changed in the PR")
        files = list(pr.get_files())
        logger.info(f"Found {len(files)} changed files")
        
        if not files:
            logger.warning("No files found in PR")
            return "No files to review in this PR."
            
        overall_comments = []
        
        # Process each file
        for file in files:
            file_path = file.filename
            logger.info(f"Reviewing file: {file_path}")
            if not file.patch:
                logger.info(f"No patch available for {file_path}, skipping")
                continue
                
            # Skip binary files, very large files
            if file.patch and len(file.patch) > 20000:
                logger.warning(f"File {file_path} has a very large patch ({len(file.patch)} chars), truncating")
                file_patch = file.patch[:20000] + "\n[Patch truncated due to size]"
            else:
                file_patch = file.patch
                
            # Analyze the file
            prompt = f"""
            {config['prompt']}
            
            Review the following file changed in a pull request: {file_path}
            
            ```
            {file_patch}
            ```
            
            Provide specific line-by-line comments where issues are found. For each comment, include the exact line number.
            Format each comment as follows:

            LINE: [line number]
            COMMENT: [your detailed comment about the issue]
            
            Also provide a brief overall assessment of the file at the end.
            """
            
            logger.info(f"Sending request to analyze {file_path}")
            try:
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"]
                )
                
                analysis = response.choices[0].message.content
                logger.info(f"Received analysis for {file_path}")
                
                # Extract line-specific comments
                line_comments = []
                overall_file_comments = []
                current_lines = []
                
                # Parse line-specific comments
                lines = analysis.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    if line.startswith("LINE:"):
                        line_num = line.replace("LINE:", "").strip()
                        
                        # Look for the corresponding comment
                        comment_text = ""
                        j = i + 1
                        while j < len(lines) and not lines[j].strip().startswith("LINE:"):
                            if lines[j].strip().startswith("COMMENT:"):
                                comment_text = lines[j].replace("COMMENT:", "").strip()
                                # Include additional comment lines
                                j += 1
                                while j < len(lines) and not (lines[j].strip().startswith("LINE:") or lines[j].strip().startswith("COMMENT:")):
                                    if lines[j].strip():  # Only add non-empty lines
                                        comment_text += "\n" + lines[j].strip()
                                    j += 1
                                break
                            j += 1
                            
                        if comment_text:
                            try:
                                line_num = int(line_num)
                                line_comments.append((line_num, comment_text))
                            except ValueError:
                                logger.warning(f"Invalid line number: {line_num}, adding to overall comments")
                                overall_file_comments.append(f"Comment for line {line_num}: {comment_text}")
                        
                        i = j - 1  # Adjust i to continue from the right position
                    else:
                        # Collect lines that aren't line-specific comments for overall assessment
                        if not line.startswith("COMMENT:"):
                            overall_file_comments.append(line)
                    i += 1
                
                # Add overall file comments if there are any
                if overall_file_comments:
                    overall_file_assessment = "\n".join(overall_file_comments)
                    overall_comments.append(f"## {file_path}\n\n{overall_file_assessment}")
                
                # Post line-specific comments
                for line_num, comment_text in line_comments:
                    logger.info(f"Adding comment to {file_path}:{line_num}")
                    success = post_inline_comments(pr, file_path, line_num, comment_text)
                    if not success:
                        # If inline comment fails, add to overall comments
                        overall_comments.append(f"Comment for {file_path}:{line_num}: {comment_text}")
                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {str(e)}")
                overall_comments.append(f"Error analyzing {file_path}: {str(e)}")
        
        # Return overall assessment
        if overall_comments:
            return "## Overall Code Review\n\n" + "\n\n".join(overall_comments)
        else:
            return "Code review completed. Inline comments have been added to the PR."
            
    except Exception as e:
        logger.error(f"Error performing inline code review: {str(e)}")
        return f"Error performing inline code review: {str(e)}"

def handle_command(pr, command, comment_id):
    """Handle comment commands"""
    # Determine if using simplified command format
    prefix = config.get("command_prefix", "/")
    
    # Check if this is a simplified command
    if command.startswith(prefix) and not command.startswith("/ai-review"):
        # Strip the prefix and get the command
        cmd = command[len(prefix):].strip().split(' ', 1)
        action = cmd[0].lower()
        params = cmd[1] if len(cmd) > 1 else ""
    else:
        # Original format: /ai-review command [params]
        parts = command.split(' ', 2)
        action = parts[1] if len(parts) > 1 else "review"
        params = parts[2] if len(parts) > 2 else ""
        
    logger.info(f"Handling command: '{action}' with params: '{params}'")
    
    client = get_client()  # Initialize client here for all commands
    
    if action in ["summarize", "summary", "s"]:
        logger.info("Executing summarize command")
        summary = summarize_pr(pr)
        post_comment(pr, f"## PR Summary\n\n{summary}")
        
    elif action in ["review", "r"]:
        logger.info("Executing review command")
        # Run both types of review - overall and inline comments
        post_comment(pr, "Reviewing code... this may take a minute.")
        
        # Add inline comments
        review_code_with_inline_comments(pr)
        
        # Also add overall review
        review = review_code(pr)
        post_comment(pr, f"## AI Code Review\n\n{review}")
        
    elif action in ["ask", "a", "q", "question"]:
        logger.info(f"Executing ask command with question: {params}")
        
        # Check if diff file exists and has content
        if not check_diff_file():
            error_message = "Unable to process question: The diff file is missing or empty."
            logger.error(error_message)
            post_comment(pr, f"## Error\n\n{error_message}")
            return
            
        try:
            with open('pr_diff.txt', 'r', encoding='utf-8') as f:
                diff_text = f.read()
                
            if len(diff_text) > 20000:
                logger.warning("Diff too large for query, truncating")
                diff_text = diff_text[:20000]
        except Exception as e:
            logger.error(f"Error loading diff file for query: {str(e)}")
            diff_text = f"[Unable to load diff: {str(e)}]"
            
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
        
    elif action in ["clean", "clear", "c"]:
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
        
        # Also try to remove inline comments
        try:
            review_comments = pr.get_comments()
            for comment in review_comments:
                if tag in comment.body:
                    try:
                        comment.delete()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting review comment: {str(e)}")
        except Exception as e:
            logger.error(f"Error retrieving review comments: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} comments")
        post_comment(pr, f"AI review comments have been cleaned up. {deleted_count} comments were removed.")
        
    elif action in ["help", "h", "?"]:
        logger.info("Executing help command")
        help_text = f"""
        ## AI Code Review Help
        
        Available commands:
        - `{prefix}summary` or `{prefix}s` - Generate a summary of the PR
        - `{prefix}review` or `{prefix}r` - Perform a code review (with inline comments)
        - `{prefix}ask [your question]` or `{prefix}q [your question]` - Ask a question about the code
        - `{prefix}clean` or `{prefix}c` - Remove all AI review comments
        - `{prefix}help` or `{prefix}h` - Show this help message
        
        You can customize the behavior by adding a `.github/ai-review-config.yml` file to your repository.
        
        Full command format (also supported): `/ai-review [command] [params]`
        """
        post_comment(pr, help_text)
        
    else:
        logger.warning(f"Unknown command received: {action}")
        post_comment(pr, f"Unknown command: {action}. Try `{prefix}help` for a list of commands.")

def main():
    logger.info("Starting main execution flow")
    try:
        if event_name == "pull_request":
            pr_number_str = os.environ.get("PR_NUMBER")
            if not pr_number_str:
                event_path = os.environ.get("GITHUB_EVENT_PATH", "")
                if event_path:
                    try:
                        with open(event_path, 'r') as f:
                            event_data = json.load(f)
                            pr_number_str = str(event_data.get('number', ''))
                            logger.info(f"Extracted PR number {pr_number_str} from event data")
                    except Exception as e:
                        logger.error(f"Failed to extract PR number from event data: {str(e)}")
            
            if not pr_number_str:
                logger.error("Could not determine PR number. Exiting.")
                sys.exit(1)
                
            pr_number = int(pr_number_str)
            logger.info(f"Processing pull request #{pr_number}")
            pr = repo.get_pull(pr_number)
            
            # Check if PR contains changes
            if not check_diff_file():
                logger.warning("Diff file is problematic, proceeding with caution")
            
            # For new PRs, provide both a summary and review
            logger.info("Generating PR summary")
            summary = summarize_pr(pr)
            post_comment(pr, f"## PR Summary\n\n{summary}")
            
            # Add inline comments
            logger.info("Performing inline code review")
            review_code_with_inline_comments(pr)
            
            # Also add overall review
            logger.info("Generating overall code review")
            review = review_code(pr)
            post_comment(pr, f"## AI Code Review\n\n{review}")
            
        elif event_name == "issue_comment":
            comment_body = os.environ.get("COMMENT_BODY", "")
            issue_number = int(os.environ["ISSUE_NUMBER"])
            comment_id = os.environ["COMMENT_ID"]
            logger.info(f"Processing comment on issue #{issue_number}, comment ID: {comment_id}")
            
            prefix = config.get("command_prefix", "/")
            # Process commands that start with prefix or /ai-review
            if comment_body.startswith(prefix) or comment_body.startswith("/ai-review"):
                logger.info(f"AI review command detected: {comment_body}")
                pr = repo.get_pull(issue_number)
                
                # If command is a review or ask, make sure we have the diff
                if any(keyword in comment_body.lower() for keyword in ["review", "r", "ask", "a", "q", "question"]):
                    if not os.path.exists('pr_diff.txt'):
                        logger.info("Fetching PR diff for command")
                        diff_url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/pull/{issue_number}.diff"
                        try:
                            import requests
                            response = requests.get(diff_url)
                            if response.status_code == 200:
                                with open('pr_diff.txt', 'w', encoding='utf-8') as f:
                                    f.write(response.text)
                                logger.info(f"Downloaded diff, size: {len(response.text)} bytes")
                            else:
                                logger.error(f"Failed to download diff, status code: {response.status_code}")
                        except Exception as e:
                            logger.error(f"Error downloading diff: {str(e)}")
                
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
import os
import sys
import re
import yaml
import json
import logging
from github import Github
from github.GithubException import GithubException, UnknownObjectException
from github.Commit import Commit
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
    config_path = '.github/ai-review-config.yml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config and isinstance(user_config, dict):
                config.update(user_config)
                logger.info("Custom configuration loaded successfully")
            else:
                logger.warning("Custom configuration file is empty or invalid. Using defaults.")
    else:
        logger.info("No custom configuration file found. Using defaults.")
except Exception as e:
    logger.warning(f"Error loading custom configuration: {str(e)}. Using defaults.")

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
    try:
        return openai.OpenAI(**client_params)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        sys.exit(1)

def check_diff_file():
    """Check if diff file exists and has content"""
    diff_file_path = 'pr_diff.txt'
    try:
        if not os.path.exists(diff_file_path):
            logger.error(f"{diff_file_path} file does not exist")
            return False
            
        file_size = os.path.getsize(diff_file_path)
        logger.info(f"{diff_file_path} file exists, size: {file_size} bytes")
        
        if file_size < 10: # Allow for very small diffs but log warning
              logger.warning(f"{diff_file_path} exists but is very small ({file_size} bytes)")
              with open(diff_file_path, 'r', encoding='utf-8') as f:
                 content = f.read()
                 logger.warning(f"Small diff content: '{content[:100]}...'")
              # Decide if small diffs should proceed. Let's allow them for now.
              # return False # Uncomment this line to block very small diffs

        with open(diff_file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            logger.info(f"First line of diff: {first_line[:100]}...") # Log more chars
            
        return True
    except Exception as e:
        logger.error(f"Error checking diff file ({diff_file_path}): {str(e)}")
        return False

def get_pr_diff_text():
    """Loads diff text, handles truncation and errors."""
    if not check_diff_file():
        return None # Indicate diff is unavailable

    diff_file_path = 'pr_diff.txt'
    try:
        with open(diff_file_path, 'r', encoding='utf-8') as f:
            diff_text = f.read()
        
        logger.info(f"Loaded diff file, size: {len(diff_text)} bytes")
        max_diff_chars = 24000 # Consider making this configurable
        if len(diff_text) > max_diff_chars:
            logger.warning(f"Diff too large, truncating to {max_diff_chars} characters")
            diff_text = diff_text[:max_diff_chars] + f"\n[Diff truncated due to size limit of {max_diff_chars} characters]"
        
        return diff_text
    except Exception as e:
        logger.error(f"Error loading diff file ({diff_file_path}): {str(e)}")
        return f"[Unable to load diff: {str(e)}]" # Return error string

def summarize_pr(pr):
    """Generate a summary of the PR"""
    logger.info(f"Generating summary for PR #{pr.number}: {pr.title}")
    client = get_client()
    
    title = pr.title
    body = pr.body or ""
    
    diff_text = get_pr_diff_text()
    if diff_text is None:
        logger.warning("Diff file problematic. Using PR Title/Body only for summary.")
        diff_text = "[Diff content unavailable]"
    elif not diff_text.strip() or diff_text.startswith("[Unable to load diff"):
        logger.warning("Diff content is effectively empty or indicates loading error.")
        # Proceed but use the potentially error-containing diff_text

    prompt = f"""
    {config['prompt']}
    
    Please provide a summary of the following pull request:
    
    Title: {title}
    Description: {body}
    
    Changes:
    ```diff
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
        logger.error(f"Error generating summary via API: {str(e)}")
        return f"Error generating summary: {str(e)}"

def review_code(pr):
    """Perform an overall code review on the PR"""
    logger.info(f"Performing overall code review for PR #{pr.number}")
    client = get_client()
    
    diff_text = get_pr_diff_text()
    if diff_text is None:
        error_message = "Unable to perform code review: The diff file is missing or empty. This might happen if the PR doesn't contain any changes or if there was an error fetching the diff."
        logger.error(error_message)
        return error_message
    elif not diff_text.strip() or diff_text.startswith("[Unable to load diff"):
         logger.warning("Diff content is effectively empty or indicates loading error. Review might be inaccurate.")
         # Proceed with potentially problematic diff_text

    prompt = f"""
    {config['prompt']}
    
    Please review the following code changes (provided as a diff) and provide detailed feedback.
    If you don't see any meaningful code changes in the diff, please state that clearly.
    
    Changes for review:
    ```diff
    {diff_text}
    ```
    
    Focus on:
    - Code quality issues
    - Potential bugs or edge cases
    - Security vulnerabilities
    - Performance bottlenecks or improvements
    - Adherence to best practices
    - Maintainability and readability
    
    Structure your feedback using markdown. Use headings (e.g., `### Security Concerns`) for different categories if applicable. Be specific and provide examples where possible. Do not comment on the diff format itself unless it indicates a problem with the changes.
    """
    
    logger.info(f"Sending request to {config['model']} for overall code review")
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        logger.info("Successfully received overall code review response")
        review_content = response.choices[0].message.content
        
        # Basic check for empty-looking response
        if "no meaningful code changes" in review_content.lower() or "no code to review" in review_content.lower() or len(review_content) < 100:
            logger.warning(f"Model indicated potentially empty diff or provided short review: '{review_content[:100]}...'")
            # Optionally, add a fallback here like in the original code to fetch files via API if needed
            
        return review_content
    except Exception as e:
        logger.error(f"Error performing overall code review via API: {str(e)}")
        return f"Error performing overall code review: {str(e)}"

def post_comment(pr, content, reply_to=None):
    """Post or update a general comment on the PR"""
    tag = f"<!-- {config['comment_tag']} -->"
    full_comment = f"{tag}\n{content}"
    
    max_comment_length = 65536 # GitHub API limit
    if len(full_comment) > max_comment_length:
        logger.warning(f"Comment length ({len(full_comment)}) exceeds GitHub limit ({max_comment_length}). Truncating.")
        full_comment = full_comment[:max_comment_length - 100] + "\n\n[Comment truncated due to length limit]"

    try:
        if reply_to:
            # Update existing comment - requires comment ID
            logger.info(f"Attempting to update existing comment {reply_to}")
            comment = repo.get_issue_comment(int(reply_to))
            comment.edit(full_comment)
            logger.info(f"Comment {reply_to} updated successfully on PR #{pr.number}")
        else:
            # Create new comment
            logger.info(f"Creating new comment on PR #{pr.number}")
            pr.create_issue_comment(full_comment)
            logger.info(f"New comment created successfully on PR #{pr.number}")
    except UnknownObjectException:
         logger.error(f"Error posting comment: Comment {reply_to} not found (maybe deleted?). Attempting to post new comment.")
         if reply_to: # Only try posting new if we were trying to update
              post_comment(pr, content) # Recursive call without reply_to
    except GithubException as e:
        logger.error(f"Error posting/updating comment on PR #{pr.number}: {e.status} {e.data}")
    except Exception as e:
        logger.error(f"Unexpected error posting/updating comment on PR #{pr.number}: {str(e)}")

def post_inline_comments(pr: 'PullRequest', file_path: str, line_num: int, content: str) -> bool:
    """Post an inline comment on a specific line in the PR, trying multiple methods."""
    tag = f"<!-- {config['comment_tag']} -->"
    full_comment = f"{tag}\n{content}"
    logger.info(f"Attempting to add inline comment to {file_path}:{line_num}")

    # Ensure comment isn't too long for inline comments (limit is same as regular comments)
    max_comment_length = 65536
    if len(full_comment) > max_comment_length:
        logger.warning(f"Inline comment for {file_path}:{line_num} too long ({len(full_comment)}), truncating.")
        full_comment = full_comment[:max_comment_length - 100] + "\n\n[Comment truncated due to length limit]"

    try:
        # Method 1: Create a standalone review comment using create_review_comment
        # Requires the Commit object
        head_sha = pr.head.sha
        logger.debug(f"Fetching commit object for SHA: {head_sha}")
        commit: Commit = repo.get_commit(sha=head_sha)

        try:
            logger.debug(f"Attempting method 1: create_review_comment for {file_path}:{line_num}")
            pr.create_review_comment(
                body=full_comment,
                commit=commit,         # Correct: Pass the Commit object
                path=file_path,
                line=line_num          # Correct: Use 'line' parameter for line number in the file
            )
            logger.info(f"Inline comment added successfully (Method 1) to {file_path}:{line_num}")
            return True
        except GithubException as e1:
            logger.warning(f"Method 1 (create_review_comment) failed for {file_path}:{line_num}: {e1.status} {e1.data}. Trying Method 2.")

            # Method 2: Create a review and add the comment to it using create_review
            try:
                logger.debug(f"Attempting method 2: create_review for {file_path}:{line_num}")
                # create_review uses commit_id (string SHA)
                review = pr.create_review(
                    commit_id=head_sha,    # Correct: Use commit_id string SHA
                    body="",               # Keep body empty unless needed for overall review msg
                    event="COMMENT",       # Post as a comment
                    comments=[{
                        'path': file_path,
                        'line': line_num,  # Use line number here as well
                        'body': full_comment
                    }]
                )
                logger.info(f"Inline comment added successfully (Method 2 - Review) to {file_path}:{line_num}. Review ID: {review.id}")
                return True
            except GithubException as e2:
                logger.error(f"Method 2 (create_review) also failed for {file_path}:{line_num}: {e2.status} {e2.data}")
                if e2.status == 422:
                     logger.error(f"Error 422 suggests line {line_num} in {file_path} might not be part of the diff or file is too large/binary.")
                return False # Both methods failed for this comment
            except Exception as e2:
                logger.error(f"Method 2 (create_review) failed with unexpected error for {file_path}:{line_num}: {str(e2)}")
                return False # Both methods failed

        except Exception as e1_unexpected: # Catch other errors from Method 1
             logger.error(f"Method 1 (create_review_comment) failed with unexpected error for {file_path}:{line_num}: {str(e1_unexpected)}. Trying Method 2.")
             # Duplicate Method 2 attempt here for robustness
             try:
                logger.debug(f"Attempting method 2 after unexpected error in method 1 for {file_path}:{line_num}")
                review = pr.create_review(
                    commit_id=head_sha,
                    body="",
                    event="COMMENT",
                    comments=[{
                        'path': file_path,
                        'line': line_num,
                        'body': full_comment
                    }]
                )
                logger.info(f"Inline comment added successfully (Method 2 - Review after error) to {file_path}:{line_num}. Review ID: {review.id}")
                return True
             except GithubException as e2:
                 logger.error(f"Method 2 (create_review) also failed after unexpected error in method 1 for {file_path}:{line_num}: {e2.status} {e2.data}")
                 if e2.status == 422:
                     logger.error(f"Error 422 suggests line {line_num} in {file_path} might not be part of the diff or file is too large/binary.")
                 return False
             except Exception as e2:
                 logger.error(f"Method 2 (create_review) failed with unexpected error after unexpected error in method 1 for {file_path}:{line_num}: {str(e2)}")
                 return False

    except GithubException as e_setup:
        logger.error(f"Failed to get commit {head_sha} needed for inline comment on {file_path}:{line_num}: {e_setup.status} {e_setup.data}")
        return False
    except Exception as e_setup_unexpected: # Catch other errors during setup
        logger.error(f"Error preparing to add inline comment to {file_path}:{line_num}: {str(e_setup_unexpected)}")
        return False

    # Fallthrough case (should ideally not be reached)
    logger.error(f"Failed to add inline comment to {file_path}:{line_num} after all attempts.")
    return False


def review_code_with_inline_comments(pr):
    """Perform a code review analyzing each file for inline comments"""
    logger.info(f"Performing inline code review analysis for PR #{pr.number}")
    client = get_client()
    
    try:
        logger.info("Fetching files changed in the PR via API")
        files = list(pr.get_files())
        logger.info(f"Found {len(files)} changed files via API")
        
        if not files:
            logger.warning("No files found in PR via API. Skipping inline review analysis.")
            return "No files found in this PR to analyze for inline comments."
            
        file_analysis_results = []
        inline_comments_to_post = []
        
        # Process each file for analysis
        for file in files:
            file_path = file.filename
            file_status = file.status
            logger.info(f"Analyzing file: {file_path} (Status: {file_status})")

            # Skip deleted files, vendor/generated files (optional), potentially binary?
            if file_status == 'removed':
                 logger.info(f"Skipping deleted file: {file_path}")
                 continue
            # Add more skip conditions if needed (e.g., based on path, size)

            if not file.patch:
                logger.info(f"No patch data available for {file_path} via API, skipping analysis")
                file_analysis_results.append(f"### {file_path}\n\nNo patch data available for analysis.")
                continue
                
            # Handle large patches
            max_patch_chars = 20000 # Make configurable?
            if len(file.patch) > max_patch_chars:
                logger.warning(f"Patch for {file_path} is large ({len(file.patch)} chars), truncating for analysis")
                file_patch = file.patch[:max_patch_chars] + f"\n[Patch truncated due to size limit of {max_patch_chars} characters]"
            else:
                file_patch = file.patch
                
            # Prompt for line-specific comments
            prompt = f"""
            {config['prompt']}
            
            Review the following code changes for the file `{file_path}` provided as a patch/diff.
            Focus on identifying specific lines where improvements can be made or issues exist.
            
            ```diff
            {file_patch}
            ```
            
            For each specific issue you find, provide a comment formatted exactly like this:

            LINE: [line number in the file based on the diff]
            COMMENT: [Your detailed comment about the issue on that specific line]

            If you have general comments about the file that aren't tied to a specific line, list them at the end under a heading `### Overall File Comments`.
            If no issues are found, state "No specific issues found in this file."

            Use the line numbers present in the diff context (usually marked with `+` or `-` and preceded by line numbers) to determine the correct line number for your `LINE:` tag. Target the changed lines (`+`) primarily, but commenting on context lines (` `) or deleted lines (`-`) is also acceptable if relevant.
            """
            
            logger.info(f"Sending request to analyze {file_path}")
            try:
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config["max_tokens"], # Consider adjusting based on expected response size
                    temperature=config["temperature"]
                )
                
                analysis = response.choices[0].message.content
                logger.info(f"Received analysis for {file_path} ({len(analysis)} chars)")
                
                # Extract line-specific comments and overall comments from the analysis
                current_file_inline_comments = []
                current_file_overall_comments = []
                
                lines = analysis.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    if line.startswith("LINE:"):
                        try:
                            line_num_str = line.replace("LINE:", "").strip().split(' ')[0] # Handle potential extra text
                            line_num = int(line_num_str)
                            
                            # Look for the corresponding COMMENT line(s)
                            comment_text = ""
                            j = i + 1
                            while j < len(lines) and not lines[j].strip().startswith("LINE:"):
                                if lines[j].strip().startswith("COMMENT:"):
                                    comment_part = lines[j].replace("COMMENT:", "").strip()
                                    comment_text += comment_part
                                    # Consume subsequent lines belonging to the same comment
                                    k = j + 1
                                    while k < len(lines) and not (lines[k].strip().startswith("LINE:") or lines[k].strip().startswith("COMMENT:")):
                                        if lines[k].strip(): # Add non-empty continuation lines
                                            comment_text += "\n" + lines[k].strip()
                                        k += 1
                                    j = k - 1 # Position j correctly for the outer loop
                                    break # Found the comment block for this LINE
                                j += 1
                                
                            if comment_text:
                                logger.debug(f"Parsed comment for {file_path}:{line_num} - '{comment_text[:50]}...'")
                                current_file_inline_comments.append({'file_path': file_path, 'line_num': line_num, 'content': comment_text})
                            else:
                                logger.warning(f"Found 'LINE: {line_num}' but no subsequent 'COMMENT:' for file {file_path}")
                            
                            i = j # Move main loop index forward
                        except ValueError:
                            logger.warning(f"Invalid line number found in analysis for {file_path}: '{line}'. Treating as overall comment.")
                            current_file_overall_comments.append(line)
                        except Exception as parse_err:
                             logger.error(f"Error parsing comment block near '{line}' in analysis for {file_path}: {parse_err}")
                             current_file_overall_comments.append(line) # Add potentially problematic line to overall
                    else:
                        # Collect lines that aren't part of a LINE/COMMENT block as overall comments
                         if not line.startswith("COMMENT:"): # Avoid adding stray COMMENT lines
                             current_file_overall_comments.append(line)
                    i += 1
                
                # Add parsed inline comments to the list to be posted later
                inline_comments_to_post.extend(current_file_inline_comments)

                # Store overall file comments for the final summary
                overall_assessment = "\n".join(current_file_overall_comments).strip()
                if overall_assessment and "No specific issues found" not in overall_assessment:
                     file_analysis_results.append(f"### {file_path}\n\n{overall_assessment}")
                elif not current_file_inline_comments and "No specific issues found" not in overall_assessment:
                     # If no inline comments and no real overall comments, maybe mention it was checked
                     file_analysis_results.append(f"### {file_path}\n\nNo specific issues or comments generated by AI.")

                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path} via API: {str(e)}")
                file_analysis_results.append(f"### {file_path}\n\nError during analysis: {str(e)}")
        
        # Now, post all collected inline comments
        logger.info(f"Attempting to post {len(inline_comments_to_post)} inline comments.")
        successful_inline_comments = 0
        failed_inline_comments = []

        for comment_data in inline_comments_to_post:
            success = post_inline_comments(
                pr, 
                comment_data['file_path'], 
                comment_data['line_num'], 
                comment_data['content']
            )
            if success:
                successful_inline_comments += 1
            else:
                # If inline comment fails, add it to the overall results instead
                failed_inline_comments.append(
                    f"Failed to post inline comment for `{comment_data['file_path']}` line {comment_data['line_num']}:\n```\n{comment_data['content']}\n```"
                )
        
        logger.info(f"Posted {successful_inline_comments} inline comments successfully.")
        
         # Construct the final overall review message
        final_review_parts = []
        if file_analysis_results:
            final_review_parts.append("## AI File Analysis Summary")
            final_review_parts.extend(file_analysis_results)
        
        if failed_inline_comments:
             final_review_parts.append("\n## Failed Inline Comments")
             final_review_parts.append("The following comments could not be placed inline and are included here:")
             final_review_parts.extend(failed_inline_comments)

        if not final_review_parts:
             return "Inline code review analysis complete. No major overall comments generated, check inline comments posted directly on the code."
        else:
             return "\n\n".join(final_review_parts)
            
    except GithubException as e:
         logger.error(f"GitHub API error during inline review file fetch: {e.status} {e.data}")
         return f"Error fetching PR files for inline review: {e.status} {e.data}"
    except Exception as e:
        logger.error(f"Unexpected error during inline code review process: {str(e)}")
        return f"Error performing inline code review analysis: {str(e)}"


def handle_command(pr, command_body, comment_id):
    """Handle commands received in PR comments."""
    prefix = config.get("command_prefix", "/")
    is_ai_review_command = command_body.startswith("/ai-review ")
    is_prefix_command = command_body.startswith(prefix) and not is_ai_review_command

    # Extract action and params based on command format
    action = None
    params = ""
    if is_ai_review_command:
        parts = command_body.split(' ', 2)
        action = parts[1].lower() if len(parts) > 1 else "review" # Default action
        params = parts[2] if len(parts) > 2 else ""
    elif is_prefix_command:
        parts = command_body[len(prefix):].strip().split(' ', 1)
        action = parts[0].lower()
        params = parts[1] if len(parts) > 1 else ""
    else:
         logger.info("Comment does not match command format, ignoring.")
         return # Not a command

    logger.info(f"Handling command: '{action}' with params: '{params}' from comment {comment_id}")
    
    client = get_client() # Ensure client is available

    # --- Command implementations ---
    if action in ["summarize", "summary", "s"]:
        logger.info("Executing summarize command")
        post_comment(pr, "Generating PR summary...") # Ack comment
        summary = summarize_pr(pr)
        post_comment(pr, f"## PR Summary\n\n{summary}") # Post result
        
    elif action in ["review", "r"]:
        logger.info("Executing review command (overall + inline)")
        ack_message = "Starting AI code review (overall analysis and inline comments)... This may take a few minutes."
        post_comment(pr, ack_message) # Ack comment

        # Perform inline analysis and posting
        inline_review_summary = review_code_with_inline_comments(pr)
        
        # Perform overall review based on diff
        overall_review = review_code(pr)

        # Combine results into one comment
        final_comment = "## AI Code Review Complete\n\n"
        if overall_review and not overall_review.startswith("Error"):
             final_comment += "### Overall Analysis (from Diff):\n" + overall_review + "\n\n"
        else:
             final_comment += "### Overall Analysis (from Diff):\n" + (overall_review or "Overall review did not produce output.") + "\n\n"

        if inline_review_summary and not inline_review_summary.startswith("Error"):
            # Append the summary from the inline function (which includes file summaries and failed comments)
             final_comment += inline_review_summary # It already includes headers
        else:
             final_comment += "### Inline Comment Summary:\n" + (inline_review_summary or "Inline analysis did not produce a summary.") + "\n"

        post_comment(pr, final_comment) # Post combined result

    elif action in ["review-inline", "ri"]:
         logger.info("Executing review-inline command")
         post_comment(pr, "Starting AI inline code review analysis...") # Ack
         inline_review_summary = review_code_with_inline_comments(pr)
         post_comment(pr, inline_review_summary or "Inline review analysis complete.") # Post result
    
    elif action in ["review-overall", "ro"]:
         logger.info("Executing review-overall command")
         post_comment(pr, "Starting AI overall code review (based on diff)...") # Ack
         overall_review = review_code(pr)
         post_comment(pr, f"## AI Overall Code Review\n\n{overall_review}") # Post result

    elif action in ["ask", "a", "q", "question"]:
        if not params:
             post_comment(pr, f"Please provide a question after the `{prefix}{action}` command.")
             return
        logger.info(f"Executing ask command with question: {params}")
        post_comment(pr, f"Processing your question: '{params}'...") # Ack

        diff_text = get_pr_diff_text()
        if diff_text is None or diff_text.startswith("[Unable to load diff"):
             post_comment(pr, f"## Error\n\nCould not load PR diff to answer the question. Diff status: {diff_text or 'Not found'}")
             return

        prompt = f"""
        {config['prompt']}
        
        Context: Reviewing changes in a Pull Request.
        Diff of changes:
        ```diff
        {diff_text}
        ```
        
        User Question: {params}
        
        Based on the provided diff and your expertise, please provide a helpful and concise answer to the user's question. If the diff does not provide enough context, say so.
        """
        
        logger.info("Sending ask request to model")
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max(1000, config["max_tokens"] // 4), # Use fewer tokens for answers?
                temperature=config["temperature"]
            )
            answer = response.choices[0].message.content
            logger.info("Successfully received response to question")
            post_comment(pr, f"## Answer to: {params}\n\n{answer}")
        except Exception as e:
            logger.error(f"Error processing ask command via API: {str(e)}")
            post_comment(pr, f"## Error\n\nSorry, there was an error processing your question: {str(e)}")
        
    elif action in ["clean", "clear", "c"]:
        logger.info("Executing clean command")
        post_comment(pr, "Cleaning up AI review comments...") # Ack
        
        tag_to_find = f"<!-- {config['comment_tag']} -->"
        deleted_issue_comments = 0
        deleted_review_comments = 0
        
        # Delete general issue comments
        try:
            comments = pr.get_issue_comments()
            for comment in comments:
                if tag_to_find in comment.body:
                    logger.debug(f"Deleting issue comment {comment.id}")
                    try:
                        comment.delete()
                        deleted_issue_comments += 1
                    except GithubException as e_del:
                        logger.error(f"Error deleting issue comment {comment.id}: {e_del.status} {e_del.data}")
                    except Exception as e_del:
                         logger.error(f"Unexpected error deleting issue comment {comment.id}: {str(e_del)}")
        except GithubException as e_fetch:
             logger.error(f"Error fetching issue comments: {e_fetch.status} {e_fetch.data}")
        except Exception as e_fetch:
             logger.error(f"Unexpected error fetching issue comments: {str(e_fetch)}")

        # Delete inline review comments (these are PullRequestComment objects)
        try:
            review_comments = pr.get_comments() # Gets PullRequestComment objects
            for comment in review_comments:
                if tag_to_find in comment.body:
                    logger.debug(f"Deleting review comment {comment.id}")
                    try:
                        comment.delete()
                        deleted_review_comments += 1
                    except GithubException as e_del:
                        logger.error(f"Error deleting review comment {comment.id}: {e_del.status} {e_del.data}")
                    except Exception as e_del:
                         logger.error(f"Unexpected error deleting review comment {comment.id}: {str(e_del)}")
        except GithubException as e_fetch:
             logger.error(f"Error fetching review comments: {e_fetch.status} {e_fetch.data}")
        except Exception as e_fetch:
             logger.error(f"Unexpected error fetching review comments: {str(e_fetch)}")
        
        total_deleted = deleted_issue_comments + deleted_review_comments
        logger.info(f"Cleaned up {total_deleted} comments ({deleted_issue_comments} issue, {deleted_review_comments} review).")
        post_comment(pr, f"AI review comments cleaned up. {total_deleted} comments were removed.")
        
    elif action in ["help", "h", "?"]:
        logger.info("Executing help command")
        help_text = f"""
        ## AI Code Reviewer Help

        You can interact with the AI reviewer by adding comments to the Pull Request. Start your comment with `{prefix}` followed by a command.

        **Available Commands:**

        *   `{prefix}review` or `{prefix}r`: (Default) Performs a full review, including overall analysis based on the diff and posting inline comments for specific suggestions.
        *   `{prefix}summary` or `{prefix}s`: Generates a concise summary of the PR's purpose and key changes based on the title, description, and diff.
        *   `{prefix}review-inline` or `{prefix}ri`: Analyzes each changed file and posts suggestions as inline comments directly on the relevant lines of code.
        *   `{prefix}review-overall` or `{prefix}ro`: Provides an overall review feedback based on the complete diff of the PR.
        *   `{prefix}ask [your question]` or `{prefix}q [your question]`: Ask a specific question about the code changes in the PR. The AI will use the diff as context.
        *   `{prefix}clean` or `{prefix}c`: Removes all comments previously posted by this AI reviewer (identified by a hidden tag).
        *   `{prefix}help` or `{prefix}h`: Displays this help message.

        **Note:** Commands are processed based on comments. If you push new changes, you might want to run `{prefix}review` again.

        *(You can also use the older format: `/ai-review [command] [params]`)*
        """
        post_comment(pr, help_text)
        
    else:
        logger.warning(f"Unknown command received: '{action}'")
        post_comment(pr, f"Sorry, I don't understand the command '{action}'. Try `{prefix}help` for available commands.")


def main():
    logger.info("Starting main execution flow")
    try:
        if event_name == "pull_request" or event_name == "pull_request_target": # Handle both trigger types
            event_path = os.environ.get("GITHUB_EVENT_PATH")
            if not event_path or not os.path.exists(event_path):
                 logger.error("GITHUB_EVENT_PATH is missing or invalid. Cannot determine PR number.")
                 sys.exit(1)

            try:
                 with open(event_path, 'r') as f:
                     event_data = json.load(f)
                 action = event_data.get('action')
                 pr_data = event_data.get('pull_request')
                 if not pr_data:
                      logger.error("No 'pull_request' data found in the event payload.")
                      sys.exit(1)
                 pr_number = pr_data.get('number')
                 if pr_number is None:
                     logger.error("Could not extract PR number from event payload.")
                     sys.exit(1)

                 logger.info(f"Processing event '{event_name}', action: '{action}' for PR #{pr_number}")
                 pr = repo.get_pull(pr_number)

                 # Decide whether to run review automatically based on action
                 # e.g., run on 'opened', 'reopened', 'synchronize'
                 auto_review_actions = ['opened', 'reopened', 'synchronize']
                 if action in auto_review_actions:
                     logger.info(f"Action '{action}' triggers automatic review.")
                     
                     # Check if diff exists before proceeding
                     if not check_diff_file():
                          logger.warning("Diff file is missing or empty. Skipping automatic review. Bot can still be triggered by commands.")
                          # Optionally post a comment indicating the issue
                          # post_comment(pr, "AI Reviewer: Could not find code changes (diff) to review automatically.")
                          return # Exit gracefully

                     # Post Ack comment
                     post_comment(pr, "AI Reviewer preparing initial review (summary + inline comments)...")
                     
                     # Run summary and inline review
                     summary = summarize_pr(pr)
                     inline_review_summary = review_code_with_inline_comments(pr) # This posts inline comments directly

                     # Post combined initial comment
                     initial_comment = f"## Initial AI Review\n\n### PR Summary\n{summary}\n\n"
                     initial_comment += (inline_review_summary or "Inline review analysis complete.") # Append summary from inline function
                     post_comment(pr, initial_comment)

                 else:
                      logger.info(f"Action '{action}' does not trigger automatic review. Waiting for commands.")

            except FileNotFoundError:
                 logger.error(f"Event payload file not found at: {event_path}")
                 sys.exit(1)
            except json.JSONDecodeError:
                 logger.error(f"Failed to parse JSON from event payload at: {event_path}")
                 sys.exit(1)
            except KeyError as e:
                 logger.error(f"Missing expected key '{e}' in event payload.")
                 sys.exit(1)

        elif event_name == "issue_comment":
            event_path = os.environ.get("GITHUB_EVENT_PATH")
            if not event_path or not os.path.exists(event_path):
                 logger.error("GITHUB_EVENT_PATH is missing or invalid. Cannot process comment.")
                 sys.exit(1)

            try:
                 with open(event_path, 'r') as f:
                     event_data = json.load(f)

                 # Check if the comment is on a PR
                 if 'issue' not in event_data or 'pull_request' not in event_data['issue']:
                      logger.info("Comment is not on a Pull Request. Ignoring.")
                      return

                 # Check if the action is 'created' (ignore edits, deletions)
                 action = event_data.get('action')
                 if action != 'created':
                      logger.info(f"Ignoring comment action '{action}'. Only processing 'created' comments.")
                      return

                 comment_data = event_data.get('comment')
                 issue_data = event_data.get('issue')
                 if not comment_data or not issue_data:
                      logger.error("Missing 'comment' or 'issue' data in payload.")
                      return # Or sys.exit(1) depending on strictness

                 pr_number = issue_data.get('number')
                 comment_id = comment_data.get('id')
                 comment_body = comment_data.get('body')
                 comment_user = comment_data.get('user', {}).get('login')

                 if pr_number is None or comment_id is None or comment_body is None:
                     logger.error("Missing PR number, comment ID, or body in comment event payload.")
                     return

                 # Prevent bot feedback loops
                 bot_user = g.get_user().login # Get the login of the user associated with GITHUB_TOKEN
                 if comment_user == bot_user:
                      logger.info(f"Ignoring comment {comment_id} from the bot itself ({comment_user}).")
                      return

                 logger.info(f"Processing comment {comment_id} by '{comment_user}' on PR #{pr_number}")
                 
                 # Check if it's a command
                 prefix = config.get("command_prefix", "/")
                 if comment_body.startswith(prefix) or comment_body.startswith("/ai-review "):
                      logger.info("Command detected. Handling...")
                      pr = repo.get_pull(pr_number)

                      # Ensure diff file is available if needed for the command
                      # Commands needing diff: review, r, review-overall, ro, ask, a, q
                      needs_diff = any(cmd in action for cmd in ["review", "r", "review-overall", "ro", "ask", "a", "q"])
                      if needs_diff and not os.path.exists('pr_diff.txt'):
                           # Attempt to fetch diff on demand (this might be slow/complex in action env)
                           # For now, rely on the checkout step providing it.
                           logger.warning("Command requires diff, but pr_diff.txt not found. Results may be inaccurate or fail.")
                           # You could add the requests based fetch here if desired, similar to original code

                      handle_command(pr, comment_body, comment_id)
                 else:
                      logger.info("Comment is not a command, ignoring.")

            except FileNotFoundError:
                 logger.error(f"Event payload file not found at: {event_path}")
                 sys.exit(1)
            except json.JSONDecodeError:
                 logger.error(f"Failed to parse JSON from event payload at: {event_path}")
                 sys.exit(1)
            except KeyError as e:
                 logger.error(f"Missing expected key '{e}' in comment event payload.")
                 sys.exit(1)
            except UnknownObjectException:
                 logger.error(f"Could not find PR #{pr_number}. It might be closed or inaccessible.")
                 # Don't exit, just log error for this event
            except GithubException as e:
                 logger.error(f"GitHub API error processing comment event: {e.status} {e.data}")
                 # Don't exit, just log error for this event

        else:
            logger.warning(f"Unsupported event type: {event_name}. This script primarily handles 'pull_request', 'pull_request_target', and 'issue_comment'.")

    except UnknownObjectException as e:
         logger.error(f"Failed to get a GitHub object (e.g., PR, repo): {e.status} {e.data}. Check repo name and permissions.")
         sys.exit(1)
    except GithubException as e:
        logger.error(f"Unhandled GitHub API error in main: {e.status} {e.data}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logger.info(f"Script execution started for event: {os.environ.get('GITHUB_EVENT_NAME')}")
    main()
    logger.info("Script execution finished.")
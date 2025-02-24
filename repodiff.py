import subprocess
import sys
import os
import json
import tiktoken
import argparse
import tempfile
import re
from typing import Dict, List, Optional, Tuple

# Tokenizer function using OpenAI's tiktoken for LLMs (GPT-3/4)
def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to execute the git diff command and return the result
def run_git_diff(commit1, commit2):
    try:
        result = subprocess.run(
            ["git", "diff", commit1, commit2, "--unified=999999", "--ignore-all-space"],
            capture_output=True, text=True, check=True, encoding='utf-8', errors='replace'
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}")
        sys.exit(1)

def parse_unified_diff(diff_output: str) -> Dict[str, List[Dict]]:
    """Parse the unified diff output into a dictionary of files and their hunks."""
    files = {}
    current_file = None
    current_hunks = []
    lines = diff_output.split('\n')
    
    for line in lines:
        if line.startswith('diff --git'):
            if current_file:
                files[current_file] = current_hunks
            current_file = None
            current_hunks = []
        elif line.startswith('--- a/'):
            continue
        elif line.startswith('+++ b/'):
            current_file = line[6:]  # Remove '+++ b/' prefix
        elif line.startswith('@@'):
            # Parse hunk header
            match = re.match(r'@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@', line)
            if match:
                old_start = int(match.group(1))
                old_count = int(match.group(2) or 1)
                new_start = int(match.group(3))
                new_count = int(match.group(4) or 1)
                current_hunks.append({
                    'header': line,
                    'old_start': old_start,
                    'old_count': old_count,
                    'new_start': new_start,
                    'new_count': new_count,
                    'lines': []
                })
        elif current_file and current_hunks:
            current_hunks[-1]['lines'].append(line)
    
    if current_file:
        files[current_file] = current_hunks
    
    return files

def find_matching_rule(filename: str, filters: List[Dict]) -> Dict:
    """Find the first matching filter rule for a filename."""
    import fnmatch
    
    for filter_rule in filters:
        if fnmatch.fnmatch(filename, filter_rule['file_pattern']):
            return filter_rule
    return {'context_lines': 3}  # Default rule

def apply_context_filter(hunks: List[Dict], context_lines: int) -> List[Dict]:
    """Adjust the context lines in hunks to match the specified number."""
    filtered_hunks = []
    
    for hunk in hunks:
        lines = hunk['lines']
        filtered_lines = []
        change_indices = []
        
        # First, find all the changed lines (+ or -)
        for i, line in enumerate(lines):
            if line.startswith('+') or line.startswith('-'):
                change_indices.append(i)
        
        if not change_indices:
            continue
        
        # Now determine which context lines to keep
        lines_to_keep = set()
        for change_idx in change_indices:
            # Add the changed line
            lines_to_keep.add(change_idx)
            # Add context lines before
            for i in range(max(0, change_idx - context_lines), change_idx):
                lines_to_keep.add(i)
            # Add context lines after
            for i in range(change_idx + 1, min(len(lines), change_idx + context_lines + 1)):
                lines_to_keep.add(i)
        
        # Keep lines in their original order
        for i, line in enumerate(lines):
            if i in lines_to_keep:
                filtered_lines.append(line)
        
        if filtered_lines:
            new_hunk = hunk.copy()
            new_hunk['lines'] = filtered_lines
            filtered_hunks.append(new_hunk)
    
    return filtered_hunks

def apply_signature_removal(hunks: List[Dict], threshold: int) -> List[Dict]:
    """Process C# file hunks to keep method signatures but replace bodies with { ... }."""
    processed_hunks = []
    
    for hunk in hunks:
        lines = hunk['lines']
        processed_lines = []
        in_method = False
        brace_count = 0
        method_lines = []
        
        for line in lines:
            if not line.strip():
                processed_lines.append(line)
                continue
                
            # Check for method signature
            if re.match(r'^\s*(public|private|protected|internal|\[|\s)*\s*\w+\s+\w+\s*\(', line):
                if in_method:
                    # Handle nested methods by keeping the outer one
                    processed_lines.extend(method_lines)
                in_method = True
                brace_count = 0
                method_lines = [line]
                continue
            
            if in_method:
                method_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0:
                    # Method ended
                    if len(method_lines) <= threshold:
                        processed_lines.extend(method_lines)
                    else:
                        # Replace method body with { ... }
                        processed_lines.append(method_lines[0])  # Signature
                        processed_lines.append('    { ... }')
                        processed_lines.append('}')
                    in_method = False
                    method_lines = []
            else:
                processed_lines.append(line)
        
        if processed_lines:
            new_hunk = hunk.copy()
            new_hunk['lines'] = processed_lines
            processed_hunks.append(new_hunk)
    
    return processed_hunks

def post_process_files(patch_dict: Dict[str, List[Dict]], config: Dict) -> Dict[str, List[Dict]]:
    """Apply post-processing filters to each file in the patch."""
    processed_dict = {}
    
    for filename, hunks in patch_dict.items():
        rule = find_matching_rule(filename, config['filters'])
        
        if rule.get('include_entire_file_with_signatures', False):
            threshold = rule.get('method_body_threshold', 10)
            processed_dict[filename] = apply_signature_removal(hunks, threshold)
        else:
            context_lines = rule.get('context_lines', 3)
            processed_dict[filename] = apply_context_filter(hunks, context_lines)
    
    return processed_dict

def reconstruct_patch(patch_dict: Dict[str, List[Dict]]) -> str:
    """Reconstruct a unified diff from the processed patch dictionary."""
    output = []
    
    # Add header explaining placeholders
    output.append('NOTE: Some method bodies have been replaced with "{ ... }" to improve clarity for code reviews and LLM analysis.\n')
    
    for filename, hunks in patch_dict.items():
        output.append(f'diff --git a/{filename} b/{filename}')
        output.append(f'--- a/{filename}')
        output.append(f'+++ b/{filename}')
        
        for hunk in hunks:
            output.append(hunk['header'])
            output.extend(hunk['lines'])
    
    return '\n'.join(output)

# Function to load config (diff options and tiktoken model) from JSON config file
def load_config(config_file_name="config.json"):
    # Check if running as a PyInstaller bundle (frozen)
    if getattr(sys, 'frozen', False):
        # If the application is run as a PyInstaller bundle, use the _MEIPASS directory
        script_dir = os.path.dirname(sys.executable)
    else:
        # If running in a normal Python environment, use the script directory
        script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the path to config.json
    config_path = os.path.join(script_dir, config_file_name)
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    else:
        print(f"Config file '{config_file_name}' not found in script directory '{script_dir}'.")
        sys.exit(1)

# Function to get the latest commit hash for the current branch
def get_latest_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, encoding='utf-8', errors='replace'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting the latest commit: {e}")
        sys.exit(1)

# Function to get the latest common commit between the current branch and base branch
def get_latest_common_commit_with_branch(branch):
    try:
        result = subprocess.run(
            ["git", "merge-base", "HEAD", branch],
            capture_output=True, text=True, check=True, encoding='utf-8', errors='replace'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting the latest common commit with '{branch}': {e}")
        sys.exit(1)

# Main function to generate the combined diff and calculate token count
def main(commit1, commit2, output_file):
    # Load the config from the default or specified path
    config = load_config()
    
    # Extract tiktoken model from the config
    tiktoken_model = config.get("tiktoken_model", "gpt-4o")
    
    # Get the raw diff output
    raw_diff = run_git_diff(commit1, commit2)
    
    # Parse and process the diff
    patch_dict = parse_unified_diff(raw_diff)
    processed_dict = post_process_files(patch_dict, config)
    final_output = reconstruct_patch(processed_dict)
    
    # Write the processed diff to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output)
    
    # Calculate token count using the tiktoken model
    token_count = count_tokens(final_output, tiktoken_model)
    
    # Output results
    print(f"Processed diff written to {output_file}")
    print(f"Total number of tokens: {token_count}")

# Entry point of the script
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run git diff between two commits and analyze with LLM.")
    parser.add_argument("-o", "--output_file", help="The file to output the combined diff.")
    parser.add_argument("-c1", "--commit1", help="The first commit hash.")
    parser.add_argument("-c2", "--commit2", help="The second commit hash.")
    parser.add_argument("-b", "--branch", help="Compare the latest commit on the current branch to the latest common commit with another branch (e.g., master).")

    args = parser.parse_args()

    # Determine the commit hashes
    if args.branch:
        commit1 = get_latest_common_commit_with_branch(args.branch)
        commit2 = get_latest_commit()

        # Print the commits being used for the comparison
        print(f"Comparing latest common commit with branch '{args.branch}' ({commit1[:12]}) and the latest commit on the current branch ({commit2[:12]}).")
    else:
        if not args.commit1 or not args.commit2:
            print("You must either provide two commit hashes using --commit1 and --commit2, or use the -b option to compare against another branch.")
            sys.exit(1)
        commit1 = args.commit1
        commit2 = args.commit2

    # Set output file or default to the user's temporary directory
    if args.output_file:
        output_file = args.output_file
    else:
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, "repodiff", "repodiff_output.txt")
        print(f"No output file specified. Using temporary directory: {output_file}")

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    main(commit1, commit2, output_file)

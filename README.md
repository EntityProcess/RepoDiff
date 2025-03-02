# RepoDiff

**RepoDiff** is a tool designed to simplify code reviews by generating dynamic git diffs between two commits or branches. It optimizes diffs for analysis by large language models (LLMs) with features like context line adjustment and method body removal.

## Features

- Generate diffs between two commits or branches with a single pass
- Configurable file pattern matching for different file types
- Smart method body removal for C# files to improve readability
- Adjustable context lines per file pattern
- Token counting for estimating LLM query costs
- Combines all changes into a single, well-formatted output

## Installation

### Option 1: Download the Executable

1. Go to the [Releases](https://github.com/EntityProcess/RepoDiff/releases) page.
2. Download the latest version of the `repodiff.exe` executable.
3. Move the `repodiff.exe` file to a directory included in your system's `PATH`.

### Option 2: Install from Source

```bash
git clone https://github.com/EntityProcess/RepoDiff.git
cd RepoDiff
pip install -e .
```

### Option 3: Build the Executable Yourself

Clone the repository and navigate to the directory:

```bash
git clone https://github.com/EntityProcess/RepoDiff.git
cd RepoDiff
```

Install PyInstaller and build the executable:

```bash
pip install pyinstaller
# On Windows, run:
build.bat
```

Add `./RepoDiff/dist` to your `PATH` environmental variable.

## Usage

### Compare Latest Commit with Another Branch

To compare the latest commit in the current branch with the latest common commit in another branch:

```bash
repodiff -b main -o output.txt
```

### Compare Two Specific Commits

```bash
repodiff -c1 abcdef1234567890 -c2 0987654321fedcba -o output.txt
```

Parameters:
* `-b`, `--branch`: Branch to compare with (e.g., `main` or `master`)
* `-c1`, `--commit1`: First commit hash
* `-c2`, `--commit2`: Second commit hash
* `-o`, `--output_file`: (Optional) Path to the output file. If not provided, the diff will be written to a default file in the system's temporary directory.
* `--version`: Display the current version of RepoDiff

## Configuration

RepoDiff uses a `config.json` file in the project root directory. Example configuration:

```json
{
  "tiktoken_model": "gpt-4o",
  "filters": [
    {
      "file_pattern": "*.cs",
      "include_entire_file_with_signatures": true,
      "method_body_threshold": 10
    },
    {
      "file_pattern": "*Test*.cs",
      "context_lines": 20
    },
    {
      "file_pattern": "*.xml",
      "context_lines": 5
    },
    {
      "file_pattern": "*",
      "context_lines": 3
    }
  ]
}
```

Configuration options:

* `tiktoken_model`: Specifies the language model for token counting (e.g., "gpt-4o").
* `filters`: An array of filter rules that determine how different files are processed.
  * `file_pattern`: Glob pattern to match files (e.g., "*.cs", "*Test*.cs").
  * `include_entire_file_with_signatures`: (Optional) When true, keeps method signatures but replaces large method bodies with `{ ... }`.
  * `method_body_threshold`: (Optional) Maximum number of lines in a method before its body is replaced with `{ ... }`.
  * `context_lines`: (Optional) Number of context lines to show around changes (default: 3).

Filter rules are applied in order, with the first matching pattern being used.

## Output Format

The tool generates a unified diff format with some enhancements:

1. A header explaining any placeholders used (e.g., `{ ... }` for removed method bodies).
2. Standard git diff headers for each file.
3. Modified hunks based on the applied filters:
   - Adjusted context lines
   - Method bodies replaced with `{ ... }` where applicable
   - Original line numbers preserved

Example output:

```diff
NOTE: Some method bodies have been replaced with "{ ... }" to improve clarity for code reviews and LLM analysis.

diff --git a/src/MyClass.cs b/src/MyClass.cs
--- a/src/MyClass.cs
+++ b/src/MyClass.cs
@@ -10,7 +10,7 @@ public class MyClass
     public void ProcessData(int value)
     {
         { ... }
    }
```

## Prerequisites

- **PowerShell (Windows Only)**: If you're using Windows, you need to run the script in PowerShell. The pattern matching functionality in the script will not work properly in Command Prompt (`cmd`).
- **Python 3.x**: Ensure Python is installed on your system.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_filters.py

# Run with coverage
pytest --cov=repodiff
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.
#!/bin/bash

# Script to convert example directories to separate GitHub repositories and add them as submodules
# Prerequisites: GitHub CLI (gh) must be installed and authenticated

set -e

# Configuration
MAIN_REPO="nicewebrl"
GITHUB_USER=$(gh api user --jq .login)
EXAMPLES_DIR="examples"

# Check if GitHub CLI is authenticated
if ! gh auth status &>/dev/null; then
    echo "Error: GitHub CLI is not authenticated. Run 'gh auth login' first."
    exit 1
fi

# Get list of example directories
EXAMPLE_DIRS=($(ls -d examples/*/))

echo "Found ${#EXAMPLE_DIRS[@]} example directories:"
for dir in "${EXAMPLE_DIRS[@]}"; do
    echo "  - $dir"
done

echo ""
read -p "Do you want to proceed with creating repositories for these examples? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Function to sanitize sensitive files
sanitize_sensitive_files() {
    local temp_dir="$1"
    
    # Remove sensitive files
    rm -f "$temp_dir/google-cloud-key.json"
    rm -rf "$temp_dir/__pycache__"
    rm -rf "$temp_dir/data/"
    rm -rf "$temp_dir/craftax_cache/"
    rm -rf "$temp_dir/model_params/"
    
    # Create .gitignore
    cat > "$temp_dir/.gitignore" << 'EOF'
# Sensitive configuration
config.py
google-cloud-key.json
*.env
.env*

# Data and cache
data/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
craftax_cache/
model_params/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
EOF
    
    # Handle config files - create template versions
    if [[ -f "$temp_dir/config.py" ]]; then
        cat > "$temp_dir/config_template.py" << 'EOF'
"""
Configuration file for API keys and other settings.
Copy this file to config.py and fill in your actual API keys.
"""

# API Keys - Replace with your actual keys
GEMINI_API_KEY = "your-gemini-api-key-here"
CLAUDE_API_KEY = "your-claude-api-key-here"
CHATGPT_API_KEY = "your-openai-api-key-here"

# API Endpoints
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"

# Model Settings
CLAUDE_MODEL = "claude-3-opus-20240229"
CHATGPT_MODEL = "gpt-3.5-turbo" 

# Google Cloud Storage
BUCKET_NAME = "your-bucket-name"
GOOGLE_CREDENTIALS = "./google-cloud-key.json"

# Data
DATA_DIR = "./user_data"
EOF
        rm "$temp_dir/config.py"
    fi
}

# Function to create repository for an example
create_example_repo() {
    local example_dir="$1"
    local example_name=$(basename "$example_dir")
    local repo_name="${MAIN_REPO}-example-${example_name}"
    
    echo "Processing: $example_name"
    
    # Create temporary directory for the new repo
    local temp_dir="/tmp/${repo_name}"
    rm -rf "$temp_dir"
    mkdir -p "$temp_dir"
    
    # Copy example files to temp directory
    cp -r "$example_dir"* "$temp_dir/"
    
    # Sanitize sensitive files
    sanitize_sensitive_files "$temp_dir"
    
    # Create README if it doesn't exist
    if [[ ! -f "$temp_dir/README.md" ]]; then
        cat > "$temp_dir/README.md" << EOF
# ${example_name^} Example

This is an example from the nicewebrl framework demonstrating ${example_name} functionality.

## Installation

From the main nicewebrl repository:
\`\`\`bash
pip install -e ".[${example_name}]"
\`\`\`

## Setup

If this example requires API keys, copy \`config_template.py\` to \`config.py\` and fill in your API keys.

## Running

\`\`\`bash
python web_app.py
\`\`\`

For more details, see the main [nicewebrl repository](https://github.com/${GITHUB_USER}/${MAIN_REPO}).
EOF
    fi
    
    # Initialize git repo in temp directory
    cd "$temp_dir"
    git init
    git add .
    git commit -m "Initial commit: ${example_name} example from nicewebrl

Extracted from main nicewebrl repository to enable independent development
and easier distribution as a standalone example.

🤖 Generated with Claude Code"
    
    # Check if repository already exists
    if gh repo view "$GITHUB_USER/$repo_name" &>/dev/null; then
        echo "⚠️  Repository $repo_name already exists. Skipping creation."
        cd - > /dev/null
        rm -rf "$temp_dir"
        echo "$repo_name"
        return
    fi
    
    # Create GitHub repository
    echo "Creating GitHub repository: $repo_name"
    gh repo create "$repo_name" --public --description "nicewebrl example: ${example_name}" --source .
    
    # Push to GitHub
    git branch -M main
    git push -u origin main
    
    echo "✅ Created repository: https://github.com/${GITHUB_USER}/${repo_name}"
    
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    echo "$repo_name"
}

# Array to store created repository names
created_repos=()

# Create repositories for each example
for example_dir in "${EXAMPLE_DIRS[@]}"; do
    example_name=$(basename "$example_dir")
    
    echo ""
    echo "===================="
    echo "Creating repo for: $example_name"
    echo "===================="
    
    repo_name=$(create_example_repo "$example_dir")
    created_repos+=("$repo_name")
done

echo ""
echo "===================="
echo "PHASE 2: Converting to submodules"
echo "===================="

# Now remove original directories and add as submodules
for i in "${!EXAMPLE_DIRS[@]}"; do
    example_dir="${EXAMPLE_DIRS[$i]}"
    repo_name="${created_repos[$i]}"
    example_name=$(basename "$example_dir")
    
    echo "Converting $example_name to submodule..."
    
    # Remove original directory from git tracking
    git rm -rf "$example_dir"
    
    # Add as submodule
    git submodule add "https://github.com/${GITHUB_USER}/${repo_name}.git" "$example_dir"
done

# Commit the submodule changes
git add .gitmodules
git commit -m "Convert examples to submodules

- Moved individual examples to separate repositories
- Added examples as git submodules for easier maintenance
- Each example can now be developed independently

Created repositories:
$(printf '%s\n' "${created_repos[@]}" | sed 's/^/- /')

🤖 Generated with Claude Code"

echo ""
echo "✅ Successfully converted all examples to submodules!"
echo ""
echo "Created repositories:"
for repo in "${created_repos[@]}"; do
    echo "  - https://github.com/${GITHUB_USER}/${repo}"
done

echo ""
echo "Next steps:"
echo "1. Push the main repository changes: git push"
echo "2. Users can clone with submodules: git clone --recurse-submodules <repo-url>"
echo "3. To update submodules: git submodule update --remote"
echo "4. For examples with API keys, users need to copy config_template.py to config.py and add their keys"
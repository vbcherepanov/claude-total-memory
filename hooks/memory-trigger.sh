#!/usr/bin/env bash
#
# PostToolUse:Bash hook — suggest memory_save after significant operations
#
# Triggers on: git commit, docker, migrations, package installs, builds
#
# Add to ~/.claude/settings.json:
#   "hooks": {
#     "PostToolUse": [{
#       "type": "command",
#       "command": "/path/to/claude-total-memory/hooks/memory-trigger.sh",
#       "matcher": "Bash"
#     }]
#   }

# Read tool input from stdin (JSON)
INPUT=$(cat)

# Extract the command that was run
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('input',{}).get('command',''))" 2>/dev/null)

if [ -z "$COMMAND" ]; then
    exit 0
fi

# Check for significant operations
case "$COMMAND" in
    *"git commit"*)
        echo "MEMORY_HINT: Git commit detected. Consider saving the commit scope and key changes with memory_save(type='fact')."
        ;;
    *"docker compose up"*|*"docker-compose up"*|*"docker build"*)
        echo "MEMORY_HINT: Docker operation detected. Consider saving infrastructure config with memory_save(type='fact')."
        ;;
    *"migrate"*|*"migration"*)
        echo "MEMORY_HINT: Database migration detected. Consider saving schema changes with memory_save(type='fact')."
        ;;
    *"make setup"*|*"make init"*|*"make deploy"*)
        echo "MEMORY_HINT: Project setup/build detected. Consider saving infrastructure facts with memory_save(type='fact')."
        ;;
    *"npm install"*|*"npm ci"*|*"yarn add"*|*"pnpm add"*)
        echo "MEMORY_HINT: Package install detected. Consider saving dependency changes with memory_save(type='fact')."
        ;;
    *"pip install"*|*"poetry add"*|*"pipenv install"*)
        echo "MEMORY_HINT: Python package install detected. Consider saving dependency changes with memory_save(type='fact')."
        ;;
    *"go mod"*|*"go get"*)
        echo "MEMORY_HINT: Go module change detected. Consider saving dependency changes with memory_save(type='fact')."
        ;;
    *"cargo build"*|*"cargo add"*)
        echo "MEMORY_HINT: Rust cargo operation detected. Consider saving build/dependency facts with memory_save(type='fact')."
        ;;
    *"composer require"*|*"composer install"*)
        echo "MEMORY_HINT: Composer operation detected. Consider saving dependency changes with memory_save(type='fact')."
        ;;
esac

#!/usr/bin/env bash
#
# Codex CLI notify hook — reminder to save knowledge after agent turns
#
# This hook is OPTIONAL. Codex hooks are experimental.
# Memory works without hooks — AGENTS.md instructions are sufficient.
#
# Add to ~/.codex/config.toml:
#   notify = ["/path/to/claude-total-memory/hooks/codex-notify.sh"]

# Read JSON payload from stdin
INPUT=$(cat 2>/dev/null || echo "{}")

# Detect project from cwd in payload
CWD=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cwd',''))" 2>/dev/null || echo "")
if [ -n "$CWD" ]; then
    PROJECT=$(basename "$CWD")
else
    PROJECT=$(basename "$(pwd)" 2>/dev/null || echo "unknown")
fi

# macOS notification
if [ "$(uname)" = "Darwin" ]; then
    osascript -e "display notification \"Remember: memory_save & self_reflect\" with title \"Total Memory — $PROJECT\"" 2>/dev/null
fi

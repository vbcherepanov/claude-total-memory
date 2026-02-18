#!/usr/bin/env bash
#
# PostToolUse:Write,Edit hook — suggest memory_observe after file changes
#
# Add to ~/.claude/settings.json:
#   "hooks": {
#     "PostToolUse": [
#       {
#         "type": "command",
#         "command": "/path/to/claude-total-memory/hooks/auto-capture.sh",
#         "matcher": "Write"
#       },
#       {
#         "type": "command",
#         "command": "/path/to/claude-total-memory/hooks/auto-capture.sh",
#         "matcher": "Edit"
#       }
#     ]
#   }

# Read tool input from stdin (JSON)
INPUT=$(cat)

# Extract tool name and file path
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_name',''))" 2>/dev/null)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('input',{}).get('file_path',''))" 2>/dev/null)

if [ -z "$TOOL_NAME" ] || [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Only hint for non-trivial files
case "$FILE_PATH" in
    *.md|*.txt|*.log|*.json|*.yaml|*.yml|*.toml|*.lock|*.sum)
        # Skip config/doc files — usually not worth observing
        exit 0
        ;;
esac

FILENAME=$(basename "$FILE_PATH")
echo "MEMORY_HINT: File changed: ${FILENAME}. Consider: memory_observe(tool_name=\"${TOOL_NAME}\", summary=\"Modified ${FILENAME}\", files_affected=[\"${FILE_PATH}\"])"

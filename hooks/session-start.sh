#!/usr/bin/env bash
#
# SessionStart hook — remind Claude to use memory_recall + self_rules_context
#
# Add to ~/.claude/settings.json:
#   "hooks": {
#     "SessionStart": [{
#       "type": "command",
#       "command": "/path/to/claude-total-memory/hooks/session-start.sh"
#     }]
#   }

# Detect project and branch
PROJECT=$(basename "$(pwd)" 2>/dev/null || echo "unknown")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

HINT="MEMORY_HINT: Project: ${PROJECT}"
if [ -n "$BRANCH" ]; then
    HINT="${HINT}, Branch: ${BRANCH}"
fi
HINT="${HINT}. Use memory_recall(query=\"your task\", project=\"${PROJECT}\") to search past knowledge."
HINT="${HINT} Also run self_rules_context(project=\"${PROJECT}\") to load behavioral rules."

echo "$HINT"

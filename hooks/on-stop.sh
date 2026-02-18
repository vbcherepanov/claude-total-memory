#!/usr/bin/env bash
#
# Stop hook — remind Claude to save knowledge and reflect when session ends
#
# Add to ~/.claude/settings.json:
#   "hooks": {
#     "Stop": [{
#       "type": "command",
#       "command": "/path/to/claude-total-memory/hooks/on-stop.sh"
#     }]
#   }

PROJECT=$(basename "$(pwd)" 2>/dev/null || echo "unknown")

echo "MEMORY_WARNING: Session ending. Before closing:"
echo "  1. Save important knowledge with memory_save(project=\"${PROJECT}\")"
echo "  2. Record a reflection: self_reflect(reflection=\"...\", task_summary=\"...\", project=\"${PROJECT}\")"

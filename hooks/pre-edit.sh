#!/usr/bin/env bash
# ===========================================
# PreToolUse hook for Write|Edit - file_context guard
#
# Emits a reminder to call file_context(path) before editing a file.
# The agent then calls the MCP tool and reads warnings/risk_score.
#
# Hook: PreToolUse (matcher: "Write|Edit")
# ===========================================

source "$(dirname "$0")/lib/common.sh"

TOOL=$(hook_get 'tool_name')
FILE_PATH=$(hook_get 'tool_input.file_path')

case "$TOOL" in
    Write|Edit) ;;
    *) exit 0 ;;
esac

[ -z "$FILE_PATH" ] && exit 0

case "$FILE_PATH" in
    */.git/*|*/node_modules/*|*/.venv/*|/tmp/*) exit 0 ;;
esac

cat <<EOF
<system-reminder>
Memory pre-edit guard: before editing \`$FILE_PATH\`, call
  file_context(path="$FILE_PATH")
If risk_score > 0.3, read the returned warnings and incorporate them into the edit.
Skip if file_context was already called for this path in the current turn.
</system-reminder>
EOF

exit 0

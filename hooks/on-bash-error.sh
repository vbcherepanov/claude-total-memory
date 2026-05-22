#!/usr/bin/env bash
# ===========================================
# PostToolUse hook for shell commands - learn_error trigger
#
# Fires on a non-zero command exit and reminds the agent to call learn_error.
# The reminder is high-signal and leaves root_cause/fix to the agent after it
# has inspected the actual failure.
#
# Hook: PostToolUse (matcher: "Bash")
# ===========================================

source "$(dirname "$0")/lib/common.sh"

TOOL=$(hook_get 'tool_name')
case "$TOOL" in
    Bash|bash|exec_command|functions.exec_command) ;;
    *) exit 0 ;;
esac

EXIT_CODE=$(hook_get 'tool_response.exit_code')
[ -z "$EXIT_CODE" ] && EXIT_CODE=$(hook_get 'tool_output.exit_code')
[ -z "$EXIT_CODE" ] && EXIT_CODE=$(hook_get 'tool_output.exitCode')
[ -z "$EXIT_CODE" ] && EXIT_CODE=$(hook_get 'tool_result.exit_code')

if [ -z "$EXIT_CODE" ] || [ "$EXIT_CODE" = "0" ]; then
    exit 0
fi

COMMAND=$(hook_get 'tool_input.command' | head -c 200)
[ -z "$COMMAND" ] && COMMAND=$(hook_get 'tool_input.cmd' | head -c 200)
STDERR=$(hook_get 'tool_response.stderr' | head -c 500)
[ -z "$STDERR" ] && STDERR=$(hook_get 'tool_output.stderr' | head -c 500)
[ -z "$STDERR" ] && STDERR=$(hook_get 'tool_output.error' | head -c 500)

case "$STDERR" in
    *"permission denied by user"*|*"User denied"*|*"SIGINT"*) exit 0 ;;
esac

[ -z "$STDERR" ] && exit 0

PROJECT=$(hook_project_name)

cat <<EOF
<system-reminder>
Memory bash-error trigger: command exited $EXIT_CODE in project "$PROJECT".
If the root cause is reproducible and fixable, call:
  learn_error(
      file="<path if relevant>",
      error="$(printf '%s' "$STDERR" | tr '\n' ' ' | head -c 220)",
      root_cause="<what actually failed>",
      fix="<what resolves it>",
      pattern="<short slug, e.g. sqlite-locked-during-ddl>",
      project="$PROJECT"
  )
Skip if this was user-aborted, interactive, or benign.
</system-reminder>
EOF

exit 0

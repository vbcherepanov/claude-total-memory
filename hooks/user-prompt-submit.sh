#!/usr/bin/env bash
# ===========================================
# UserPromptSubmit Hook — capture each user prompt into `intents` table
#
# Fires on every prompt the user submits. Reads the full hook JSON from
# stdin, extracts the prompt + session_id + cwd and calls save_intent()
# asynchronously so the hook NEVER blocks input.
#
# Env:
#   CLAUDE_MEMORY_INSTALL_DIR — install root (auto-resolved)
#   CLAUDE_MEMORY_DIR         — memory storage (~/.claude-memory)
#
# Hook: UserPromptSubmit (matcher: "")
# ===========================================

# Resolve install / memory dirs (same layout as other hooks).
CLAUDE_MEMORY_INSTALL_DIR="${CLAUDE_MEMORY_INSTALL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
CLAUDE_MEMORY_DIR="${CLAUDE_MEMORY_DIR:-${TAM_MEMORY_DIR:-$HOME/.tam}}"
TAM_MEMORY_DIR="${TAM_MEMORY_DIR:-$CLAUDE_MEMORY_DIR}"

HOOK_PYTHON="${CLAUDE_MEMORY_INSTALL_DIR}/.venv/bin/python"
if [ ! -x "$HOOK_PYTHON" ]; then
    HOOK_PYTHON="python3"
fi

SRC_DIR="${CLAUDE_MEMORY_INSTALL_DIR}/src"
DB_PATH="${CLAUDE_MEMORY_DIR}/memory.db"

# Cache stdin to a temp file so the background Python can read it after
# this shell exits. Using `-c` with a script keeps python's stdin free
# for the redirected JSON.
TMP_INPUT="$(mktemp -t cmm-uprompt.XXXXXX)"
cat > "$TMP_INPUT"

# Detach a single Python process — it will read the cached JSON, write
# the intent, then delete the temp file. Never block the caller.
(
    "$HOOK_PYTHON" -c '
import json, os, sys
from pathlib import Path

src_dir = sys.argv[1]
db_path = sys.argv[2]
tmp = sys.argv[3]

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    raw = Path(tmp).read_text()
except Exception:
    raw = ""
finally:
    try:
        os.unlink(tmp)
    except Exception:
        pass

if not raw:
    sys.exit(0)

try:
    data = json.loads(raw)
except Exception:
    sys.exit(0)

# Claude Code UserPromptSubmit payload shape:
#   { "session_id": "...", "cwd": "...", "prompt": "..." }
# Older builds embedded the prompt inside `user_message.content`; fall back.
prompt = data.get("prompt")
if not prompt:
    user_msg = data.get("user_message") or {}
    if isinstance(user_msg, dict):
        prompt = user_msg.get("content") or ""
prompt = (prompt or "").strip()
if not prompt:
    sys.exit(0)

session_id = data.get("session_id") or os.environ.get("CLAUDE_SESSION_ID") or "unknown"
cwd = data.get("cwd") or os.getcwd()
project = os.environ.get("MEMORY_PROJECT") or os.path.basename(cwd) or "unknown"

if not Path(db_path).exists():
    sys.exit(0)

try:
    from intents import save_intent
    save_intent(db_path, prompt, session_id, project)
except Exception:
    # Hook must never crash the user session.
    pass
' "$SRC_DIR" "$DB_PATH" "$TMP_INPUT" >/dev/null 2>&1
) &

# Detach the background process from this shell's job table.
disown 2>/dev/null || true
exit 0

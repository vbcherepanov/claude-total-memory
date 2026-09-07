#!/usr/bin/env bash
#
# total-agent-memory — Manual Setup
# For users who prefer step-by-step control.
#
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolution: TAM_MEMORY_DIR > legacy CLAUDE_MEMORY_DIR > ~/.tam > legacy ~/.claude-memory > fresh ~/.tam
if [ -n "${TAM_MEMORY_DIR:-}" ]; then
    MEM="$TAM_MEMORY_DIR"
elif [ -n "${CLAUDE_MEMORY_DIR:-}" ]; then
    MEM="$CLAUDE_MEMORY_DIR"
    echo "  WARN: CLAUDE_MEMORY_DIR is deprecated, please switch to TAM_MEMORY_DIR" >&2
elif [ -d "$HOME/.tam" ]; then
    MEM="$HOME/.tam"
elif [ -d "$HOME/.claude-memory" ]; then
    MEM="$HOME/.claude-memory"
else
    MEM="$HOME/.tam"
fi

echo "╔═════════════════════════════════════════════════════╗"
echo "║  total-agent-memory v12.0 — Manual Setup            ║"
echo "╚═════════════════════════════════════════════════════╝"
echo ""

# 1. Dirs
echo "→ Creating directories..."
mkdir -p "$MEM"/{raw,chroma,backups}

# 2. Venv
echo "→ Creating Python venv..."
python3 -m venv "$DIR/.venv"
source "$DIR/.venv/bin/activate"

echo "→ Installing dependencies..."
pip install -q --upgrade pip
# Install from requirements.txt, not a hand-written list. The list here had
# drifted: it floored mcp at >=1.0.0 (which resolves the 2.x SDK that broke
# every install before 13.0.0) and named sentence-transformers rather than
# the fastembed backend the server actually uses.
pip install -q -r "$DIR/requirements.txt"

# 3. Pre-download model
echo "→ Loading embedding model..."
python3 -c "
import os
from fastembed import TextEmbedding
name = os.environ.get('FASTEMBED_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
TextEmbedding(name)
print(f'  OK: Model ready ({name})')
" 2>/dev/null || echo "  (will load on first use)"

# 4. MCP config
PY="$DIR/.venv/bin/python"
SRV="$DIR/src/server.py"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ INSTALLED!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Add MCP server to ~/.claude/settings.json:"
echo ""
echo '{'
echo '  "mcpServers": {'
echo '    "memory": {'
echo "      \"command\": \"$PY\","
echo "      \"args\": [\"$SRV\"],"
echo '      "env": {'
echo "        \"TAM_MEMORY_DIR\": \"$MEM\""
echo '      }'
echo '    }'
echo '  }'
echo '}'
echo ""
echo "That's it. Start claude as usual — memory is automatic."
echo ""
echo "Optional: Copy CLAUDE.md.template to your project"
echo "to instruct Claude to use memory automatically."
echo ""

#!/usr/bin/env bash
#
# total-agent-memory — One-Command Installer (multi-IDE)
#
# Usage:
#   bash install.sh                       # = --ide claude-code (default)
#   bash install.sh --ide claude-code
#   bash install.sh --ide cursor
#   bash install.sh --ide gemini-cli
#   bash install.sh --ide opencode
#   bash install.sh --ide codex
#   bash install.sh --uninstall           # remove background services (per OS)
#
# Env:
#   INSTALL_TEST_MODE=1   skip pip install, model pre-download, dashboard
#                         service, LaunchAgents (for test harness)
#   CLAUDE_MEMORY_DIR=... override memory directory (default: ~/.claude-memory)
#   OLLAMA_URL=...        override Ollama probe URL
#   MEMORY_LLM_MODEL=...  override expected model name
#   FAKE_UNAME=Linux      override uname() for tests (Linux|Darwin)
#   XDG_CONFIG_HOME=...   override systemd --user target dir
#   INSTALL_OVERWRITE_HOOKS=1   force-overwrite existing files in ~/.claude/hooks/
#                               (default: skip existing, preserve user customizations)
#
set -e

# Allow tests to override uname without faking binaries
OS_NAME="${FAKE_UNAME:-$(uname)}"

# -- Parse CLI args --
IDE="claude-code"
UNINSTALL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ide=*)
            IDE="${1#*=}"
            shift
            ;;
        --ide)
            IDE="$2"
            shift 2
            ;;
        --uninstall)
            UNINSTALL=1
            shift
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: bash install.sh [--ide claude-code|cursor|gemini-cli|opencode|codex] [--uninstall]" >&2
            exit 2
            ;;
    esac
done

case "$IDE" in
    claude-code|cursor|gemini-cli|opencode|codex) ;;
    *)
        echo "ERROR: unsupported --ide value: $IDE" >&2
        echo "Supported: claude-code, cursor, gemini-cli, opencode, codex" >&2
        exit 2
        ;;
esac

echo ""
echo "======================================================="
echo "  total-agent-memory v7.0 — Installer (IDE: $IDE)"
echo "======================================================="
echo ""

# -- Config --
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
MEMORY_DIR="${CLAUDE_MEMORY_DIR:-$HOME/.claude-memory}"
VENV_DIR="$INSTALL_DIR/.venv"
DASHBOARD_SERVICE="$INSTALL_DIR/scripts/dashboard-service.sh"

# Test mode: skip heavy steps (pip, model DL, launchctl, dashboard install)
TEST_MODE="${INSTALL_TEST_MODE:-0}"

# -----------------------------------------------------------------
# Linux helpers: WSL detection + systemd --user services setup
# -----------------------------------------------------------------

is_wsl() {
    # WSL1/WSL2 expose "microsoft" in /proc/version
    grep -qi microsoft /proc/version 2>/dev/null
}

systemd_user_available() {
    # systemctl --user works only when the user session bus is up.
    # On WSL2 this requires systemd=true in /etc/wsl.conf (WSLg).
    command -v systemctl >/dev/null 2>&1 && \
        systemctl --user show-environment >/dev/null 2>&1
}

# List of unit files this installer manages. Keep in sync with systemd/.
_systemd_units() {
    cat <<'EOF'
claude-memory-reflection.service
claude-memory-reflection.path
claude-memory-dashboard.service
claude-memory-orphan-backfill.service
claude-memory-orphan-backfill.timer
claude-memory-check-updates.service
claude-memory-check-updates.timer
EOF
}

# Services that should be enabled + started after install.
_systemd_enable_units() {
    cat <<'EOF'
claude-memory-reflection.path
claude-memory-dashboard.service
claude-memory-orphan-backfill.timer
claude-memory-check-updates.timer
EOF
}

install_systemd_user_services() {
    local src_dir="$INSTALL_DIR/systemd"
    local target_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

    if [ ! -d "$src_dir" ]; then
        echo "  SKIP: $src_dir missing (no systemd templates)"
        return 0
    fi

    mkdir -p "$target_dir"
    mkdir -p "$MEMORY_DIR/logs"

    # Substitute install-time paths in templates and drop them in target_dir.
    local unit
    while IFS= read -r unit; do
        [ -z "$unit" ] && continue
        local tpl="$src_dir/$unit"
        if [ ! -f "$tpl" ]; then
            echo "  WARN: template missing: $tpl"
            continue
        fi
        sed -e "s|@INSTALL_DIR@|$INSTALL_DIR|g" \
            -e "s|@MEMORY_DIR@|$MEMORY_DIR|g" \
            -e "s|@HOME@|$HOME|g" \
            "$tpl" > "$target_dir/$unit"
    done <<EOF
$(_systemd_units)
EOF

    echo "  OK: systemd units copied to $target_dir"

    # Activation requires a live user session bus. On WSL2 without systemd
    # enabled, or inside some CI sandboxes, systemctl --user will fail —
    # we still keep the files on disk so the user can enable them later.
    if systemd_user_available; then
        systemctl --user daemon-reload >/dev/null 2>&1 || true
        local en
        while IFS= read -r en; do
            [ -z "$en" ] && continue
            systemctl --user enable --now "$en" >/dev/null 2>&1 \
                && echo "  OK: enabled $en" \
                || echo "  WARN: failed to enable $en (systemctl --user enable returned non-zero)"
        done <<EOF
$(_systemd_enable_units)
EOF
    else
        if is_wsl; then
            echo "  WARN: systemd --user not available (WSL2 without systemd=true in /etc/wsl.conf)."
            echo "        Units copied to $target_dir — enable manually after enabling systemd:"
            echo "          printf '[boot]\\nsystemd=true\\n' | sudo tee -a /etc/wsl.conf"
            echo "          wsl.exe --shutdown   # from Windows, then reopen the shell"
            echo "          systemctl --user daemon-reload"
            echo "          systemctl --user enable --now claude-memory-reflection.path"
        else
            echo "  WARN: systemctl --user bus not reachable — units staged at $target_dir."
            echo "        Start the user manager (e.g. 'loginctl enable-linger $USER'),"
            echo "        then: systemctl --user daemon-reload && systemctl --user enable --now claude-memory-reflection.path"
        fi
    fi
}

uninstall_systemd_user_services() {
    local target_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

    if systemd_user_available; then
        local en
        while IFS= read -r en; do
            [ -z "$en" ] && continue
            systemctl --user disable --now "$en" >/dev/null 2>&1 || true
        done <<EOF
$(_systemd_enable_units)
EOF
    fi

    local unit
    while IFS= read -r unit; do
        [ -z "$unit" ] && continue
        rm -f "$target_dir/$unit"
    done <<EOF
$(_systemd_units)
EOF

    if systemd_user_available; then
        systemctl --user daemon-reload >/dev/null 2>&1 || true
    fi
    echo "  OK: systemd units removed from $target_dir"
}

uninstall_launch_agents() {
    local la_dir="$HOME/Library/LaunchAgents"
    local NAME LABEL
    if [ ! -d "$la_dir" ]; then
        return 0
    fi
    for TPL in "$INSTALL_DIR"/launchagents/*.plist; do
        [ -f "$TPL" ] || continue
        NAME=$(basename "$TPL")
        LABEL=$(basename "$NAME" .plist)
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
        rm -f "$la_dir/$NAME"
    done
    echo "  OK: LaunchAgents removed from $la_dir"
}

# -- Handle --uninstall early and exit --
if [ "$UNINSTALL" = "1" ]; then
    echo ""
    echo "-> Uninstalling total-agent-memory background services..."
    if [ "$OS_NAME" = "Darwin" ]; then
        uninstall_launch_agents
    elif [ "$OS_NAME" = "Linux" ]; then
        uninstall_systemd_user_services
    else
        echo "  SKIP: no background services to remove on $OS_NAME"
    fi
    echo "  Note: MCP config entries and memory dir were kept. Remove manually if desired:"
    echo "    - $HOME/.claude/settings.json (mcpServers.memory + hooks)"
    echo "    - $MEMORY_DIR"
    exit 0
fi

# -- 1. Create memory directories --
echo "-> Step 1: Creating memory directories..."
mkdir -p "$MEMORY_DIR"/{raw,chroma,transcripts,queue,backups,extract-queue}
echo "  OK: $MEMORY_DIR"

# -- 2. Python venv + deps --
echo "-> Step 2: Setting up Python environment..."

if ! command -v python3 &>/dev/null; then
    echo "  ERROR: python3 not found. Please install Python 3.10+"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "  ERROR: Python 3.10+ required, found $PY_VERSION"
    exit 1
fi

echo "  Python $PY_VERSION found"

if [ "$TEST_MODE" = "1" ]; then
    echo "  SKIP (test mode): venv creation and pip install"
    PY_PATH="$(command -v python3)"
else
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
        echo "  Existing venv found, updating dependencies..."
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        pip install -q --upgrade -r "$INSTALL_DIR/requirements.txt" -r "$INSTALL_DIR/requirements-dev.txt" 2>&1 | tail -1
    else
        python3 -m venv "$VENV_DIR"
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        pip install -q --upgrade pip
        echo "  Installing dependencies (this may take 2-3 minutes on first run)..."
        pip install -q -r "$INSTALL_DIR/requirements.txt" -r "$INSTALL_DIR/requirements-dev.txt" 2>&1 | tail -1
    fi
    # v9 — editable install registers `[project.scripts]` entry-points
    # (claude-total-memory, ctm-lookup, lookup-memory) on PATH inside the venv.
    echo "  Installing claude-total-memory package (registers ctm-lookup / lookup-memory)..."
    pip install -q -e "$INSTALL_DIR" 2>&1 | tail -1 || echo "  WARN: editable install failed; CLI entry-points may be missing."
    echo "  OK: Dependencies installed"
    PY_PATH="$VENV_DIR/bin/python"
fi

SRV_PATH="$INSTALL_DIR/src/server.py"

# -- 3. Pre-download embedding model --
echo "-> Step 3: Loading embedding model (first time only)..."
if [ "$TEST_MODE" = "1" ]; then
    echo "  SKIP (test mode): embedding model pre-download"
else
    python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print(f'  OK: Model ready ({m.get_sentence_embedding_dimension()}d embeddings)')
" 2>/dev/null || echo "  WARNING: Will download on first use"
fi

# =============================================================
# Step 4: Register MCP server with the chosen IDE
# =============================================================

# ----- Helper: JSON merge (works for claude-code / cursor / gemini-cli / opencode)
# Arg 1: config path
# Arg 2: parent key ("mcpServers" or "mcp")
# Uses env vars: PY_PATH, SRV_PATH, MEMORY_DIR
_json_merge_mcp() {
    local config_path="$1"
    local parent_key="$2"
    mkdir -p "$(dirname "$config_path")"
    CONFIG_PATH="$config_path" PARENT_KEY="$parent_key" PY_PATH="$PY_PATH" SRV_PATH="$SRV_PATH" MEMORY_DIR="$MEMORY_DIR" \
    python3 - <<'PY'
import json, os

path = os.environ['CONFIG_PATH']
parent = os.environ['PARENT_KEY']

server_entry = {
    'command': os.environ['PY_PATH'],
    'args': [os.environ['SRV_PATH']],
    'env': {
        'CLAUDE_MEMORY_DIR': os.environ['MEMORY_DIR'],
        'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    },
}

data = {}
if os.path.exists(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

if parent not in data or not isinstance(data.get(parent), dict):
    data[parent] = {}
data[parent]['memory'] = server_entry

with open(path, 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')

print(f'  OK: MCP memory registered in {path} (key: {parent})')
PY
}

# -----------------------------------------------------------------
# Install hooks into ~/.claude/hooks/
#
# Copies:
#   hooks/*.sh                        — core hooks (session-start, on-stop, etc.)
#   hooks/lib/common.sh               — shared utils sourced by example hooks
#   examples/hooks/pre-edit.sh        — v7.0 file_context guard (PreToolUse)
#   examples/hooks/on-bash-error.sh   — v7.0 learn_error trigger (PostToolUse)
#
# Behaviour:
#   Existing files are preserved (user may have customized them). Set
#   INSTALL_OVERWRITE_HOOKS=1 to force-overwrite. Skipped files are logged.
#   All copied .sh files get chmod +x.
# -----------------------------------------------------------------
install_hooks_to_home() {
    local hooks_target="$HOME/.claude/hooks"
    local lib_target="$hooks_target/lib"
    local overwrite="${INSTALL_OVERWRITE_HOOKS:-0}"
    local copied=0 skipped=0

    mkdir -p "$lib_target"

    _copy_hook() {
        local src="$1"
        local dst="$2"
        if [ ! -f "$src" ]; then
            return 0
        fi
        if [ -f "$dst" ] && [ "$overwrite" != "1" ]; then
            echo "  SKIP (exists): $(basename "$dst") — set INSTALL_OVERWRITE_HOOKS=1 to replace"
            skipped=$((skipped + 1))
            return 0
        fi
        cp "$src" "$dst"
        chmod +x "$dst" 2>/dev/null || true
        copied=$((copied + 1))
    }

    # Core hooks (hooks/*.sh — excluding lib/)
    local src
    for src in "$INSTALL_DIR"/hooks/*.sh; do
        [ -f "$src" ] || continue
        _copy_hook "$src" "$hooks_target/$(basename "$src")"
    done

    # Shared lib
    if [ -f "$INSTALL_DIR/hooks/lib/common.sh" ]; then
        local lib_dst="$lib_target/common.sh"
        if [ -f "$lib_dst" ] && [ "$overwrite" != "1" ]; then
            echo "  SKIP (exists): lib/common.sh"
            skipped=$((skipped + 1))
        else
            cp "$INSTALL_DIR/hooks/lib/common.sh" "$lib_dst"
            copied=$((copied + 1))
        fi
    fi

    # Example hooks — treat as optional add-ons (pre-edit.sh, on-bash-error.sh).
    # These sit under examples/ in the repo but are expected to live in
    # ~/.claude/hooks/ for the hook matchers registered below to find them.
    local ex
    for ex in pre-edit.sh on-bash-error.sh; do
        src="$INSTALL_DIR/examples/hooks/$ex"
        [ -f "$src" ] || continue
        _copy_hook "$src" "$hooks_target/$ex"
    done

    echo "  OK: Hooks synced to $hooks_target (copied=$copied, skipped=$skipped)"
}

register_mcp_claude_code() {
    local settings="$HOME/.claude/settings.json"
    echo "-> Step 4: Configuring Claude Code MCP server..."
    # Try `claude mcp add-json` CLI first (preferred by official tool)
    if command -v claude >/dev/null 2>&1 && [ "$TEST_MODE" != "1" ]; then
        local payload
        payload=$(python3 -c "
import json, os
print(json.dumps({
    'command': os.environ['PY_PATH'],
    'args': [os.environ['SRV_PATH']],
    'env': {
        'CLAUDE_MEMORY_DIR': os.environ['MEMORY_DIR'],
        'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    },
}))
" PY_PATH="$PY_PATH" SRV_PATH="$SRV_PATH" MEMORY_DIR="$MEMORY_DIR" 2>/dev/null) || payload=""
        if [ -n "$payload" ] && claude mcp add-json memory "$payload" --scope user >/dev/null 2>&1; then
            echo "  OK: Registered via 'claude mcp add-json' (scope: user)"
        else
            _json_merge_mcp "$settings" "mcpServers"
        fi
    else
        _json_merge_mcp "$settings" "mcpServers"
    fi

    # -- 4a. Copy hooks into ~/.claude/hooks/ --
    echo "-> Step 4a: Installing hook scripts..."
    install_hooks_to_home

    # -- 4b. Register hooks in settings.json (claude-code only) --
    # Paths resolve against $HOME/.claude/hooks/ so users can edit them
    # independently of the install tree.
    echo "-> Step 4b: Registering hooks..."
    local HOOKS_DIR="$HOME/.claude/hooks"
    local HOOK_SESSION="$HOOKS_DIR/session-start.sh"
    local HOOK_SESSION_END="$HOOKS_DIR/session-end.sh"
    local HOOK_STOP="$HOOKS_DIR/on-stop.sh"
    local HOOK_BASH="$HOOKS_DIR/memory-trigger.sh"
    local HOOK_WRITE="$HOOKS_DIR/auto-capture.sh"
    local HOOK_PROMPT="$HOOKS_DIR/user-prompt-submit.sh"
    local HOOK_POSTTOOL="$HOOKS_DIR/post-tool-use.sh"
    local HOOK_PREEDIT="$HOOKS_DIR/pre-edit.sh"
    local HOOK_BASH_ERR="$HOOKS_DIR/on-bash-error.sh"

    SETTINGS_PATH="$settings" \
    HOOK_SESSION="$HOOK_SESSION" HOOK_SESSION_END="$HOOK_SESSION_END" \
    HOOK_STOP="$HOOK_STOP" HOOK_BASH="$HOOK_BASH" HOOK_WRITE="$HOOK_WRITE" \
    HOOK_PROMPT="$HOOK_PROMPT" HOOK_POSTTOOL="$HOOK_POSTTOOL" \
    HOOK_PREEDIT="$HOOK_PREEDIT" HOOK_BASH_ERR="$HOOK_BASH_ERR" \
    python3 - <<'PY'
import json, os

path = os.environ['SETTINGS_PATH']
data = {}
if os.path.exists(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        data = {}

data.setdefault('hooks', {})
hooks = data['hooks']

def _has_cmd(entries, cmd):
    """Check if a matcher block with the given command already exists."""
    if not isinstance(entries, list):
        return False
    for block in entries:
        if not isinstance(block, dict):
            continue
        for h in block.get('hooks', []) or []:
            if isinstance(h, dict) and h.get('command') == cmd:
                return True
    return False

def _set_single(key, matcher, cmd):
    """Set a hook list with a single entry (overwrites prior CMM-owned entry)."""
    hooks[key] = [
        {'matcher': matcher, 'hooks': [{'type': 'command', 'command': cmd}]}
    ]

def _ensure_entry(key, matcher, cmd):
    """Append {matcher, [cmd]} to hooks[key] if not already present."""
    entries = hooks.setdefault(key, [])
    if _has_cmd(entries, cmd):
        return
    entries.append(
        {'matcher': matcher, 'hooks': [{'type': 'command', 'command': cmd}]}
    )

# Primary claude-total-memory hooks — always registered.
_set_single('SessionStart', '',           os.environ['HOOK_SESSION'])
_set_single('SessionEnd',   '',           os.environ['HOOK_SESSION_END'])
_set_single('Stop',         '',           os.environ['HOOK_STOP'])

# v8.0: capture user prompts as intents (always registered; safe no-op when
# the intents table is missing).
_set_single('UserPromptSubmit', '', os.environ['HOOK_PROMPT'])

# PostToolUse: Bash (+optional on-bash-error), Write|Edit (auto-capture),
# and opt-in post-tool-use (no-op without MEMORY_POST_TOOL_CAPTURE=1).
hooks['PostToolUse'] = [
    {'matcher': 'Bash', 'hooks': [
        {'type': 'command', 'command': os.environ['HOOK_BASH']},
    ]},
    {'matcher': 'Write|Edit', 'hooks': [
        {'type': 'command', 'command': os.environ['HOOK_WRITE']},
    ]},
    {'matcher': '', 'hooks': [
        {'type': 'command', 'command': os.environ['HOOK_POSTTOOL']},
    ]},
]
# Append on-bash-error if the script was installed (examples/hooks).
if os.path.isfile(os.environ['HOOK_BASH_ERR']):
    for block in hooks['PostToolUse']:
        if block.get('matcher') == 'Bash':
            block['hooks'].append(
                {'type': 'command', 'command': os.environ['HOOK_BASH_ERR']}
            )
            break

# PreToolUse: pre-edit guard (only if the script was installed).
if os.path.isfile(os.environ['HOOK_PREEDIT']):
    _ensure_entry('PreToolUse', 'Write|Edit', os.environ['HOOK_PREEDIT'])

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')

present = sorted(hooks.keys())
print('  OK: Hooks registered: ' + ', '.join(present))
PY
}

register_mcp_cursor() {
    echo "-> Step 4: Configuring Cursor MCP server..."
    _json_merge_mcp "$HOME/.cursor/mcp.json" "mcpServers"
}

register_mcp_gemini_cli() {
    echo "-> Step 4: Configuring Gemini CLI MCP server..."
    _json_merge_mcp "$HOME/.gemini/settings.json" "mcpServers"
}

register_mcp_opencode() {
    echo "-> Step 4: Configuring OpenCode MCP server..."
    _json_merge_mcp "$HOME/.opencode/config.json" "mcp"
}

register_mcp_codex() {
    echo "-> Step 4: Configuring Codex CLI MCP server..."
    local codex_dir="$HOME/.codex"
    local config_path="$codex_dir/config.toml"
    mkdir -p "$codex_dir"

    CODEX_CONFIG="$config_path" PY_PATH="$PY_PATH" SRV_PATH="$SRV_PATH" MEMORY_DIR="$MEMORY_DIR" \
    python3 - <<'PY'
import os, re

config_path = os.environ['CODEX_CONFIG']

# Escape backslashes and double quotes for safe TOML embedding
def toml_escape(s):
    return s.replace('\\', '/').replace('"', '\\"')

py_path = toml_escape(os.environ['PY_PATH'])
srv_path = toml_escape(os.environ['SRV_PATH'])
memory_dir = toml_escape(os.environ['MEMORY_DIR'])

toml_block = f'''
# --- Claude Total Memory MCP Server ---
[mcp_servers.memory]
command = "{py_path}"
args = ["{srv_path}"]
required = true
startup_timeout_sec = 15.0
tool_timeout_sec = 120.0

[mcp_servers.memory.env]
CLAUDE_MEMORY_DIR = "{memory_dir}"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MEMORY_TRIPLE_TIMEOUT_SEC = "120"
MEMORY_ENRICH_TIMEOUT_SEC = "90"
MEMORY_REPR_TIMEOUT_SEC = "120"
MEMORY_TRIPLE_MAX_PREDICT = "512"
# --- End Claude Total Memory ---
'''

content = ''
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        content = f.read()

if '[mcp_servers.memory]' in content:
    pattern = r'# --- Claude Total Memory MCP Server ---.*?# --- End Claude Total Memory ---'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, toml_block.strip(), content, flags=re.DOTALL)
    else:
        content = re.sub(r'\[mcp_servers\.memory\].*?(?=\n\[|\Z)', toml_block.strip(), content, flags=re.DOTALL)
    print('  OK: Updated existing memory config in ' + config_path)
else:
    content = content.rstrip() + '\n' + toml_block
    print('  OK: Added memory config to ' + config_path)

content = content.lstrip('\n')
with open(config_path, 'w') as f:
    f.write(content)
PY

    # -- Install Codex Skill --
    local skill_target="$HOME/.agents/skills/memory"
    local skill_src="$INSTALL_DIR/codex-skill"
    if [ -d "$skill_src" ]; then
        echo "-> Step 4b: Installing Codex memory skill..."
        mkdir -p "$skill_target/agents"
        cp "$skill_src/SKILL.md" "$skill_target/SKILL.md"
        if [ -d "$skill_src/agents" ]; then
            cp "$skill_src/agents/"* "$skill_target/agents/" 2>/dev/null || true
        fi
        echo "  OK: Skill installed to $skill_target"
    fi
}

# Dispatch
case "$IDE" in
    claude-code) register_mcp_claude_code ;;
    cursor)      register_mcp_cursor ;;
    gemini-cli)  register_mcp_gemini_cli ;;
    opencode)    register_mcp_opencode ;;
    codex)       register_mcp_codex ;;
esac

# -- 4c. Ollama check + optional install prompt --
echo ""
echo "-> Step 4c: Checking Ollama (optional but strongly recommended)..."
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MEMORY_LLM_MODEL="${MEMORY_LLM_MODEL:-qwen2.5-coder:7b}"

if [ "$TEST_MODE" = "1" ]; then
    echo "  SKIP (test mode): Ollama probe"
elif curl -sf --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "  OK: Ollama is running at $OLLAMA_URL"
    if curl -sf --max-time 2 "$OLLAMA_URL/api/tags" | grep -q "\"$MEMORY_LLM_MODEL\""; then
        echo "  OK: Model '$MEMORY_LLM_MODEL' is installed"
    else
        echo "  WARN: Model '$MEMORY_LLM_MODEL' NOT installed."
        echo "        For full v6.0 features, pull it now:"
        echo "          ollama pull $MEMORY_LLM_MODEL"
        echo "        Without it: deep KG triples, multi-repr, enrichment, fact merger disabled."
    fi
else
    echo "  WARN: Ollama NOT detected at $OLLAMA_URL"
    echo ""
    echo "  Without Ollama, ~40% of v6 features stay dormant:"
    echo "    - Deep KG triples (subject -> predicate -> object edges)"
    echo "    - Multi-representation embeddings (summary/keywords/questions/compressed)"
    echo "    - Entity/intent/topic extraction (deep enrichment)"
    echo "    - Semantic fact merger + HyDE query expansion"
    echo ""
    echo "  To enable the full experience:"
    if [ "$OS_NAME" = "Darwin" ]; then
        echo "    brew install ollama  (or download from https://ollama.ai)"
    else
        echo "    curl -fsSL https://ollama.com/install.sh | sh"
    fi
    echo "    ollama serve &"
    echo "    ollama pull $MEMORY_LLM_MODEL"
    echo ""
    echo "  System will still install now and work in degraded mode."
    echo "  Set MEMORY_LLM_ENABLED=auto after Ollama is ready — it picks up automatically."
fi

# -- 5. Dashboard service --
if [ "$TEST_MODE" = "1" ]; then
    echo "-> Step 5: SKIP (test mode) dashboard service"
else
    echo "-> Step 5: Setting up dashboard service..."
    "$DASHBOARD_SERVICE" install "$PY_PATH" "$INSTALL_DIR" "$MEMORY_DIR"
fi

# -- 5b. Optional background agents --
# macOS: LaunchAgents. Linux: systemd --user. Both only for claude-code IDE.
if [ "$TEST_MODE" != "1" ] && [ "$IDE" = "claude-code" ] && [ "$OS_NAME" = "Darwin" ] && [ -d "$INSTALL_DIR/launchagents" ]; then
    echo "-> Step 5b: Installing background LaunchAgents (reflection, orphan-backfill, check-updates)..."
    LA_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$LA_DIR"
    for TPL in "$INSTALL_DIR"/launchagents/*.plist; do
        NAME=$(basename "$TPL")
        DEST="$LA_DIR/$NAME"
        sed "s|__HOME__|$HOME|g" "$TPL" > "$DEST"
        LABEL=$(basename "$NAME" .plist)
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
        launchctl bootstrap "gui/$(id -u)" "$DEST" 2>/dev/null || \
        launchctl load "$DEST" 2>/dev/null || true
    done
    echo "  OK: Background agents installed"
elif [ "$IDE" = "claude-code" ] && [ "$OS_NAME" = "Linux" ] && [ -d "$INSTALL_DIR/systemd" ]; then
    echo "-> Step 5b: Installing background systemd --user services (reflection, dashboard, orphan-backfill, check-updates)..."
    install_systemd_user_services
fi

# -- 6. Verify --
echo ""
echo "-> Step 6: Verifying installation..."

if [ -f "$SRV_PATH" ]; then
    echo "  OK: Server: $SRV_PATH"
else
    echo "  FAIL: Server not found at $SRV_PATH"
fi

case "$IDE" in
    claude-code)
        CFG="$HOME/.claude/settings.json"
        python3 -c "
import json
with open('$CFG') as f:
    s = json.load(f)
assert 'memory' in s.get('mcpServers', {})
print('  OK: MCP server configured in $CFG')
" 2>/dev/null || echo "  FAIL: MCP config issue ($CFG)"
        ;;
    cursor)
        CFG="$HOME/.cursor/mcp.json"
        python3 -c "
import json
with open('$CFG') as f:
    s = json.load(f)
assert 'memory' in s.get('mcpServers', {})
print('  OK: MCP server configured in $CFG')
" 2>/dev/null || echo "  FAIL: MCP config issue ($CFG)"
        ;;
    gemini-cli)
        CFG="$HOME/.gemini/settings.json"
        python3 -c "
import json
with open('$CFG') as f:
    s = json.load(f)
assert 'memory' in s.get('mcpServers', {})
print('  OK: MCP server configured in $CFG')
" 2>/dev/null || echo "  FAIL: MCP config issue ($CFG)"
        ;;
    opencode)
        CFG="$HOME/.opencode/config.json"
        python3 -c "
import json
with open('$CFG') as f:
    s = json.load(f)
assert 'memory' in s.get('mcp', {})
print('  OK: MCP server configured in $CFG')
" 2>/dev/null || echo "  FAIL: MCP config issue ($CFG)"
        ;;
    codex)
        CFG="$HOME/.codex/config.toml"
        if grep -q "mcp_servers.memory" "$CFG" 2>/dev/null; then
            echo "  OK: MCP server configured in $CFG"
        else
            echo "  FAIL: MCP config issue ($CFG)"
        fi
        ;;
esac

if [ -d "$MEMORY_DIR" ]; then
    echo "  OK: Memory directory: $MEMORY_DIR"
else
    echo "  FAIL: Memory directory issue"
fi

if [ "$TEST_MODE" != "1" ]; then
    "$PY_PATH" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR/src')
try:
    import server  # noqa
    print('  OK: Server imports cleanly')
except Exception as e:
    print(f'  WARN: Server import issue ({e}); will verify on first use')
" 2>/dev/null || echo "  INFO: Server test skipped"
fi

# -- Done --
echo ""
echo "======================================================="
echo ""
echo "  INSTALLED SUCCESSFULLY (IDE: $IDE)"
echo ""
case "$IDE" in
    claude-code)
        echo "  Claude Code now has persistent memory."
        echo "  Just start 'claude' as usual — memory is automatic."
        ;;
    cursor)
        echo "  Cursor now has persistent memory."
        echo "  Restart Cursor — the 'memory' MCP server will auto-start."
        ;;
    gemini-cli)
        echo "  Gemini CLI now has persistent memory."
        echo "  Restart 'gemini' — the 'memory' MCP server will auto-start."
        ;;
    opencode)
        echo "  OpenCode now has persistent memory."
        echo "  Restart 'opencode' — the 'memory' MCP server will auto-start."
        ;;
    codex)
        echo "  Codex CLI now has persistent memory."
        echo "  Start 'codex' as usual — type /mcp to verify."
        ;;
esac
echo ""
echo "  MCP command:"
echo "    $PY_PATH -m claude_total_memory.server"
echo "    (or: $PY_PATH $SRV_PATH)"
echo ""
echo "  Web dashboard:"
echo "    http://localhost:37737"
echo ""
if [ "$TEST_MODE" != "1" ]; then
    "$DASHBOARD_SERVICE" print-management 2>/dev/null || true
fi
echo ""
echo "  Optional: Copy CLAUDE.md.template to your project"
echo "  to instruct Claude to use memory automatically."
echo ""
echo "======================================================="

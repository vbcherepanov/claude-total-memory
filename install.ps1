#Requires -Version 5.1
<#
.SYNOPSIS
    total-agent-memory v8.0 - One-Command Installer (Windows, multi-IDE)

.DESCRIPTION
    Creates Python venv, installs dependencies, downloads embedding model,
    registers the MCP server with the chosen IDE, installs v8.0 hooks and
    configures Windows Task Scheduler background tasks for reflection,
    orphan-backfill and check-updates.

.PARAMETER Ide
    Target IDE: claude-code (default), cursor, gemini-cli, opencode, codex.

.PARAMETER Uninstall
    Remove scheduled tasks and MCP entries (leaves venv and memory.db).

.PARAMETER TestMode
    Skip pip install, embedding model download, Task Scheduler registration
    and dashboard service. Used by test harness.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1
    powershell -ExecutionPolicy Bypass -File install.ps1 -Ide cursor
    powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall
#>

param(
    [ValidateSet("claude-code", "cursor", "gemini-cli", "opencode", "codex")]
    [string]$Ide = "claude-code",
    [switch]$Uninstall,
    [switch]$TestMode
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# TestMode can also be forced via env (parity with INSTALL_TEST_MODE=1 in bash)
if (-not $TestMode -and $env:INSTALL_TEST_MODE -eq "1") {
    $TestMode = $true
}

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  total-agent-memory v8.0.0 - Installer (Windows)"       -ForegroundColor Cyan
Write-Host "  IDE: $Ide$(if ($TestMode) {' [TEST MODE]'})"            -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# -- Config --
$InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HomeDir = if ($env:USERPROFILE) { $env:USERPROFILE } else { $env:HOME }
$MemoryDir = if ($env:CLAUDE_MEMORY_DIR) { $env:CLAUDE_MEMORY_DIR } else { [System.IO.Path]::Combine($HomeDir, ".claude-memory") }
$VenvDir = [System.IO.Path]::Combine($InstallDir, ".venv")
$ClaudeDir = [System.IO.Path]::Combine($HomeDir, ".claude")
$ClaudeSettings = [System.IO.Path]::Combine($ClaudeDir, "settings.json")

# Scheduled task names (used in both install + uninstall paths)
$TaskReflection     = "total-agent-memory-reflection"
$TaskOrphanBackfill = "total-agent-memory-orphan-backfill"
$TaskCheckUpdates   = "total-agent-memory-check-updates"
$TaskDashboard      = "ClaudeTotalMemoryDashboard"

# ===================================================================
# Uninstall branch (short-circuits before Python bootstrap)
# ===================================================================
function Invoke-Uninstall {
    Write-Host "-> Uninstalling total-agent-memory (config + scheduled tasks)..." -ForegroundColor Yellow

    # Scheduled tasks
    foreach ($t in @($TaskReflection, $TaskOrphanBackfill, $TaskCheckUpdates, $TaskDashboard)) {
        try {
            if (Get-Command Unregister-ScheduledTask -ErrorAction SilentlyContinue) {
                Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction SilentlyContinue
                Write-Host "  OK: Removed scheduled task $t" -ForegroundColor Green
            }
        } catch {
            Write-Host "  SKIP: $t (not registered)" -ForegroundColor DarkYellow
        }
    }

    # Claude Code settings.json - drop memory MCP + our hooks
    if (Test-Path $ClaudeSettings) {
        try {
            $raw = Get-Content $ClaudeSettings -Raw
            $settings = $raw | ConvertFrom-Json
            $changed = $false

            if ($settings.PSObject.Properties.Match('mcpServers').Count -gt 0 -and $settings.mcpServers) {
                if ($settings.mcpServers.PSObject.Properties.Match('memory').Count -gt 0) {
                    $settings.mcpServers.PSObject.Properties.Remove('memory')
                    $changed = $true
                }
            }
            if ($settings.PSObject.Properties.Match('hooks').Count -gt 0 -and $settings.hooks) {
                foreach ($evt in @("SessionStart", "SessionEnd", "Stop", "UserPromptSubmit", "PreToolUse", "PostToolUse")) {
                    if ($settings.hooks.PSObject.Properties.Match($evt).Count -gt 0) {
                        $settings.hooks.PSObject.Properties.Remove($evt)
                        $changed = $true
                    }
                }
            }

            if ($changed) {
                $settings | ConvertTo-Json -Depth 10 | Set-Content -Path $ClaudeSettings -Encoding UTF8
                Write-Host "  OK: Cleaned memory entries from $ClaudeSettings" -ForegroundColor Green
            }
        } catch {
            Write-Host "  WARN: Could not parse $ClaudeSettings ($($_.Exception.Message))" -ForegroundColor DarkYellow
        }
    }

    Write-Host ""
    Write-Host "  Uninstall complete. Venv and memory.db were left intact." -ForegroundColor Green
    Write-Host "  Delete manually if desired:" -ForegroundColor DarkGray
    Write-Host "    $VenvDir" -ForegroundColor DarkGray
    Write-Host "    $MemoryDir" -ForegroundColor DarkGray
    Write-Host ""
}

if ($Uninstall) {
    Invoke-Uninstall
    exit 0
}

# ===================================================================
# 1. Memory directories
# ===================================================================
Write-Host "-> Step 1: Creating memory directories..." -ForegroundColor Yellow
$dirs = @("raw", "chroma", "transcripts", "queue", "backups", "extract-queue", "logs")
foreach ($d in $dirs) {
    $path = [System.IO.Path]::Combine($MemoryDir, $d)
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}
Write-Host "  OK: $MemoryDir" -ForegroundColor Green

# ===================================================================
# 2. Python venv + deps
# ===================================================================
Write-Host "-> Step 2: Setting up Python environment..." -ForegroundColor Yellow

$pythonCmd = $null
foreach ($cmd in @("python3", "python")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) {
            $parts = $ver.Split(".")
            $major = [int]$parts[0]; $minor = [int]$parts[1]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                $pythonCmd = $cmd
                Write-Host "  Python $ver found ($cmd)" -ForegroundColor Green
                break
            }
        }
    } catch {}
}

if (-not $pythonCmd) {
    Write-Host "  ERROR: Python 3.10+ not found. Install from https://python.org" -ForegroundColor Red
    exit 1
}

$VenvPython = [System.IO.Path]::Combine($VenvDir, "Scripts", "python.exe")
$VenvPip = [System.IO.Path]::Combine($VenvDir, "Scripts", "pip.exe")

if ($TestMode) {
    Write-Host "  SKIP (test mode): venv creation and pip install" -ForegroundColor DarkYellow
    # Use system python so downstream config steps still resolve a path
    try {
        $VenvPython = (Get-Command $pythonCmd -ErrorAction Stop).Source
    } catch {
        $VenvPython = $pythonCmd
    }
} else {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "  Creating virtual environment..."
        & $pythonCmd -m venv $VenvDir
    }
    if (-not (Test-Path $VenvPython)) {
        Write-Host "  ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    & $VenvPip install -q --upgrade pip 2>$null
    Write-Host "  Installing dependencies (this may take 2-3 minutes on first run)..."
    $req = [System.IO.Path]::Combine($InstallDir, "requirements.txt")
    $reqDev = [System.IO.Path]::Combine($InstallDir, "requirements-dev.txt")
    & $VenvPip install -q -r $req -r $reqDev 2>&1 | Select-Object -Last 1
    Write-Host "  OK: Dependencies installed" -ForegroundColor Green
}

$SrvPath = [System.IO.Path]::Combine($InstallDir, "src", "server.py")

# ===================================================================
# 3. Pre-download embedding model
# ===================================================================
Write-Host "-> Step 3: Loading embedding model (first time only)..." -ForegroundColor Yellow
if ($TestMode) {
    Write-Host "  SKIP (test mode): embedding model pre-download" -ForegroundColor DarkYellow
} else {
    try {
        & $VenvPython -c @"
import os
from fastembed import TextEmbedding
name = os.environ.get('FASTEMBED_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
TextEmbedding(name)
print(f'  OK: Model ready ({name})')
"@ 2>$null
    } catch {
        Write-Host "  WARNING: Will download on first use" -ForegroundColor DarkYellow
    }
}

# ===================================================================
# Helper: merge a JSON MCP config (works for 4 of 5 IDEs)
# ===================================================================
function Merge-JsonMcp {
    param(
        [Parameter(Mandatory=$true)][string]$ConfigPath,
        [Parameter(Mandatory=$true)][string]$ParentKey   # "mcpServers" or "mcp"
    )
    $parentDir = Split-Path -Parent $ConfigPath
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    $data = [ordered]@{}
    if (Test-Path $ConfigPath) {
        try {
            $raw = Get-Content $ConfigPath -Raw
            if ($raw -and $raw.Trim()) {
                # ConvertFrom-Json returns PSCustomObject; convert to hashtable for editing
                $parsed = $raw | ConvertFrom-Json
                if ($parsed) {
                    $data = ConvertTo-HashtableFromPSObject $parsed
                }
            }
        } catch {
            # Broken file -> start fresh
            $data = [ordered]@{}
        }
    }

    if (-not $data.Contains($ParentKey) -or -not ($data[$ParentKey] -is [System.Collections.IDictionary])) {
        $data[$ParentKey] = [ordered]@{}
    }

    $data[$ParentKey]["memory"] = [ordered]@{
        command = $VenvPython
        args    = @($SrvPath)
        env     = [ordered]@{
            CLAUDE_MEMORY_DIR = $MemoryDir
        }
    }

    $json = ($data | ConvertTo-Json -Depth 10)
    Set-Content -Path $ConfigPath -Value $json -Encoding UTF8
    Write-Host "  OK: MCP memory registered in $ConfigPath (key: $ParentKey)" -ForegroundColor Green
}

function ConvertTo-HashtableFromPSObject {
    param([Parameter(ValueFromPipeline=$true)]$InputObject)
    process {
        if ($null -eq $InputObject) { return $null }
        if ($InputObject -is [System.Collections.IDictionary]) {
            $out = [ordered]@{}
            foreach ($k in $InputObject.Keys) { $out[$k] = ConvertTo-HashtableFromPSObject $InputObject[$k] }
            return $out
        }
        if ($InputObject -is [System.Collections.IEnumerable] -and -not ($InputObject -is [string])) {
            return @($InputObject | ForEach-Object { ConvertTo-HashtableFromPSObject $_ })
        }
        if ($InputObject.PSObject -and $InputObject.PSObject.Properties) {
            $out = [ordered]@{}
            foreach ($p in $InputObject.PSObject.Properties) { $out[$p.Name] = ConvertTo-HashtableFromPSObject $p.Value }
            return $out
        }
        return $InputObject
    }
}

# ===================================================================
# Register-Mcp-* functions (one per IDE)
# ===================================================================

function Register-Mcp-ClaudeCode {
    Write-Host "-> Step 4: Configuring Claude Code MCP server..." -ForegroundColor Yellow
    Merge-JsonMcp -ConfigPath $ClaudeSettings -ParentKey "mcpServers"

    # -- 4b. Register v8.0 hooks --
    Write-Host "-> Step 4b: Registering v8.0 hooks..." -ForegroundColor Yellow

    $userHookDir = [System.IO.Path]::Combine($ClaudeDir, "hooks")
    if (-not (Test-Path $userHookDir)) {
        New-Item -ItemType Directory -Path $userHookDir -Force | Out-Null
    }

    $srcHookDir = [System.IO.Path]::Combine($InstallDir, "hooks")
    $hookNames = @(
        "session-start.ps1",
        "session-end.ps1",
        "on-stop.ps1",
        "memory-trigger.ps1",
        "auto-capture.ps1",
        "user-prompt-submit.ps1",
        "post-tool-use.ps1",
        "pre-edit.ps1",
        "on-bash-error.ps1"
    )
    foreach ($h in $hookNames) {
        $src = [System.IO.Path]::Combine($srcHookDir, $h)
        $dst = [System.IO.Path]::Combine($userHookDir, $h)
        if (Test-Path $src) {
            if (Test-Path $dst) {
                Write-Host "  SKIP: $h already exists in $userHookDir (preserving user copy)" -ForegroundColor DarkYellow
            } else {
                Copy-Item -Path $src -Destination $dst -ErrorAction SilentlyContinue
            }
        }
    }

    # Build command strings pointing at copied hook files
    $pwshPrefix = "powershell -ExecutionPolicy Bypass -NoProfile -File "
    $HookSession       = [System.IO.Path]::Combine($userHookDir, "session-start.ps1")
    $HookSessionEnd    = [System.IO.Path]::Combine($userHookDir, "session-end.ps1")
    $HookStop          = [System.IO.Path]::Combine($userHookDir, "on-stop.ps1")
    $HookBash          = [System.IO.Path]::Combine($userHookDir, "memory-trigger.ps1")
    $HookWrite         = [System.IO.Path]::Combine($userHookDir, "auto-capture.ps1")
    $HookUserPrompt    = [System.IO.Path]::Combine($userHookDir, "user-prompt-submit.ps1")
    $HookPreEdit       = [System.IO.Path]::Combine($userHookDir, "pre-edit.ps1")
    $HookPostToolUse   = [System.IO.Path]::Combine($userHookDir, "post-tool-use.ps1")
    $HookOnBashError   = [System.IO.Path]::Combine($userHookDir, "on-bash-error.ps1")

    # Merge hooks block into settings.json
    $data = [ordered]@{}
    if (Test-Path $ClaudeSettings) {
        try {
            $raw = Get-Content $ClaudeSettings -Raw
            if ($raw -and $raw.Trim()) {
                $data = ConvertTo-HashtableFromPSObject ($raw | ConvertFrom-Json)
            }
        } catch {
            $data = [ordered]@{}
        }
    }
    if (-not $data.Contains("hooks") -or -not ($data.hooks -is [System.Collections.IDictionary])) {
        $data["hooks"] = [ordered]@{}
    }

    $data.hooks["SessionStart"] = @(
        @{ matcher = ""; hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookSession`"" }) }
    )
    $data.hooks["SessionEnd"] = @(
        @{ matcher = ""; hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookSessionEnd`"" }) }
    )
    $data.hooks["Stop"] = @(
        @{ matcher = ""; hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookStop`"" }) }
    )
    $data.hooks["UserPromptSubmit"] = @(
        @{ matcher = ""; hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookUserPrompt`"" }) }
    )
    $data.hooks["PreToolUse"] = @(
        @{ matcher = "Write|Edit"; hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookPreEdit`"" }) }
    )
    $data.hooks["PostToolUse"] = @(
        @{ matcher = "Bash";       hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookBash`"" }) },
        @{ matcher = "Bash";       hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookOnBashError`"" }) },
        @{ matcher = "Write|Edit"; hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookWrite`"" }) },
        @{ matcher = "*";          hooks = @(@{ type = "command"; command = $pwshPrefix + "`"$HookPostToolUse`"" }) }
    )

    ($data | ConvertTo-Json -Depth 10) | Set-Content -Path $ClaudeSettings -Encoding UTF8
    Write-Host "  OK: v8.0 hooks registered (SessionStart/End, Stop, UserPromptSubmit, PreToolUse, PostToolUse)" -ForegroundColor Green
}

function Register-Mcp-Cursor {
    Write-Host "-> Step 4: Configuring Cursor MCP server..." -ForegroundColor Yellow
    $cfg = [System.IO.Path]::Combine($HomeDir, ".cursor", "mcp.json")
    Merge-JsonMcp -ConfigPath $cfg -ParentKey "mcpServers"
}

function Register-Mcp-GeminiCli {
    Write-Host "-> Step 4: Configuring Gemini CLI MCP server..." -ForegroundColor Yellow
    $cfg = [System.IO.Path]::Combine($HomeDir, ".gemini", "settings.json")
    Merge-JsonMcp -ConfigPath $cfg -ParentKey "mcpServers"
}

function Register-Mcp-OpenCode {
    Write-Host "-> Step 4: Configuring OpenCode MCP server..." -ForegroundColor Yellow
    $cfg = [System.IO.Path]::Combine($HomeDir, ".opencode", "config.json")
    Merge-JsonMcp -ConfigPath $cfg -ParentKey "mcp"
}

function Register-Mcp-Codex {
    Write-Host "-> Step 4: Configuring Codex CLI MCP server..." -ForegroundColor Yellow
    $codexDir = [System.IO.Path]::Combine($HomeDir, ".codex")
    $configPath = [System.IO.Path]::Combine($codexDir, "config.toml")
    if (-not (Test-Path $codexDir)) {
        New-Item -ItemType Directory -Path $codexDir -Force | Out-Null
    }

    # TOML escaping: normalize backslashes to forward slashes, escape double quotes
    $pyEsc = $VenvPython.Replace("\", "/").Replace('"', '\"')
    $srvEsc = $SrvPath.Replace("\", "/").Replace('"', '\"')
    $memEsc = $MemoryDir.Replace("\", "/").Replace('"', '\"')

    $tomlBlock = @"
# --- Claude Total Memory MCP Server ---
[mcp_servers.memory]
command = "$pyEsc"
args = ["$srvEsc"]
required = true
startup_timeout_sec = 15.0
tool_timeout_sec = 120.0

[mcp_servers.memory.env]
CLAUDE_MEMORY_DIR = "$memEsc"
MEMORY_TRIPLE_TIMEOUT_SEC = "120"
MEMORY_ENRICH_TIMEOUT_SEC = "90"
MEMORY_REPR_TIMEOUT_SEC = "120"
MEMORY_TRIPLE_MAX_PREDICT = "512"
# --- End Claude Total Memory ---
"@

    $content = ""
    if (Test-Path $configPath) {
        $content = Get-Content $configPath -Raw
        if (-not $content) { $content = "" }
    }

    $fenceRegex = '(?s)# --- Claude Total Memory MCP Server ---.*?# --- End Claude Total Memory ---'
    $sectionRegex = '(?s)\[mcp_servers\.memory\].*?(?=\n\[|\z)'

    # MatchEvaluator avoids dollar-sign / backslash interpolation in the
    # replacement string (parity with bash re.sub behavior).
    $evaluator = [System.Text.RegularExpressions.MatchEvaluator] {
        param($m)
        return $tomlBlock.Trim()
    }

    if ($content -match 'mcp_servers\.memory') {
        if ([System.Text.RegularExpressions.Regex]::IsMatch($content, $fenceRegex)) {
            $content = [System.Text.RegularExpressions.Regex]::Replace($content, $fenceRegex, $evaluator)
        } else {
            $content = [System.Text.RegularExpressions.Regex]::Replace($content, $sectionRegex, $evaluator)
        }
        Write-Host "  OK: Updated existing memory config in $configPath" -ForegroundColor Green
    } else {
        $content = $content.TrimEnd() + "`n" + $tomlBlock
        Write-Host "  OK: Added memory config to $configPath" -ForegroundColor Green
    }

    $content = $content.TrimStart("`r", "`n")
    Set-Content -Path $configPath -Value $content -Encoding UTF8

    # -- 4b. Install Codex Skill --
    $skillTarget = [System.IO.Path]::Combine($HomeDir, ".agents", "skills", "memory")
    $skillSrc = [System.IO.Path]::Combine($InstallDir, "codex-skill")
    if (Test-Path $skillSrc) {
        Write-Host "-> Step 4b: Installing Codex memory skill..." -ForegroundColor Yellow
        if (-not (Test-Path $skillTarget)) {
            New-Item -ItemType Directory -Path $skillTarget -Force | Out-Null
        }
        Copy-Item -Path (Join-Path $skillSrc "*") -Destination $skillTarget -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  OK: Skill installed to $skillTarget" -ForegroundColor Green
    }
}

# ===================================================================
# 4. Dispatch on -Ide
# ===================================================================
switch ($Ide) {
    "claude-code" { Register-Mcp-ClaudeCode }
    "cursor"      { Register-Mcp-Cursor }
    "gemini-cli"  { Register-Mcp-GeminiCli }
    "opencode"    { Register-Mcp-OpenCode }
    "codex"       { Register-Mcp-Codex }
}

# ===================================================================
# 5. Background scheduled tasks (Task Scheduler, Windows analogue
#    of macOS LaunchAgents / Linux systemd)
# ===================================================================
function Register-BackgroundTask {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$Description,
        [Parameter(Mandatory=$true)][string]$ScriptPath,
        [string[]]$ScriptArgs = @(),
        [Parameter(Mandatory=$true)]$Trigger
    )

    if (-not (Get-Command Register-ScheduledTask -ErrorAction SilentlyContinue)) {
        Write-Host "  WARN: ScheduledTasks module unavailable, skipping $Name" -ForegroundColor DarkYellow
        return
    }

    try { Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue } catch {}

    $argList = @("`"$ScriptPath`"") + ($ScriptArgs | ForEach-Object { "`"$_`"" })
    $action = New-ScheduledTaskAction -Execute $VenvPython -Argument ($argList -join " ") -WorkingDirectory $InstallDir
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 1)

    Register-ScheduledTask `
        -TaskName $Name `
        -Description $Description `
        -Action $action `
        -Trigger $Trigger `
        -Settings $settings `
        -RunLevel Limited | Out-Null
    Write-Host "  OK: Scheduled task $Name registered" -ForegroundColor Green
}

function Install-BackgroundTasks {
    Write-Host "-> Step 5: Registering background scheduled tasks..." -ForegroundColor Yellow

    $reflectionScript = [System.IO.Path]::Combine($InstallDir, "src", "tools", "run_reflection.py")
    $orphanScript     = [System.IO.Path]::Combine($InstallDir, "src", "tools", "backfill_orphan_edges.py")
    $updateScript     = [System.IO.Path]::Combine($InstallDir, "src", "tools", "check_updates.py")

    # Reflection: every 5 minutes, repeated, start when available.
    # (Windows has no native file-watch trigger like launchd's WatchPaths; a
    #  companion watch-reflect.ps1 daemon could upgrade this later. For now
    #  periodic polling - run_reflection.py has its own debounce.)
    $tReflection = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) `
        -RepetitionInterval (New-TimeSpan -Minutes 5) `
        -RepetitionDuration (New-TimeSpan -Days 365)
    Register-BackgroundTask -Name $TaskReflection `
        -Description "total-agent-memory reflection runner (periodic)" `
        -ScriptPath $reflectionScript `
        -ScriptArgs @("--scope=auto") `
        -Trigger $tReflection

    # Orphan-backfill: daily at 00:00, repeat every 6h (4 fires/day)
    $tOrphan = New-ScheduledTaskTrigger -Daily -At "00:00"
    $tOrphan.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) `
        -RepetitionInterval (New-TimeSpan -Hours 6) `
        -RepetitionDuration (New-TimeSpan -Days 365)).Repetition
    Register-BackgroundTask -Name $TaskOrphanBackfill `
        -Description "total-agent-memory orphan-edge backfill (4x daily)" `
        -ScriptPath $orphanScript `
        -ScriptArgs @("--min-mentions=1", "--limit=500", "--trigger-now") `
        -Trigger $tOrphan

    # Check-updates: weekly Monday 09:00
    $tUpdates = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "09:00"
    Register-BackgroundTask -Name $TaskCheckUpdates `
        -Description "total-agent-memory weekly update check" `
        -ScriptPath $updateScript `
        -Trigger $tUpdates
}

function Install-DashboardService {
    Write-Host "-> Step 5b: Setting up dashboard service..." -ForegroundColor Yellow
    $DashboardPath = [System.IO.Path]::Combine($InstallDir, "src", "dashboard.py")

    try { Unregister-ScheduledTask -TaskName $TaskDashboard -Confirm:$false -ErrorAction SilentlyContinue } catch {}

    try {
        $WrapperPath = [System.IO.Path]::Combine($InstallDir, "start-dashboard.cmd")
        $wrapperContent = @"
@echo off
set CLAUDE_MEMORY_DIR=$MemoryDir
set DASHBOARD_PORT=37737
"$VenvPython" "$DashboardPath"
"@
        Set-Content -Path $WrapperPath -Value $wrapperContent -Encoding ASCII

        $Action = New-ScheduledTaskAction -Execute "cmd.exe" `
            -Argument "/c `"$WrapperPath`"" -WorkingDirectory $InstallDir
        $Trigger = New-ScheduledTaskTrigger -AtLogon
        $Settings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
            -ExecutionTimeLimit (New-TimeSpan -Days 365)

        Register-ScheduledTask -TaskName $TaskDashboard -Action $Action `
            -Trigger $Trigger -Settings $Settings `
            -Description "Claude Total Memory web dashboard on port 37737" `
            -RunLevel Limited | Out-Null

        Start-ScheduledTask -TaskName $TaskDashboard -ErrorAction SilentlyContinue
        Write-Host "  OK: Dashboard scheduled task created (auto-starts on login)" -ForegroundColor Green
        Write-Host "  OK: http://localhost:37737" -ForegroundColor Green
    } catch {
        Write-Host "  INFO: Could not create scheduled task (run as admin for auto-start)" -ForegroundColor DarkYellow
        Write-Host "  Run manually: .venv\Scripts\python.exe src\dashboard.py" -ForegroundColor DarkYellow
    }
}

if ($TestMode) {
    Write-Host "-> Step 5: SKIP (test mode) scheduled tasks + dashboard service" -ForegroundColor DarkYellow
} else {
    try {
        Install-BackgroundTasks
    } catch {
        Write-Host "  WARN: Background task registration failed ($($_.Exception.Message))" -ForegroundColor DarkYellow
    }
    if ($Ide -eq "claude-code") {
        try { Install-DashboardService } catch {
            Write-Host "  WARN: Dashboard service install failed ($($_.Exception.Message))" -ForegroundColor DarkYellow
        }
    }
}

# ===================================================================
# 6. Verify
# ===================================================================
Write-Host ""
Write-Host "-> Step 6: Verifying installation..." -ForegroundColor Yellow

if (Test-Path $SrvPath) {
    Write-Host "  OK: Server: $SrvPath" -ForegroundColor Green
} else {
    Write-Host "  FAIL: Server not found at $SrvPath" -ForegroundColor Red
}

function Test-McpRegistered {
    param([string]$ConfigPath, [string]$ParentKey, [bool]$IsToml = $false)
    if (-not (Test-Path $ConfigPath)) {
        Write-Host "  FAIL: Config file missing: $ConfigPath" -ForegroundColor Red
        return
    }
    if ($IsToml) {
        $c = Get-Content $ConfigPath -Raw
        if ($c -match "mcp_servers\.memory") {
            Write-Host "  OK: MCP server configured in $ConfigPath" -ForegroundColor Green
        } else {
            Write-Host "  FAIL: MCP config missing in $ConfigPath" -ForegroundColor Red
        }
    } else {
        try {
            $data = Get-Content $ConfigPath -Raw | ConvertFrom-Json
            if ($data.$ParentKey -and $data.$ParentKey.memory) {
                Write-Host "  OK: MCP server configured in $ConfigPath" -ForegroundColor Green
            } else {
                Write-Host "  FAIL: MCP config issue ($ConfigPath)" -ForegroundColor Red
            }
        } catch {
            Write-Host "  FAIL: Config parse error ($ConfigPath)" -ForegroundColor Red
        }
    }
}

switch ($Ide) {
    "claude-code" { Test-McpRegistered -ConfigPath $ClaudeSettings -ParentKey "mcpServers" }
    "cursor"      { Test-McpRegistered -ConfigPath ([System.IO.Path]::Combine($HomeDir, ".cursor", "mcp.json")) -ParentKey "mcpServers" }
    "gemini-cli"  { Test-McpRegistered -ConfigPath ([System.IO.Path]::Combine($HomeDir, ".gemini", "settings.json")) -ParentKey "mcpServers" }
    "opencode"    { Test-McpRegistered -ConfigPath ([System.IO.Path]::Combine($HomeDir, ".opencode", "config.json")) -ParentKey "mcp" }
    "codex"       { Test-McpRegistered -ConfigPath ([System.IO.Path]::Combine($HomeDir, ".codex", "config.toml")) -ParentKey "" -IsToml $true }
}

if (Test-Path $MemoryDir) {
    Write-Host "  OK: Memory directory: $MemoryDir" -ForegroundColor Green
} else {
    Write-Host "  FAIL: Memory directory issue" -ForegroundColor Red
}

# ===================================================================
# Done
# ===================================================================
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  INSTALLED SUCCESSFULLY (IDE: $Ide)" -ForegroundColor Green
Write-Host ""
switch ($Ide) {
    "claude-code" { Write-Host "  Claude Code now has persistent memory + v8.0 hooks." }
    "cursor"      { Write-Host "  Cursor now has persistent memory. Restart Cursor." }
    "gemini-cli"  { Write-Host "  Gemini CLI now has persistent memory. Restart 'gemini'." }
    "opencode"    { Write-Host "  OpenCode now has persistent memory. Restart 'opencode'." }
    "codex"       { Write-Host "  Codex CLI now has persistent memory. Type /mcp to verify." }
}
Write-Host ""
Write-Host "  Web dashboard: http://localhost:37737"
Write-Host ""
Write-Host "  Scheduled tasks (PowerShell, as current user):"
Write-Host "    Get-ScheduledTask -TaskName $TaskReflection"
Write-Host "    Get-ScheduledTask -TaskName $TaskOrphanBackfill"
Write-Host "    Get-ScheduledTask -TaskName $TaskCheckUpdates"
Write-Host ""
Write-Host "  Uninstall (config + tasks, leaves venv/memory.db):"
Write-Host "    powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall"
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan

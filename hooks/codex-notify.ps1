#Requires -Version 5.1
<#
.SYNOPSIS
    Codex CLI notify hook — reminder to save knowledge after agent turns

.DESCRIPTION
    This hook is OPTIONAL. Codex hooks are experimental.
    Memory works without hooks — AGENTS.md instructions are sufficient.

    Add to ~/.codex/config.toml:
      notify = ["powershell", "-ExecutionPolicy", "Bypass", "-File", "C:/path/to/hooks/codex-notify.ps1"]
#>

# Read JSON payload from stdin
$Input = $null
try {
    $Input = [Console]::In.ReadToEnd() | ConvertFrom-Json
} catch {}

# Detect project
$Cwd = if ($Input -and $Input.cwd) { $Input.cwd } else { Get-Location }
$Project = Split-Path -Leaf $Cwd

# Windows toast notification
try {
    [System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms") | Out-Null
    $notify = New-Object System.Windows.Forms.NotifyIcon
    $notify.Icon = [System.Drawing.SystemIcons]::Information
    $notify.Visible = $true
    $notify.ShowBalloonTip(3000, "Total Memory — $Project", "Remember: memory_save & self_reflect", [System.Windows.Forms.ToolTipIcon]::Info)
    Start-Sleep -Seconds 3
    $notify.Dispose()
} catch {
    # Silently ignore if notification fails
}

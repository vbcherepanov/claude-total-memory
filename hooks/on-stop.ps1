# Stop hook — remind Claude to save knowledge and reflect when session ends
#
# Add to %USERPROFILE%\.claude\settings.json:
#   "hooks": {
#     "Stop": [{
#       "type": "command",
#       "command": "powershell -ExecutionPolicy Bypass -File C:\\path\\to\\claude-total-memory\\hooks\\on-stop.ps1"
#     }]
#   }

$Project = Split-Path -Leaf (Get-Location)

Write-Output "MEMORY_WARNING: Session ending. Before closing:"
Write-Output "  1. Save important knowledge with memory_save(project=`"$Project`")"
Write-Output "  2. Record a reflection: self_reflect(reflection=`"...`", task_summary=`"...`", project=`"$Project`")"

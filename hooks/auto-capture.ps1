# PostToolUse:Write,Edit hook — suggest memory_observe after file changes
#
# Add to %USERPROFILE%\.claude\settings.json:
#   "hooks": {
#     "PostToolUse": [
#       {
#         "type": "command",
#         "command": "powershell -ExecutionPolicy Bypass -File C:\\path\\to\\hooks\\auto-capture.ps1",
#         "matcher": "Write"
#       },
#       {
#         "type": "command",
#         "command": "powershell -ExecutionPolicy Bypass -File C:\\path\\to\\hooks\\auto-capture.ps1",
#         "matcher": "Edit"
#       }
#     ]
#   }

$input_json = $input | Out-String

try {
    $data = $input_json | ConvertFrom-Json
    $toolName = $data.tool_name
    $filePath = $data.input.file_path
} catch {
    exit 0
}

if (-not $toolName -or -not $filePath) { exit 0 }

# Skip config/doc files
$ext = [System.IO.Path]::GetExtension($filePath)
if ($ext -in ".md", ".txt", ".log", ".json", ".yaml", ".yml", ".toml", ".lock", ".sum") { exit 0 }

$fileName = [System.IO.Path]::GetFileName($filePath)
Write-Output "MEMORY_HINT: File changed: $fileName. Consider: memory_observe(tool_name=`"$toolName`", summary=`"Modified $fileName`", files_affected=[`"$filePath`"])"

# PostToolUse:Bash hook — suggest memory_save after significant operations
#
# Triggers on: git commit, docker, migrations, package installs, builds
#
# Add to %USERPROFILE%\.claude\settings.json:
#   "hooks": {
#     "PostToolUse": [{
#       "type": "command",
#       "command": "powershell -ExecutionPolicy Bypass -File C:\\path\\to\\claude-total-memory\\hooks\\memory-trigger.ps1",
#       "matcher": "Bash"
#     }]
#   }

$input_json = $input | Out-String

try {
    $data = $input_json | ConvertFrom-Json
    $command = $data.input.command
} catch {
    exit 0
}

if (-not $command) { exit 0 }

if ($command -match "git commit") {
    Write-Output "MEMORY_HINT: Git commit detected. Consider saving the commit scope and key changes with memory_save(type='fact')."
}
elseif ($command -match "docker.compose up|docker-compose up|docker build") {
    Write-Output "MEMORY_HINT: Docker operation detected. Consider saving infrastructure config with memory_save(type='fact')."
}
elseif ($command -match "migrate|migration") {
    Write-Output "MEMORY_HINT: Database migration detected. Consider saving schema changes with memory_save(type='fact')."
}
elseif ($command -match "make setup|make init|make deploy") {
    Write-Output "MEMORY_HINT: Project setup/build detected. Consider saving infrastructure facts with memory_save(type='fact')."
}
elseif ($command -match "npm install|npm ci|yarn add|pnpm add") {
    Write-Output "MEMORY_HINT: Package install detected. Consider saving dependency changes with memory_save(type='fact')."
}
elseif ($command -match "pip install|poetry add|pipenv install") {
    Write-Output "MEMORY_HINT: Python package install detected. Consider saving dependency changes with memory_save(type='fact')."
}
elseif ($command -match "go mod|go get") {
    Write-Output "MEMORY_HINT: Go module change detected. Consider saving dependency changes with memory_save(type='fact')."
}
elseif ($command -match "cargo build|cargo add") {
    Write-Output "MEMORY_HINT: Rust cargo operation detected. Consider saving build/dependency facts with memory_save(type='fact')."
}
elseif ($command -match "composer require|composer install") {
    Write-Output "MEMORY_HINT: Composer operation detected. Consider saving dependency changes with memory_save(type='fact')."
}

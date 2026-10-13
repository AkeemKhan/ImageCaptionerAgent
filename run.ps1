# Start the web app at http://127.0.0.1:8000. Extra arguments are passed through,
# for example:  .\run.ps1 --port 8080
Set-Location (Join-Path $PSScriptRoot "captioner-agent")
& (Join-Path $PSScriptRoot ".venv\Scripts\captioner-agent.exe") @args

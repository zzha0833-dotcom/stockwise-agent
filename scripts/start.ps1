param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501,
    [string]$CondaEnvPath = "D:\software\miniconda3\envs\stockwise-agent"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:STOCKWISE_API_URL = "http://127.0.0.1:$ApiPort/api/v1"
$stockwiseExecutable = Join-Path $CondaEnvPath "Scripts\stockwise.exe"
$streamlitExecutable = Join-Path $CondaEnvPath "Scripts\streamlit.exe"

if (-not (Test-Path -LiteralPath $stockwiseExecutable)) {
    throw "StockWise executable was not found in Conda environment: $CondaEnvPath"
}
if (-not (Test-Path -LiteralPath $streamlitExecutable)) {
    throw "Streamlit executable was not found in Conda environment: $CondaEnvPath"
}

$api = Start-Process -FilePath $stockwiseExecutable -ArgumentList @("serve-api", "--port", "$ApiPort") -PassThru -WindowStyle Hidden
$ui = Start-Process -FilePath $streamlitExecutable -ArgumentList @("run", "ui/app.py", "--server.port", "$UiPort") -PassThru -WindowStyle Hidden

Write-Host "StockWise API PID: $($api.Id)"
Write-Host "StockWise UI PID:  $($ui.Id)"
Write-Host "API docs: http://127.0.0.1:$ApiPort/docs"
Write-Host "UI:       http://127.0.0.1:$UiPort"

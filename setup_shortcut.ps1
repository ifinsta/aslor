$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$target    = Join-Path $scriptDir "start.bat"
$icon      = "shell32.dll"
$desktop   = [Environment]::GetFolderPath("Desktop")
$shortcut  = Join-Path $desktop "ASLOR Proxy.lnk"

if (-not (Test-Path $target)) {
    Write-Host "[ERROR] start.bat not found at $target" -ForegroundColor Red
    pause
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell
$lnk = $WshShell.CreateShortcut($shortcut)
$lnk.TargetPath       = $target
$lnk.WorkingDirectory = $scriptDir
$lnk.IconLocation     = "$icon,13"
$lnk.Description      = "ASLOR -- Android Studio LLM OpenAI Reasoning Proxy"
$lnk.Save()

Write-Host ""
Write-Host "  Shortcut created on your desktop: ASLOR Proxy" -ForegroundColor Green
Write-Host "  Double-click it to start the proxy server." -ForegroundColor White
Write-Host "  Dashboard: http://127.0.0.1:3001/dashboard" -ForegroundColor Gray
Write-Host ""
pause

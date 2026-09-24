param([switch]$BackendOnly)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# The CMD wrapper keeps any setup error visible when launched by double-click.
& (Join-Path $PSScriptRoot 'setup.ps1') -BackendOnly:$BackendOnly
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

function Test-BackendHealth {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/api/health' -TimeoutSec 2
        return $health.ok -eq $true
    } catch { return $false }
}

if (-not (Test-BackendHealth)) {
    $backendPython = Join-Path $PSScriptRoot 'backend\.venv\Scripts\python.exe'
    $pythonExe = $backendPython

    Write-Host "Backend Python: $pythonExe"
    $logDir = Join-Path $PSScriptRoot 'backend\outputs\dev'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stderrLog = Join-Path $logDir 'backend.stderr.log'
    $backendProcess = Start-Process -FilePath $pythonExe -ArgumentList '-m', 'backend.src.server' `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir 'backend.stdout.log') -RedirectStandardError $stderrLog
    $healthy = $false
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        if (Test-BackendHealth) { $healthy = $true; break }
        $backendProcess.Refresh()
        if ($backendProcess.HasExited) { break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $healthy) {
        if (-not $backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue }
        if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Tail 30 }
        throw "Backend did not become ready. See $stderrLog"
    }
    Write-Host "Backend started (PID $($backendProcess.Id)). Logs: $logDir"
} else {
    Write-Host 'Backend is already running.'
}

Write-Host 'Backend health OK: http://127.0.0.1:8001/api/health'
if (-not $BackendOnly) {
    Write-Host 'Starting frontend: http://127.0.0.1:5173'
    Write-Host 'Keep this window open. Press Ctrl+C to stop the frontend.'
    try {
        & npm.cmd --prefix web-react run dev -- --open
        $frontendExitCode = $LASTEXITCODE
    } finally {
        # Only stop the backend started by this invocation, never a reused service.
        if ($backendProcess -and -not $backendProcess.HasExited) {
            Stop-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue
        }
    }
    exit $frontendExitCode
}

param(
    [switch]$BackendOnly,
    [switch]$Reinstall
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Program $($Arguments -join ' ')"
    }
}

function Test-Python {
    param([string]$Program, [string[]]$Prefix = @())
    try {
        & $Program @Prefix -c 'import sys, struct; sys.exit(0 if sys.version_info >= (3, 10) and struct.calcsize(''P'') == 8 else 1)' 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Test-BackendImports {
    & $pythonExe -c 'import fastapi, uvicorn, python_multipart, numpy, cv2, torch, torchvision, mediapipe; import backend.src.app' | Out-Host
    return $LASTEXITCODE -eq 0
}

try {
    $requirements = Join-Path $PSScriptRoot 'backend\requirements.txt'
    $model = Join-Path $PSScriptRoot 'backend\models\emotion\fer2013_best_model.pth'
    foreach ($file in @($requirements, $model)) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "Required file is missing: $file. Extract the complete project, including backend/models/emotion."
        }
    }

    # Check prerequisites before downloading large Python packages.
    if (-not $BackendOnly) {
        $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
        $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if (-not $nodeCommand -or -not $npmCommand) {
            throw 'Node.js and npm were not found. Install Node.js 22.12+ (with npm), then reopen this script.'
        }
        Invoke-Checked $nodeCommand.Source @('-e', 'const v=process.versions.node.split(''.'').map(Number); process.exit(v[0]>22 || (v[0]===22 && v[1]>=12) ? 0 : 1)')
        foreach ($file in @('web-react\package.json', 'web-react\package-lock.json', 'web-react\public\mediapipe\face_mesh\face_mesh.js')) {
            if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $file) -PathType Leaf)) {
                throw "Required frontend file is missing: $file. Extract the complete project."
            }
        }
    }

    $venvDir = Join-Path $PSScriptRoot 'backend\.venv'
    $pythonExe = Join-Path $venvDir 'Scripts\python.exe'
    if (Test-Path -LiteralPath $venvDir) {
        if (-not (Test-Path -LiteralPath $pythonExe) -or -not (Test-Python $pythonExe)) {
            throw "The virtual environment is unusable. Rename $venvDir and rerun setup.cmd. Do not copy virtual environments between computers."
        }
    } else {
        $basePython = $null
        $prefix = @()
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($launcher) {
            foreach ($version in @('-3.11', '-3.12', '-3.13', '-3.10', '-3')) {
                if (Test-Python $launcher.Source @($version)) {
                    $basePython = $launcher.Source
                    $prefix = @($version)
                    break
                }
            }
        }
        if (-not $basePython) {
            $command = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($command -and (Test-Python $command.Source)) { $basePython = $command.Source }
        }
        if (-not $basePython) {
            throw '64-bit Python 3.10+ was not found. Install Python (3.11 recommended), enable Add Python to PATH, then reopen this script.'
        }
        Write-Host '[1/3] Creating isolated Python environment: backend\.venv'
        Invoke-Checked $basePython ($prefix + @('-m', 'venv', $venvDir))
    }

    $pythonVersion = & $pythonExe --version
    $backendStamp = Join-Path $venvDir '.focuslens-requirements.txt'
    $backendKey = "$PSScriptRoot|$pythonVersion|$((Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash)"
    $backendCurrent = (Test-Path -LiteralPath $backendStamp) -and ((Get-Content -LiteralPath $backendStamp -Raw).Trim() -eq $backendKey)
    $installBackend = $Reinstall -or -not $backendCurrent
    if (-not $installBackend) { $installBackend = -not (Test-BackendImports) }
    if ($installBackend) {
        # Invalidate first: an interrupted installation must be retried next time.
        Set-Content -LiteralPath $backendStamp -Value '' -Encoding UTF8
        Write-Host '[2/3] Installing backend dependencies. The first download may take several minutes.'
        Invoke-Checked $pythonExe @('-m', 'ensurepip', '--upgrade')
        Invoke-Checked $pythonExe @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Checked $pythonExe @('-m', 'pip', 'install', '-r', $requirements)
        if (-not (Test-BackendImports)) { throw 'Backend dependency validation failed. See the import error above.' }
        Set-Content -LiteralPath $backendStamp -Value $backendKey -Encoding UTF8
    } else {
        Write-Host '[2/3] Backend dependencies are ready.'
    }

    if (-not $BackendOnly) {
        $frontendDir = Join-Path $PSScriptRoot 'web-react'
        $frontendStamp = Join-Path $frontendDir 'node_modules\.focuslens-dependencies.txt'
        $nodeVersion = & $nodeCommand.Source --version
        $packageHash = (Get-FileHash -LiteralPath (Join-Path $frontendDir 'package.json') -Algorithm SHA256).Hash
        $lockHash = (Get-FileHash -LiteralPath (Join-Path $frontendDir 'package-lock.json') -Algorithm SHA256).Hash
        $frontendKey = "$PSScriptRoot|$nodeVersion|$packageHash|$lockHash"
        $frontendCurrent = (Test-Path -LiteralPath $frontendStamp) -and ((Get-Content -LiteralPath $frontendStamp -Raw).Trim() -eq $frontendKey)
        if ($Reinstall -or -not $frontendCurrent -or -not (Test-Path -LiteralPath (Join-Path $frontendDir 'node_modules\.bin\vite.cmd'))) {
            Write-Host '[3/3] Installing frontend dependencies.'
            if (Test-Path -LiteralPath $frontendStamp) { Set-Content -LiteralPath $frontendStamp -Value '' -Encoding UTF8 }
            Invoke-Checked $npmCommand.Source @('--prefix', $frontendDir, 'ci', '--include=dev', '--no-audit', '--no-fund')
            Set-Content -LiteralPath $frontendStamp -Value $frontendKey -Encoding UTF8
        } else {
            Write-Host '[3/3] Frontend dependencies are ready.'
        }
    }
    Write-Host "Environment ready: $pythonExe"
    exit 0
} catch {
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

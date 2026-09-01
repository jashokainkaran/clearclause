param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,

    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,

    [Parameter(Mandatory = $true)]
    [string]$LogPath,

    [Parameter(Mandatory = $false)]
    [string[]]$ScriptArgs = @()
)

$ErrorActionPreference = "Continue"

$logDirectory = Split-Path -Parent $LogPath
if (-not (Test-Path -LiteralPath $logDirectory)) {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
}

$exitCode = 1
$transcriptStarted = $false

try {
    Start-Transcript -Path $LogPath -Force | Out-Null
    $transcriptStarted = $true

    Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Python: $PythonExe"
    Write-Host "Script: $ScriptPath"

    if ($ScriptArgs.Count -gt 0) {
        Write-Host "Arguments: $($ScriptArgs -join ' ')"
    }

    Write-Host ""

    # Run Python directly in the console instead of piping through Tee-Object.
    # This allows Hugging Face/tqdm download progress bars to remain visible.
    & $PythonExe $ScriptPath @ScriptArgs

    if ($null -eq $LASTEXITCODE) {
        $exitCode = 0
    }
    else {
        $exitCode = [int]$LASTEXITCODE
    }

    Write-Host ""
    Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Exit code: $exitCode"
}
catch {
    Write-Error $_
    $exitCode = 1
}
finally {
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
            # Do not replace the real Python exit code because transcript
            # shutdown failed.
        }
    }
}

exit $exitCode

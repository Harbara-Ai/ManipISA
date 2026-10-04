param([switch]$Rgb, [switch]$Gui, [switch]$Interactions, [switch]$AdapterChecks)
$ErrorActionPreference = 'Stop'
if ($Interactions -and $Rgb) { throw 'Use separate interaction and RGB asset checks; the interaction fixture has no camera rig.' }
if ($Interactions -and $AdapterChecks) { throw 'AdapterChecks applies to the UR5+Wuji asset, not the contact fixture.' }
$projectRoot = Split-Path $PSScriptRoot -Parent
$mapping = subst | Out-String
if (Test-Path -LiteralPath 'I:\') {
    $expected = '(?im)^I:\\: => ' + [regex]::Escape($projectRoot) + '\s*$'
    if ($mapping -notmatch $expected) { throw 'I: is occupied by another location; use the Python entry point with a suitable short path.' }
} else {
    subst I: $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project short-path mapping.' }
}
$pythonPath = 'I:\.runtime\env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The local Isaac Lab Python environment is missing.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$mode = if ($Interactions) { 'interactions' } elseif ($Rgb) { 'rgb' } else { 'structured' }
$runName = 'runtime-' + $mode + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + ([guid]::NewGuid().ToString('N').Substring(0, 6))
$artifactRoot = 'I:\artifacts'
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$runPath = Join-Path $artifactRoot $runName
$logPath = Join-Path $artifactRoot ($runName + '.log')
$runtimeArgs = @('I:\examples\isaaclab_smoke.py', '--output', $runPath)
if ($Interactions) { $runtimeArgs[0] = 'I:\examples\isaaclab_interaction_smoke.py' }
if (-not $Gui) { $runtimeArgs += '--headless' }
if ($Rgb) { $runtimeArgs += '--rgb' }
if ($AdapterChecks) { $runtimeArgs += '--adapter-checks' }
Push-Location 'I:\'
try {
    & $pythonPath @runtimeArgs *> $logPath
    $runExit = $LASTEXITCODE
} finally { Pop-Location }
if ($runExit -eq 0) {
    $reportPath = Join-Path $runPath 'report.json'
    if (-not (Test-Path -LiteralPath $reportPath)) { $runExit = 42 }
    elseif (-not (Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json).passed) { $runExit = 43 }
}
Write-Output "Report: $runPath"
Write-Output "Log: $logPath"
exit $runExit

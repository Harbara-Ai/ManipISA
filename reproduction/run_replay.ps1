param([switch]$Gui)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$mapping = (subst | Out-String)
if (Test-Path 'I:\') {
    if ($mapping -notmatch [regex]::Escape($projectRoot)) { throw 'I: is already in use by another location.' }
} else {
    subst I: $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not map project to I:.' }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$runName = 'ball-box-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$runDir = Join-Path 'I:\reproduction\runs' $runName
New-Item -ItemType Directory -Force $runDir | Out-Null
$replayArgs = @('replay.py','--hdf5','../teleopdata/dataset/27_ball_box_loading/origin-generalization/episode_000000.hdf5','--scene','scenes/27_ball_box_loading.yaml','--restore-generalization','--enable-rgb','--enable-tactile','--output',(Join-Path $runDir 'episode_000000.hdf5'),'--save-sample-frames',(Join-Path $runDir 'samples'))
if (-not $Gui) { $replayArgs += '--headless' }
[ordered]@{python='I:\.runtime\env\Scripts\python.exe';cwd='I:\Bench2Dex';arguments=$replayArgs;startedAt=(Get-Date -Format o)} | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $runDir 'command.json')
Push-Location 'I:\Bench2Dex'
try {
    & 'I:\.runtime\env\Scripts\python.exe' @replayArgs *> (Join-Path $runDir 'replay.log')
    $replayExit = $LASTEXITCODE
} finally { Pop-Location }
if ($replayExit -eq 0 -and (-not (Test-Path (Join-Path $runDir 'episode_000000.hdf5')) -or -not (Select-String -Quiet -Path (Join-Path $runDir 'replay.log') -SimpleMatch '[replay] Done: 916 frames'))) { $replayExit = 42 }
$replayExit | Set-Content (Join-Path $runDir 'exit-code.txt')
Write-Output "Replay output: $runDir"
exit $replayExit


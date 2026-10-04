$ErrorActionPreference = 'Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$env:PYTHONUTF8='1'
$py='I:\.runtime\env\Scripts\python.exe'

if (-not (Select-String -Quiet -Path '.runtime\logs\install-isaacsim-shortpath.log' -Pattern 'Successfully installed')) { throw 'Isaac Sim installation did not report success. Inspect its log.' }
Write-Output 'Installing pinned CUDA wheels'
& $py -m pip install --no-compile '.runtime\wheels\torch-2.7.0+cu128-cp311-cp311-win_amd64.whl' '.runtime\wheels\torchvision-0.22.0+cu128-cp311-cp311-win_amd64.whl'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output 'Installing benchmark support packages'
& $py -m pip install --no-compile numpy==1.26.4 Flask h5py==3.15.1 wheel==0.43.0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output 'Installing Isaac Lab v2.3.2'
& $py -m pip install --no-compile --build-constraint reproduction\build-constraints.txt -e 'I:\IsaacLab\source\isaaclab'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m pip list --format=freeze --exclude-editable | Set-Content -Encoding utf8 reproduction\environment-freeze.txt
& $py -m pip check *> reproduction\pip-check.txt
Write-Output ('pip check exit code: ' + $LASTEXITCODE)
& $py -c "import torch,numpy,importlib.metadata as m; print({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'numpy':numpy.__version__,'isaacsim':m.version('isaacsim'),'isaaclab':m.version('isaaclab')})"
exit $LASTEXITCODE


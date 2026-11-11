param([string]$OutputDir = 'evidence/windows_baseline/environment')
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
python --version 2>&1 | Out-File "$OutputDir/python.txt"
python -m pip freeze | Out-File "$OutputDir/pip_freeze.txt"
git status --short --branch | Out-File "$OutputDir/git_status.txt"
git rev-parse HEAD | Out-File "$OutputDir/git_head.txt"
git branch --all -vv | Out-File "$OutputDir/git_branches.txt"
try { codex --version 2>&1 | Out-File "$OutputDir/codex_version.txt" } catch {}
try { where.exe cl 2>&1 | Out-File "$OutputDir/where_cl.txt" } catch {}
try { cl 2>&1 | Out-File "$OutputDir/msvc.txt" } catch {}
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber,CsProcessors,CsTotalPhysicalMemory | Format-List | Out-File "$OutputDir/windows.txt"
Write-Host "Captured environment in $OutputDir"

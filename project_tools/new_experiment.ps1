param(
    [Parameter(Mandatory=$true)][string]$Name,
    [string]$Base = 'core',
    [string]$Parent = '..'
)
$ErrorActionPreference = 'Stop'
$branch = "experiment/$Name"
$path = Join-Path $Parent ("wt-" + $Name)
git show-ref --verify --quiet "refs/heads/$branch"
if ($LASTEXITCODE -eq 0) { throw "Branch already exists: $branch" }
git worktree add $path -b $branch $Base
Write-Host "Created $branch at $path"
Write-Host 'Before editing, use Plan mode and write the experiment hypothesis/control/gates.'

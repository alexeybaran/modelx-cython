param([string]$Parent = '..')
$ErrorActionPreference = 'Stop'
$items = @(
    @{Path='wt-core'; Branch='core'},
    @{Path='wt-helpers'; Branch='feature/helpers'},
    @{Path='wt-h09'; Branch='reference/h09-compatible'}
)
foreach ($item in $items) {
    $p = Join-Path $Parent $item.Path
    if (Test-Path $p) {
        Write-Host "Exists: $p"
    } else {
        git worktree add $p $item.Branch
    }
}
Write-Host 'Use the main repository project_refs/ directory as the shared model/donor/evidence source.'

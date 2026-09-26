$ErrorActionPreference = 'Stop'

Write-Host '=== Git ==='
git --version
git status --short --branch
git branch --all

Write-Host '=== Codex ==='
try { codex --version } catch { Write-Warning 'codex not found on PATH' }
try { codex doctor } catch { Write-Warning 'codex doctor did not complete; record/fix before baseline work' }

Write-Host '=== Python ==='
python --version
python -c "import sys,platform; print(sys.executable); print(platform.platform()); print(platform.processor())"

Write-Host '=== MSVC ==='
try { where.exe cl; cl 2>&1 | Select-Object -First 8 } catch { Write-Warning 'MSVC cl.exe not found. Start a Developer PowerShell or install Visual Studio Build Tools.' }

Write-Host '=== Core identity ==='
$diff = git diff upstream-pristine..core -- modelx_cython
if ($diff) {
    Write-Error 'core compiler source differs from upstream-pristine before first promotion.'
}
Write-Host 'PASS: core compiler source is identical to upstream-pristine.'

Write-Host '=== Project files ==='
foreach ($f in @('AGENTS.md','GUIDING_PRINCIPLES.md','TARGET_ARCHITECTURE.md','PLAN.md','DECISIONS.md','TESTING.md')) {
    if (-not (Test-Path $f)) { throw "Missing $f" }
    Write-Host "OK $f"
}

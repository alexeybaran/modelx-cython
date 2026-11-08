param([Parameter(Mandatory=$true)][string]$Python,
      [Parameter(Mandatory=$true)][string]$Tag,
      [string]$Dependencies,
      [string]$TempPrefix='',
      [string]$CandidateProvider='',
      [string]$ProviderManifest='',
      [ValidateSet('both','candidate','untouched','owned','audit')][string]$Suite='both')
$ErrorActionPreference='Stop'
$compiler=Split-Path -Parent $PSScriptRoot
$project=Split-Path -Parent $compiler
$workspace=Split-Path -Parent $project
$root=Join-Path $project 'september-gap'
$controls=Join-Path $project 'collect15'
if (-not $CandidateProvider) {$CandidateProvider="$root/audit-provider"}
if (-not $ProviderManifest) {$ProviderManifest="$PSScriptRoot/frozen_provider.json"}
$evidence=Join-Path $root 'evidence/ci'
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
if ($Suite -in @('both','candidate','audit')) {
    $expected=(Get-Content -LiteralPath $ProviderManifest -Raw | ConvertFrom-Json).provider
    if (-not $expected) {throw 'Provider manifest must declare provider source hashes'}
    $actual=@{}
    foreach ($entry in $expected.PSObject.Properties) {
        $file=Join-Path "$CandidateProvider/lifelib" $entry.Name
        $digest=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne $entry.Value) {throw "Provider source does not match manifest: $($entry.Name)"}
        $actual[$entry.Name]=$digest
    }
    @{provider=$CandidateProvider;manifest=$ProviderManifest;files=$actual} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$evidence/$Tag-provider.json" -Encoding utf8
}
$support=Join-Path (Split-Path -Parent $workspace) 'modelx-cython fast/m3-20261004/r/baselines-01/dv01/support'
$env:MX_CY_TEST_BUILD_ROOT=Join-Path $env:TEMP ($TempPrefix+'o')
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUNBUFFERED='1'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
Remove-Item Env:PYTHON_GIL -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONSAFEPATH -ErrorAction SilentlyContinue
$script:failed=0
function Run-Suite($Label,$Provider,$Temp,$Paths) {
    $env:PYTHONPATH=(@($compiler,"$workspace/finalization/modelx",$Provider,$support,$Dependencies) | Where-Object {$_}) -join ';'
    & $Python -B -c 'import sys,modelx_cython,modelx,lifelib,Cython; print(sys.version,Cython.__version__,modelx_cython.__file__,modelx.__file__,lifelib.__file__)' *> "$evidence/$Tag-$Label-environment.txt"
    if ($LASTEXITCODE) { throw "Dependency import failed for $Label" }
    & $Python -B -m pytest @Paths -q --basetemp "$env:TEMP/$TempPrefix$Temp" -p no:cacheprovider --junitxml "$evidence/$Tag-$Label.xml" *> "$evidence/$Tag-$Label.log"
    $code=$LASTEXITCODE
    $code | Set-Content "$evidence/$Tag-$Label.exit"
    if ($code) {$script:failed++}
}
if ($Suite -in @('both','untouched','audit')) {
Run-Suite 'untouched' "$workspace/tm1-lifelib" 'u' @("$workspace/audit-compiler/author-upstream/modelx_cython/tests")
}
$tests="$compiler/modelx_cython/tests"
if ($Suite -in @('owned','audit')) {
Run-Suite 'owned' "$workspace/tm1-lifelib" 'v' @($tests)
}
if ($Suite -in @('both','candidate','audit')) {
Run-Suite 'candidate' $CandidateProvider 'c' @(
    "$compiler/integration_tests", "$tests/test_dict_lookup.py", "$tests/test_numeric_cache_reservation.py",
    "$tests/test_cli.py", "$tests/test_parser.py", "$tests/test_tracer.py", "$tests/test_cache_hits.py", "$tests/test_automatic_cache_policy.py", "$tests/test_audit_contracts.py",
    "$controls/test_bound_resources.py", "$controls/test_group_resources.py", "$controls/test_numeric_resources.py")
}
([int]($script:failed -gt 0)) | Set-Content "$evidence/$Tag.exit"
exit ([int]($script:failed -gt 0))

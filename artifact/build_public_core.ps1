$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python --version
python -m pip install -r analysis\requirements.txt
python artifact\make_manifest.py

$scripts = @('harmonised_crosscorpus.py','hierarchical_bootstrap.py','paired_ranking_and_rho.py','phi_openended.py','mitigation_efficacy.py','interaction_changes_verdict.py')
foreach ($script in $scripts) {
  Write-Host "`n=== $script ==="
  python (Join-Path 'analysis' $script)
  if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $script" }
}
Write-Host "`nPublic-core build completed. See artifact\builds for the manifest."

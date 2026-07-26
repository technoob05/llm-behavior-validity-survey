param([string]$Destination = 'external_sources')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$moralChoiceRevision = '9f1dbced7ecf70e334af9a88c3d93be5af0f37b8'
$promptEvalRevision = '1639d5ea14c362f6964f260ae81bd903af760187'

if (-not (Test-Path "$Destination\moralchoice\.git")) {
  git clone --filter=blob:none https://github.com/ninodimontalcino/moralchoice.git "$Destination\moralchoice"
}
git -C "$Destination\moralchoice" fetch --depth 1 origin $moralChoiceRevision
git -C "$Destination\moralchoice" checkout --detach $moralChoiceRevision

$moralOut = 'analysis\moralchoice'
New-Item -ItemType Directory -Force -Path $moralOut | Out-Null
$models = @(
  'ai21_j2-jumbo-instruct',
  'anthropic_claude-instant-v1.1',
  'anthropic_claude-v1.3',
  'bigscience_bloomz-7b1',
  'cohere_command-xlarge',
  'google_flan-t5-xl',
  'google_text-bison-001',
  'meta_opt-iml-max-small',
  'openai_gpt-3.5-turbo',
  'openai_gpt-4',
  'openai_text-davinci-002',
  'openai_text-davinci-003'
)
foreach ($ambiguity in @('high', 'low')) {
  foreach ($model in $models) {
    $source = Join-Path $Destination "moralchoice\data\responses\paper\$ambiguity\results_$model.csv"
    $target = Join-Path $moralOut "${ambiguity}__$model.csv"
    if (-not (Test-Path -LiteralPath $source)) {
      throw "Pinned MoralChoice file is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination $target -Force
  }
}
Copy-Item -LiteralPath (Join-Path $moralOut 'low__anthropic_claude-v1.3.csv') `
  -Destination (Join-Path $moralOut '_sample.csv') -Force

python analysis\fetch_prompteval.py
if ($LASTEXITCODE -ne 0) {
  throw 'PromptEval retrieval failed.'
}

$mtBenchOut = 'analysis\mtbench'
New-Item -ItemType Directory -Force -Path $mtBenchOut | Out-Null
$mtBenchFile = Join-Path $mtBenchOut 'gpt4_pair.jsonl'
$mtBenchUrl = 'https://huggingface.co/spaces/lmsys/mt-bench/resolve/6e465b26cb18b64e48b3858d54ac655736cf07b6/data/mt_bench/model_judgment/gpt-4_pair.jsonl'
Invoke-WebRequest -Uri $mtBenchUrl -OutFile $mtBenchFile
$expected = 'd662c0b7d1d297f0494fcb4cc09fe8f054fa22d75deb4754a483a921984bc585'
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $mtBenchFile).Hash.ToLowerInvariant()
if ($actual -ne $expected) {
  throw "MT-Bench checksum mismatch: expected $expected, got $actual"
}

Write-Host "Fetched pinned public inputs. PromptEval revision: $promptEvalRevision"
Write-Host 'Next: python analysis\test_public_core.py'
Write-Host 'Then: powershell -ExecutionPolicy Bypass -File artifact\build_public_core.ps1'

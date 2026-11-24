# End-to-end run on Windows: env -> tests -> train -> evaluate -> drift -> robustness -> benchmark -> lock file.
# Usage (from the repo root):   powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#   optional:                    -PyVersion "-3.12"     (default is -3.14)
#                                -Quick                 (smoke test on a subsample, numbers are not reportable)
param(
    [string]$PyVersion = "-3.14",
    [switch]$Quick
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($label, [scriptblock]$cmd) {
    Write-Host "`n=== $label ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "step failed: $label" }
}

if (-not (Test-Path ".venv")) {
    Step "create venv" { py $PyVersion -m venv .venv }
}
$py = ".\.venv\Scripts\python.exe"

Step "install requirements" { & $py -m pip install --upgrade pip; & $py -m pip install -r requirements.txt }
Step "unit tests" { & $py -m pytest -q }

$trainArgs = @()
if ($Quick) { $trainArgs += "--quick" }
Step "train" { & $py -m src.train @trainArgs }
Step "evaluate" { & $py -m src.evaluate }
Step "rolling backtest" { & $py -m src.backtest }
Step "drift simulation" { & $py -m src.monitoring --simulate }
Step "robustness" { & $py -m serving.robustness }
Step "benchmark" { & $py -m serving.benchmark }
Step "batch demo" { & $py -m serving.predict_batch --input data/sample_with_bad_rows.csv --output reports/demo_scores.csv }

& $py -m pip freeze | Out-File -Encoding ascii requirements.lock.txt
Write-Host "`nAll done. Results are in reports/ and artifacts/." -ForegroundColor Green

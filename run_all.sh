#!/usr/bin/env bash
# Same pipeline as run_all.ps1 for Linux/macOS.   Usage: bash run_all.sh [--quick]
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
VPY=.venv/bin/python
$VPY -m pip install --upgrade pip
$VPY -m pip install -r requirements.txt
$VPY -m pytest -q
$VPY -m src.train "$@"
$VPY -m src.evaluate
$VPY -m src.backtest
$VPY -m src.monitoring --simulate
$VPY -m serving.robustness
$VPY -m serving.benchmark
$VPY -m serving.predict_batch --input data/sample_with_bad_rows.csv --output reports/demo_scores.csv
$VPY -m pip freeze > requirements.lock.txt
echo "All done. Results are in reports/ and artifacts/."

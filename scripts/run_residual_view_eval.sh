#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"

python run_residual_view.py --config \
  "${1:-config/system1_eval.yaml}"

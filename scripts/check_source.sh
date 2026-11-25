#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m chem_prep doctor
"$PYTHON_BIN" -m unittest discover -s tests -v

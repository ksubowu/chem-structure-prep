#!/usr/bin/env bash
set -euo pipefail
CHEM_PREP_TOOL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$CHEM_PREP_TOOL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CHEM_PREP_DEMO_DIR="${1:-$PWD/chem_prep_demo}"
python -m chem_prep doctor
python -m chem_prep init "$CHEM_PREP_DEMO_DIR"
python -m chem_prep run --config "$CHEM_PREP_DEMO_DIR/recipe.yaml"
python -m chem_prep validate --dataset "$CHEM_PREP_DEMO_DIR/dataset"

#!/usr/bin/env bash
# Полный прогон экспериментов. Требует: pip install numpy scipy pillow trimesh rtree h5py
# и клон GNM:  git clone --depth 1 https://github.com/google/GNM ../../../GNM   (или export GNM_DIR=...)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY figures.py                 # 01..04 — только модель D.Va
$PY fit_gnm.py                 # подгонка identity GNM (σ = 3 / 1 / 0.3 мм) -> out/fit_sigma*.npz
$PY gnm_figures.py             # 05, 06 + out/*.json

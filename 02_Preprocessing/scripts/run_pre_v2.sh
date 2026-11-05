#!/bin/bash
# Jalankan-ulang praproses COG-BCI dari data terverifikasi, 21 subjek unik.
PY="$HOME/neuraldecoding/.venv/bin/python"
cd "$HOME/neuraldecoding/02_Preprocessing/scripts" || exit 1
SUBS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 25 26 28 29"
echo "=== MULAI $(date '+%F %T') — 21 subjek ==="
echo "python: $PY"
"$PY" preprocess_cogbci.py --config ../config/cogbci_v2.json --subjects $SUBS
echo "=== SELESAI praproses, kode=$? $(date '+%F %T') ==="

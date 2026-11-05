#!/bin/bash
# Parallel continuation of jalankan_v3.sh (same scripts, same arguments; only the order and
# parallelism change). graded_v3 and deepen_v3 are already done; icc_montage_v3 is running
# under the original chain and is waited for, not restarted.
set -u
ND=~/neuraldecoding
PY=$ND/.venv/bin/python
L=$ND/08_Learning_Transfer/logs
MAIN=$L/jalankan_v3.log
export PYTHONWARNINGS=ignore
ICC_PID=${1:-}

say () { echo "[$(date)] $*" | tee -a $MAIN; }
run () { local name=$1; shift; say "mulai $name"; nice -n 10 "$@"; say "$name exit $?"; }

# multi-seed is independent of everything else: start now, one worker per seed
( cd $ND/08_Learning_Transfer/scripts && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  run multiseed_v3 $PY multiseed_fdr_v3.py --seeds 5 --n-jobs 5 --eegmat ../results/eegmat_full.json > $L/multiseed_v3.log 2>&1 ) &
MS=$!

# wait for the icc_montage_v3 started by the original chain
if [ -n "$ICC_PID" ]; then
  while kill -0 "$ICC_PID" 2>/dev/null; do sleep 30; done
  say "icc_montage_v3 selesai (dari rangkaian awal)"
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $ND/08_Learning_Transfer/scripts
( run icc_ci_v3 $PY icc_ci_v3.py > $L/icc_ci_v3.log 2>&1
  run strengthen_v3 $PY strengthen_light_v3.py > $L/strengthen_light_v3.log 2>&1
  cd $ND && run e1_loso_v3 $PY 10_Revision_IASC_MOABB/scripts/e1_loso_orchestrator_v3.py --cap 100 --inner-cv 10 --n-jobs 3 --seed 0 \
      --out $ND/10_Revision_IASC_MOABB/results_v3/e1_loso.json > $ND/10_Revision_IASC_MOABB/results_v3/e1_loso.log 2>&1 ) &
A=$!
( run stew_rating_v3 $PY stew_rating_v3.py > $L/stew_rating_v3.log 2>&1
  cd $ND/new_analysis/scripts
  export ND_FEAT_ORIG=03_Features/features_v3 ND_DATA_OUT=new_analysis/data_v3
  run ds003838_v3 $PY 08_ds003838_transfer.py > $ND/new_analysis/data_v3/ds003838_transfer.log 2>&1
  run hinss_leakfree_v3 $PY 13_hinss_leakfree.py > $ND/new_analysis/data_v3/hinss_leakfree.log 2>&1 ) &
B=$!
( run cbramod_v3 $PY cbramod_graded_v3.py > $L/cbramod_graded_v3.log 2>&1 ) &
C=$!

wait $A $B $C $MS
say "SELESAI"

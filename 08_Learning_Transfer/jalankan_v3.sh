#!/bin/bash
# Re-run every analysis of the graded-index manuscript on COG-BCI n=20
# (features_v3 / derivatives_v3: sub-28 removed because its session S1 is identical to sub-25 S1).
# v2 scripts and results are left untouched; v3 copies differ only in input/output paths.
set -u
ND=~/neuraldecoding
PY=$ND/.venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore

# 1) build v3 script copies (paths only)
cd $ND/08_Learning_Transfer/scripts
for s in graded deepen multiseed_fdr icc_montage icc_ci strengthen_light cbramod_graded stew_rating; do
  sed -e 's|features_v2|features_v3|g' -e 's|results_v2|results_v3|g' -e 's|derivatives_v2|derivatives_v3|g' \
      -e 's|21 subjek unik terverifikasi|20 subjek, tanpa rekaman bersama|' -e 's|21 unique subjects|20 unique subjects|g' \
      ${s}_v2.py > ${s}_v3.py
done
sed -e 's|features_v2|features_v3|g' $ND/10_Revision_IASC_MOABB/scripts/e1_loso_orchestrator_v2.py \
    > $ND/10_Revision_IASC_MOABB/scripts/e1_loso_orchestrator_v3.py
mkdir -p $ND/08_Learning_Transfer/results_v3 $ND/08_Learning_Transfer/logs $ND/new_analysis/data_v3 $ND/10_Revision_IASC_MOABB/results_v3
L=$ND/08_Learning_Transfer/logs

run () { echo "[$(date)] mulai $1"; shift; nice -n 10 "$@"; echo "[$(date)] exit $?"; }

cd $ND/08_Learning_Transfer/scripts
run graded_v3        $PY graded_v3.py            > $L/graded_v3.log 2>&1
run deepen_v3        $PY deepen_v3.py            > $L/deepen_v3.log 2>&1
run icc_montage_v3   $PY icc_montage_v3.py       > $L/icc_montage_v3.log 2>&1
run icc_ci_v3        $PY icc_ci_v3.py            > $L/icc_ci_v3.log 2>&1
run strengthen_v3    $PY strengthen_light_v3.py  > $L/strengthen_light_v3.log 2>&1
run stew_rating_v3   $PY stew_rating_v3.py       > $L/stew_rating_v3.log 2>&1
run cbramod_v3       $PY cbramod_graded_v3.py    > $L/cbramod_graded_v3.log 2>&1

cd $ND/new_analysis/scripts
export ND_FEAT_ORIG=03_Features/features_v3 ND_DATA_OUT=new_analysis/data_v3
run ds003838_v3      $PY 08_ds003838_transfer.py > $ND/new_analysis/data_v3/ds003838_transfer.log 2>&1
run hinss_leakfree_v3 $PY 13_hinss_leakfree.py   > $ND/new_analysis/data_v3/hinss_leakfree.log 2>&1
unset ND_FEAT_ORIG ND_DATA_OUT

cd $ND
run e1_loso_v3 $PY 10_Revision_IASC_MOABB/scripts/e1_loso_orchestrator_v3.py --cap 100 --inner-cv 10 --n-jobs 4 --seed 0 \
    --out $ND/10_Revision_IASC_MOABB/results_v3/e1_loso.json > $ND/10_Revision_IASC_MOABB/results_v3/e1_loso.log 2>&1

cd $ND/08_Learning_Transfer/scripts
run multiseed_v3 $PY multiseed_fdr_v3.py --seeds 5 --n-jobs 2 --eegmat ../results/eegmat_full.json > $L/multiseed_v3.log 2>&1
echo "[$(date)] SELESAI"

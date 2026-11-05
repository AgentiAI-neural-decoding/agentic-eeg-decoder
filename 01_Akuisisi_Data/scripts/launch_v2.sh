#!/bin/bash
# Peluncur: 5 pekerja paralel, lalu verifikasi silang setelah semuanya selesai.
cd "$HOME/neuraldecoding/01_Akuisisi_Data" || exit 1
N=5
for i in $(seq 0 $((N-1))); do
  python3 scripts/redownload_cogbci_v2.py --slice "$i/$N" \
      --state "redownload_v2_state_$i.json" > "redownload_v2_w$i.log" 2>&1 &
done
wait
{
  echo "=================================================================="
  echo "SEMUA PEKERJA SELESAI — menjalankan verifikasi silang"
  echo "=================================================================="
  python3 scripts/redownload_cogbci_v2.py --verify-only
  echo "kode keluar verifikasi: $?"
} >> redownload_v2.log 2>&1

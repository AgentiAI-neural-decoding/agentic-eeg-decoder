#!/usr/bin/env python3
"""Pipeline preprocessing Hinss et al. 2021 (Passive BCI Hackathon, Zenodo 5055046).

Dataset ini SUDAH ter-epoch (2 s, 250 Hz, 61-ch, average-ref, montage terpasang) —
berbeda dari COG-BCI yang kontinu. Maka tahapan disesuaikan untuk input epoched,
namun MENYAMAKAN pembersihan dgn derivatives COG-BCI agar fitur sebanding:

  1. Muat epoch EEGLAB (.set) lewat MNE
  2. Bandpass 0.5-45 Hz (sekaligus menghapus line 50 Hz → notch tak perlu)
  3. Re-reference rata-rata
  4. ICA (extended infomax, n_comp=99% varian) pada salinan high-pass 1 Hz
  5. ICLabel → tolak komponen non-otak (eye, muscle, heart, line, channel; p>=0.7)
  6. Terapkan ICA bersih ke epoch
  7. Reject amplitudo 150 µV
  8. Simpan sub-<P>_ses-<S>_task-<TASK>_run-1_proc-clean_epo.fif
     (nama identik pola COG-BCI → langsung dikonsumsi extract_features.py)

Task berlabel yang dipakai (semua sesi tempat file itu ada):
  - MATBeasy/MATBmed/MATBdiff → hanya di S1 & S2 (3 kondisi × 2 sesi).
  - RS (resting, proxy low-workload) → ada di S1, S2, DAN S3 → ketiganya diproses.
Dibuang: RSraw (belum tersaring) & 'testset' S3 (label kompetisi disembunyikan).
Perkiraan: 15 subjek × (6 MATB + 3 RS) ≈ 135 rekaman × ~13 s ≈ 30 menit.

Output derivatives ditaruh di new_analysis/derivatives/Hinss2021 (folder 02_Preprocessing
asli read-only; skrip & turunan sesi ini dijaga di new_analysis).

Penggunaan:
    python preprocess_hinss.py --subjects P04 P05     # subset
    python preprocess_hinss.py                        # semua 15 subjek
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

mne.set_log_level("ERROR")

PROJ = Path(__file__).resolve().parent.parent.parent  # Pelaksanaan Penelitian
RAW_ROOT = PROJ / "01_Akuisisi_Data/raw/Hinss2021_PassiveBCI"
OUT_ROOT = PROJ / "new_analysis/derivatives/Hinss2021"
REJECT_LABELS = {"muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"}
# Task diselaraskan ke penamaan COG-BCI agar label_schemes cocok:
#   MATBeasy/MATBmed/MATBdiff sudah sama; resting Hinss 'RS' -> 'RS_Beg_EO' (proxy resting low).
TASK_MAP = {"MATBeasy": "MATBeasy", "MATBmed": "MATBmed", "MATBdiff": "MATBdiff",
            "RS": "RS_Beg_EO"}


def parse_meta(path: Path) -> dict | None:
    """alldata_sbj04_sess1_MATBeasy.set -> {sub:'04', ses:'S1', task:'MATBeasy'}."""
    m = re.search(r"sbj(\d+)_sess(\d+)_([A-Za-z]+)", path.stem)
    if not m:
        return None
    raw_task = m.group(3)
    if raw_task not in TASK_MAP:      # buang RSraw, testset, dll.
        return None
    return {"sub": m.group(1).zfill(2), "ses": f"S{m.group(2)}", "task": TASK_MAP[raw_task],
            "raw_task": raw_task}


def clean_epochs(ep: mne.Epochs) -> tuple[mne.Epochs, dict]:
    ep = ep.copy()
    ep.filter(0.5, 45.0, verbose="ERROR")
    ep.set_eeg_reference("average", verbose="ERROR")
    ica_src = ep.copy().filter(1.0, None, verbose="ERROR")
    ica = ICA(n_components=0.99, method="infomax", fit_params=dict(extended=True),
              random_state=42, max_iter=1000)
    ica.fit(ica_src)
    lab = label_components(ep, ica, method="iclabel")
    labels, probs = lab["labels"], lab["y_pred_proba"]
    excl = [i for i, (l, p) in enumerate(zip(labels, probs)) if l in REJECT_LABELS and p >= 0.7]
    ica.exclude = excl
    ica.apply(ep)
    n0 = len(ep)
    ep.drop_bad(reject={"eeg": 150e-6}, verbose="ERROR")
    return ep, {"n_ica": int(ica.n_components_), "n_excl": len(excl),
                "n_before": int(n0), "n_after": int(len(ep)),
                "ic_labels": dict(Counter(labels))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", help="mis. P04 P05. Kosong = semua.")
    args = ap.parse_args()

    subs = args.subjects or sorted(p.name for p in RAW_ROOT.glob("P??"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    files = []
    for s in subs:
        files += sorted((RAW_ROOT / s).rglob("*.set"))
    metas = [(f, parse_meta(f)) for f in files]
    metas = [(f, m) for f, m in metas if m is not None]
    print(f"[Hinss] {len(subs)} subjek, {len(metas)} rekaman berlabel", flush=True)

    summary = []
    t_all = time.time()
    for i, (f, m) in enumerate(metas):
        tag = f"sub-{m['sub']}_ses-{m['ses']}_task-{m['task']}_run-1"
        out_dir = OUT_ROOT / f"sub-{m['sub']}" / f"ses-{m['ses']}" / "eeg"
        out_dir.mkdir(parents=True, exist_ok=True)
        epo_path = out_dir / f"{tag}_proc-clean_epo.fif"
        rep_path = out_dir / f"{tag}_ica-report.json"
        if epo_path.exists():
            print(f"  [skip ada] {tag}", flush=True)
            continue
        t0 = time.time()
        try:
            ep = mne.io.read_epochs_eeglab(f)
            clean, rep = clean_epochs(ep)
            clean.save(epo_path, overwrite=True, verbose="ERROR")
            rep.update({"tag": tag, "src": f.name, "sec": round(time.time() - t0, 1)})
            rep_path.write_text(json.dumps(rep, indent=1))
            summary.append(rep)
            print(f"  [{i+1}/{len(metas)}] {tag}: {rep['n_ica']}IC "
                  f"{rep['n_excl']}excl {rep['n_before']}->{rep['n_after']} "
                  f"({rep['sec']}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {tag}: {type(e).__name__}: {e}", flush=True)

    (OUT_ROOT / "preprocess_summary.json").write_text(json.dumps(summary, indent=1))
    tot_b = sum(r["n_before"] for r in summary)
    tot_a = sum(r["n_after"] for r in summary)
    print(f"\n[Hinss] selesai {len(summary)} rekaman, {time.time()-t_all:.0f}s. "
          f"Epoch {tot_b}->{tot_a} ({100*tot_a/max(tot_b,1):.0f}% retensi)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

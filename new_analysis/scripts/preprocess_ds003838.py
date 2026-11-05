#!/usr/bin/env python3
"""Pipeline preprocessing OpenNeuro ds003838 (Pavlov et al. — digit-span WM load).

ds003838 = EEG 63-ch @1000 Hz kontinu (bukan ter-epoch). Task 'memory' = digit-span
serial-recall: subjek MEMORIZE (memory) atau hanya LISTEN (control) sekuens 5/9/13
digit (SOA 2 s). Task 'rest' = 4 menit mata-tertutup. Montage memuat SEMUA 14 kanal
umum proyek → drop-in ke ruang fitur transfer.

Tahapan (menyamakan pembersihan COG-BCI agar fitur sebanding):
  1. Muat raw EEGLAB kontinu (.set)
  2. Montage standard_1020; buang kanal non-EEG & tanpa posisi
  3. Notch 50/100 Hz, bandpass 0.5-45 Hz
  4. Resample 250 Hz (feature-extraction meresample ke 128 Hz)
  5. Bad channel LOF + interpolasi; re-reference rata-rata
  6. ICA extended-infomax (0.99 var) pada salinan high-pass 1 Hz
  7. ICLabel → tolak komponen non-otak (eye/muscle/heart/line/channel; p>=0.7)
  8. Terapkan ICA bersih
  9. EPOCHING per-kondisi (2 s, baseline None):
       - rest  : epoch tetap 2 s (mata tertutup) -> task-rest
       - memory: time-lock ke onset tiap digit ENCODING; dipisah per
                 (memorize/listen) x (5/9/13) -> task-{mem,ctrl}{5,9,13}
 10. Simpan sub-XXX_ses-1_task-<COND>_run-1_proc-clean_epo.fif
     (pola nama identik COG-BCI/Hinss -> langsung dikonsumsi extract_features.py)

Skema label ds003838 (ditambahkan ke features.json terpisah):
  - workload_extreme : low=rest, high=mem13 (overload span) — sebanding kontras ekstrem
  - span_graded      : mem5=0, mem9=1, mem13=2 (beban WM bertingkat, within-dataset)
  - mem_vs_ctrl      : listen(0) vs memorize(1) pada input sensorik tercocok

Output derivatives -> new_analysis/derivatives/ds003838 (folder 02_ asli read-only).

Penggunaan:
    python preprocess_ds003838.py --subjects 032 033      # subset
    python preprocess_ds003838.py                          # semua subjek ber-EEG
    python preprocess_ds003838.py --limit-memory-epochs 0  # tanpa cap (default cap 4000/kondisi)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA, find_bad_channels_lof
from mne_icalabel import label_components

mne.set_log_level("ERROR")

PROJ = Path(__file__).resolve().parent.parent.parent  # neuraldecoding root
RAW_ROOT = PROJ / "01_Akuisisi_Data/raw/ds003838"
OUT_ROOT = PROJ / "new_analysis/derivatives/ds003838"
REJECT_LABELS = {"muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"}
NOTCH = [50.0, 100.0]
BANDPASS = (0.5, 45.0)
RESAMPLE_HZ = 250.0
EPOCH_LEN = 2.0
REJECT_UV = 150e-6

# trial_type -> kondisi. Contoh:
#   "memory 01/05 correct: memorize digit 1 (first) in 5 digit sequence; ..."
#   "control 03/09: listen to digit 3 in 9 digit sequence"
ENC_RE = re.compile(r"(memorize|listen to) digit .* in (\d+) digit sequence")


def cond_of(trial_type: str) -> str | None:
    m = ENC_RE.search(trial_type)
    if not m:
        return None
    verb, n = m.group(1), m.group(2)
    prefix = "mem" if verb == "memorize" else "ctrl"
    return f"{prefix}{n}"


def preprocess_continuous(raw_path: Path, logger) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose="ERROR")
    raw.set_montage("standard_1020", on_missing="warn", verbose="ERROR")
    non_eeg = [ch for ch in raw.ch_names
               if ch.upper().startswith(("ECG", "EOG", "EMG", "EXG", "STATUS", "TRIG",
                                         "GSR", "PPG", "HEOG", "VEOG", "M1", "M2"))]
    if non_eeg:
        raw.drop_channels(non_eeg)
    no_pos = [raw.ch_names[i] for i, ch in enumerate(raw.info["chs"])
              if np.isnan(ch["loc"][:3]).any() or np.allclose(ch["loc"][:3], 0.0)]
    if no_pos:
        raw.drop_channels(no_pos)
    raw.info["line_freq"] = 50
    raw.notch_filter(freqs=NOTCH, verbose="ERROR")
    raw.filter(l_freq=BANDPASS[0], h_freq=BANDPASS[1], verbose="ERROR")
    raw.resample(RESAMPLE_HZ, verbose="ERROR")
    try:
        bads = find_bad_channels_lof(raw, threshold=1.5)
    except Exception as e:  # noqa: BLE001
        logger(f"    LOF gagal ({e}) — lewati")
        bads = []
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    # ICA
    ica_src = raw.copy().filter(1.0, None, verbose="ERROR")
    ica = ICA(n_components=0.99, method="infomax", fit_params=dict(extended=True),
              random_state=42, max_iter=1000)
    ica.fit(ica_src)
    lab = label_components(raw, ica, method="iclabel")
    labels, probs = lab["labels"], lab["y_pred_proba"]
    excl = [i for i, (l, p) in enumerate(zip(labels, probs)) if l in REJECT_LABELS and p >= 0.7]
    ica.exclude = excl
    ica.apply(raw)
    raw._ppinfo = {"n_ica": int(ica.n_components_), "n_excl": len(excl),
                   "n_bads": len(bads), "ic_labels": dict(Counter(labels))}
    return raw


def epochs_from_rest(raw: mne.io.BaseRaw) -> mne.Epochs:
    return mne.make_fixed_length_epochs(raw, duration=EPOCH_LEN, preload=True, verbose="ERROR")


def epochs_from_memory(raw: mne.io.BaseRaw, events_tsv: Path, cap: int) -> dict[str, mne.Epochs]:
    """Bangun epoch encoding 2 s per kondisi dari events.tsv (onset detik)."""
    rows = []
    with open(events_tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        oi, ti = header.index("onset"), header.index("trial_type")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(oi, ti):
                continue
            c = cond_of(p[ti])
            if c is None:
                continue
            try:
                onset = float(p[oi])
            except ValueError:
                continue
            rows.append((onset, c))
    # kelompokkan per kondisi, cap opsional
    by_cond: dict[str, list[float]] = defaultdict(list)
    for onset, c in rows:
        by_cond[c].append(onset)
    out: dict[str, mne.Epochs] = {}
    sf = raw.info["sfreq"]
    tmax_samp = int(round(EPOCH_LEN * sf))
    n_times = len(raw.times)
    for c, onsets in by_cond.items():
        if cap and len(onsets) > cap:
            onsets = onsets[:cap]
        samples = [int(round(o * sf)) for o in onsets]
        samples = [s for s in samples if s >= 0 and s + tmax_samp < n_times]
        if not samples:
            continue
        ev = np.column_stack([np.array(samples), np.zeros(len(samples), int),
                              np.ones(len(samples), int)])
        ep = mne.Epochs(raw, ev, event_id={c: 1}, tmin=0.0,
                        tmax=EPOCH_LEN - 1.0 / sf, baseline=None,
                        preload=True, reject=None, verbose="ERROR")
        out[c] = ep
    return out


def save_epochs(ep: mne.Epochs, sub: str, cond: str, summary: list) -> None:
    n0 = len(ep)
    ep.drop_bad(reject={"eeg": REJECT_UV}, verbose="ERROR")
    out_dir = OUT_ROOT / f"sub-{sub}" / "ses-1" / "eeg"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sub-{sub}_ses-1_task-{cond}_run-1"
    ep.save(out_dir / f"{tag}_proc-clean_epo.fif", overwrite=True, verbose="ERROR")
    summary.append({"sub": sub, "cond": cond, "n_before": int(n0), "n_after": int(len(ep))})


def eeg_subjects() -> list[str]:
    subs = sorted({p.parts[-3].split("-")[1]
                   for p in RAW_ROOT.glob("sub-*/eeg/*task-memory_eeg.set")})
    return subs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--cap", type=int, default=4000,
                    help="cap epoch per kondisi memory (0=tanpa cap)")
    args = ap.parse_args()

    subs = args.subjects or eeg_subjects()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[ds003838] {len(subs)} subjek ber-EEG", flush=True)
    logger = lambda m: print(m, flush=True)  # noqa: E731

    summary, ica_reports = [], []
    t_all = time.time()
    for i, sub in enumerate(subs):
        subdir = RAW_ROOT / f"sub-{sub}" / "eeg"
        mem_set = subdir / f"sub-{sub}_task-memory_eeg.set"
        rest_set = subdir / f"sub-{sub}_task-rest_eeg.set"
        # skip bila sudah ada output rest+mem13 (penanda selesai)
        done_marker = OUT_ROOT / f"sub-{sub}" / "ses-1" / "eeg" / \
            f"sub-{sub}_ses-1_task-mem13_run-1_proc-clean_epo.fif"
        if done_marker.exists():
            print(f"  [skip ada] sub-{sub}", flush=True)
            continue
        t0 = time.time()
        try:
            if rest_set.exists():
                raw_r = preprocess_continuous(rest_set, logger)
                save_epochs(epochs_from_rest(raw_r), sub, "rest", summary)
                ica_reports.append({"sub": sub, "file": "rest", **raw_r._ppinfo})
            if mem_set.exists():
                raw_m = preprocess_continuous(mem_set, logger)
                ica_reports.append({"sub": sub, "file": "memory", **raw_m._ppinfo})
                ev_tsv = subdir / f"sub-{sub}_task-memory_events.tsv"
                for cond, ep in epochs_from_memory(raw_m, ev_tsv, args.cap).items():
                    save_epochs(ep, sub, cond, summary)
            conds = {r["cond"]: r["n_after"] for r in summary if r["sub"] == sub}
            print(f"  [{i+1}/{len(subs)}] sub-{sub} ({time.time()-t0:.0f}s): "
                  f"{ {k: conds[k] for k in sorted(conds)} }", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] sub-{sub}: {type(e).__name__}: {e}", flush=True)

    (OUT_ROOT / "preprocess_summary.json").write_text(json.dumps(summary, indent=1))
    (OUT_ROOT / "ica_reports.json").write_text(json.dumps(ica_reports, indent=1))
    tot_b = sum(r["n_before"] for r in summary)
    tot_a = sum(r["n_after"] for r in summary)
    print(f"\n[ds003838] selesai. {len(summary)} berkas kondisi, {time.time()-t_all:.0f}s. "
          f"Epoch {tot_b}->{tot_a} ({100*tot_a/max(tot_b,1):.0f}% retensi)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

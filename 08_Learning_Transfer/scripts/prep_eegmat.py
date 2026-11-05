#!/usr/bin/env python3
"""Plan B / Stage 08 — bawa EEGMAT (PhysioNet mental-arithmetic) ke ruang fitur 14-kanal kita.

EEGMAT: 36 subjek, 2 EDF/subjek -- Subject*_1.edf (rest/baseline, label low=0) dan
Subject*_2.edf (serial-subtraction arithmetic, label high=1). 10/20, ~19 EEG + ECG.
Label performa per-subjek (G=good/B=poor counters) dari subject-info.csv.

Harmonisasi ke 14 kanal Emotiv bersama (AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4):
  - rename 10/20 lama -> baru (T3->T7,T4->T8,T5->P7,T6->P8), strip prefiks "EEG ".
  - 10 kanal tersedia langsung; 4 hilang (AF3,AF4,FC5,FC6) direkonstruksi via
    interpolasi spherical-spline (tambah kanal kosong -> set montase -> interpolate_bads).
  - filter 0.5-45 Hz + notch 50 Hz, epoch 2 s non-overlap, resample 128 Hz.
Fitur identik 03_Features: PSD 70-d (relative+log), cov 14x14 (OAS), tangent global 105-d.
Output -> 03_Features/features/EEGMAT_workload_binary.npz (drop-in utk transfer_stats.py)
          + 08_Learning_Transfer/results/eegmat_meta.json (grup performa per subjek).

Penggunaan: python prep_eegmat.py --raw ../../01_Akuisisi_Data/raw/EEGMAT --out ../../03_Features/features
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np
import mne
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

mne.set_log_level("ERROR")

COMMON = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}  # 10/20 lama -> baru
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
TARGET_SFREQ = 128
EPOCH_S = 2.0


def clean_ch_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    ren = {}
    for ch in raw.ch_names:
        n = re.sub(r"^EEG\s+", "", ch).strip()
        n = n.replace("-A1", "").replace("-A2", "").replace("-LE", "").replace("-REF", "").strip()
        n = RENAME.get(n, n)
        ren[ch] = n
    raw.rename_channels(ren)
    return raw


def to_common14(raw: mne.io.BaseRaw):
    """Pilih EEG, set montase 10-20, interpolasi 4 kanal yang hilang -> 14 kanal urut COMMON."""
    # buang non-EEG (ECG/EKG/EOG, A1/A2 refs, dst.) -- nama bisa spt "ECG ECG", "A2"
    drop = [c for c in raw.ch_names
            if any(k in c.upper() for k in ("ECG", "EKG", "EOG"))
            or c.upper() in ("A1", "A2", "A2A1", "STATUS")]
    if drop:
        raw.drop_channels(drop)
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn")
    present = [c for c in COMMON if c in raw.ch_names]
    missing = [c for c in COMMON if c not in raw.ch_names]
    # tambah kanal hilang sebagai nol, tandai bad, lalu interpolate_bads (spherical spline)
    if missing:
        info_add = mne.create_info(missing, raw.info["sfreq"], "eeg")
        add = mne.io.RawArray(np.zeros((len(missing), raw.n_times)), info_add, verbose="ERROR")
        raw.add_channels([add], force_update_info=True)
        raw.set_montage(montage, match_case=False, on_missing="warn")
        raw.info["bads"] = list(missing)
        raw.interpolate_bads(reset_bads=True, mode="accurate")
    raw.pick(COMMON).reorder_channels(COMMON)
    return raw, present, missing


def epochs_from_raw(raw: mne.io.BaseRaw) -> mne.Epochs:
    raw.filter(0.5, 45.0, fir_design="firwin", verbose="ERROR")
    try:
        raw.notch_filter(50.0, verbose="ERROR")
    except Exception:
        pass
    ep = mne.make_fixed_length_epochs(raw, duration=EPOCH_S, overlap=0.0, preload=True, verbose="ERROR")
    if abs(ep.info["sfreq"] - TARGET_SFREQ) > 1e-3:
        ep.resample(TARGET_SFREQ)
    return ep


def band_power(ep: mne.Epochs) -> np.ndarray:
    sf = ep.info["sfreq"]; n_fft = int(2.0 * sf)
    spec = ep.compute_psd(method="welch", fmin=1, fmax=45, n_fft=n_fft, verbose="ERROR")
    psds, freqs = spec.get_data(return_freqs=True)
    total = psds.sum(-1, keepdims=True) + 1e-20
    feats = []
    for lo, hi in BANDS.values():
        idx = (freqs >= lo) & (freqs < hi)
        feats.append(psds[:, :, idx].sum(-1) / total[:, :, 0])
    return np.log(np.concatenate(feats, axis=1) + 1e-12).astype(np.float32)


def load_perf_groups(raw_dir: Path) -> dict:
    """subject-info.csv -> {subject_id: 'G'/'B'} (Count quality 1=G,0=B)."""
    f = raw_dir / "subject-info.csv"
    groups = {}
    if not f.exists():
        return groups
    for row in csv.DictReader(f.open()):
        sid = (row.get("Subject") or row.get("subject") or "").strip()
        q = (row.get("Count quality") or row.get("count quality") or "").strip()
        if sid:
            groups[sid] = "G" if q in ("1", "1.0") else "B"
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raw_dir = args.raw.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    perf = load_perf_groups(raw_dir)

    edfs = sorted(raw_dir.rglob("*.edf"))
    print(f"[EEGMAT] {len(edfs)} EDF; perf-labels for {len(perf)} subjects", flush=True)
    cov_est = Covariances(estimator="oas")
    X_psd, X_cov, subs, ys, tasks = [], [], [], [], []
    miss_report = None
    for f in edfs:
        m = re.search(r"(Subject\d+)_([12])", f.stem)
        if not m:
            print(f"  [skip] unparsed {f.name}"); continue
        sid, cond = m.group(1), m.group(2)
        y = 0 if cond == "1" else 1  # _1 rest=low, _2 arithmetic=high
        try:
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            raw = clean_ch_names(raw)
            raw, present, missing = to_common14(raw)
            if miss_report is None:
                miss_report = {"present": present, "interpolated": missing}
                print(f"  channels present={present}\n  interpolated={missing}", flush=True)
            ep = epochs_from_raw(raw)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}"); continue
        if len(ep) == 0:
            continue
        data = ep.get_data(copy=False)
        X_psd.append(band_power(ep))
        X_cov.append(cov_est.fit_transform(data * 1e6).astype(np.float32))
        n = len(ep); subs += [sid] * n; ys += [y] * n; tasks += [("rest" if y == 0 else "arith")] * n

    X_psd = np.concatenate(X_psd); X_cov = np.concatenate(X_cov)
    y = np.array(ys, int); subjects = np.array(subs)
    groups = np.array([f"EEGMAT-{s}" for s in subs])
    tangent = TangentSpace(metric="riemann").fit_transform(X_cov).astype(np.float32)
    fn = out / "EEGMAT_workload_binary.npz"
    np.savez_compressed(fn, X_psd=X_psd, X_cov=X_cov, X_tangent=tangent, y=y,
                        groups=groups, tasks=np.array(tasks), subjects=subjects)
    cls, cnt = np.unique(y, return_counts=True)
    print(f"[save] {fn.name}: {len(y)} epoch, {len(np.unique(subjects))} subj, "
          f"classes {dict(zip(cls.tolist(), cnt.tolist()))}", flush=True)

    meta = {"n_epochs": int(len(y)), "n_subjects": int(len(np.unique(subjects))),
            "channels": miss_report, "perf_groups": perf,
            "class_counts": {int(c): int(n) for c, n in zip(cls, cnt)}}
    (Path(__file__).resolve().parent.parent / "results" / "eegmat_meta.json").write_text(json.dumps(meta, indent=2))
    print("[meta] -> results/eegmat_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

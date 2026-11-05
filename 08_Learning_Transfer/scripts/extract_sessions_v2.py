#!/usr/bin/env python3
"""Re-extract COG-BCI graded-task features keeping SESSION, plus the full research montage,
to support: test-retest reliability (ICC across sessions) and the 14-ch vs full-montage cost.

For n-back (zero/one/twoBACK -> 0/1/2) and MATB (easy/med/diff -> 0/1/2), per cleaned epoch file:
  - 14-ch Emotiv-harmonised PSD (70) + covariance (14x14)
  - full research-montage PSD + covariance on the channels common to all graded files (~63)
keeping subject, session, level. Output -> 03_Features/features/COG-BCI_{nback,matb}_sess.npz
(holding both 14-ch and full-montage feature blocks + session labels).
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import mne
from pyriemann.estimation import Covariances
mne.set_log_level("ERROR")

BASE = Path(__file__).resolve().parent.parent.parent
DERIV = BASE / "02_Preprocessing" / "derivatives_v2" / "COG-BCI"
OUT = BASE / "03_Features" / "features_v2"
COMMON = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
SF = 128
NBACK = {"zeroBACK": 0, "oneBACK": 1, "twoBACK": 2}
MATB = {"MATBeasy": 0, "MATBmed": 1, "MATBdiff": 2}


def parse(path):
    n = path.stem
    sub = re.search(r"sub-(\w+?)_", n); ses = re.search(r"ses-(\w+?)_", n)
    task = re.search(r"task-(.+?)_run", n)
    return (sub.group(1) if sub else "?", ses.group(1) if ses else "?",
            task.group(1) if task else "?")


def bandpower(ep, picks=None):
    sf = ep.info["sfreq"]; n_fft = int(2 * sf)
    spec = ep.compute_psd(method="welch", fmin=1, fmax=45, n_fft=n_fft, picks=picks, verbose="ERROR")
    psds, freqs = spec.get_data(return_freqs=True)
    tot = psds.sum(-1, keepdims=True) + 1e-20
    feats = [psds[:, :, (freqs >= lo) & (freqs < hi)].sum(-1) / tot[:, :, 0] for lo, hi in BANDS.values()]
    return np.log(np.concatenate(feats, 1) + 1e-12).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)   # features/ digitignore -> tak ada di fresh clone
    files = sorted(DERIV.rglob("*_proc-clean_epo.fif"))
    # graded files only
    sel = []
    for f in files:
        _, _, task = parse(f)
        if task in NBACK or task in MATB:
            sel.append(f)
    print(f"[extract] {len(sel)} graded epoch files (dari {len(files)} total .fif di {DERIV})", flush=True)
    if not sel:
        print(f"ERROR: 0 file graded ditemukan. Cek bahwa {DERIV} berisi *_proc-clean_epo.fif "
              f"untuk task n-back/MATB (symlink derivatives ke Drive sudah benar?).", flush=True)
        return 1
    # pass 1: find EEG channels common to ALL selected files (for a consistent full montage)
    full = None
    for f in sel:
        try:
            info = mne.io.read_info(f, verbose="ERROR")
        except Exception:
            ep = mne.read_epochs(f, verbose="ERROR"); info = ep.info
        eeg = [info["ch_names"][i] for i in mne.pick_types(info, eeg=True)]
        full = set(eeg) if full is None else (full & set(eeg))
    full = [c for c in sorted(full)]
    print(f"[extract] full research montage = {len(full)} channels common to all", flush=True)

    cov14 = Covariances("oas"); covF = Covariances("oas")
    acc = {"nback": {}, "matb": {}}
    for fam in acc:
        acc[fam] = {k: [] for k in ("p14", "c14", "pF", "cF", "y", "subj", "sess")}
    for i, f in enumerate(sel):
        sub, ses, task = parse(f)
        fam, lvl = ("nback", NBACK[task]) if task in NBACK else ("matb", MATB[task])
        try:
            ep = mne.read_epochs(f, verbose="ERROR")
            ep.pick(full)                                   # restrict to common EEG set
            if abs(ep.info["sfreq"] - SF) > 1e-3:
                ep.resample(SF)
            ep14 = ep.copy().pick(COMMON).reorder_channels(COMMON)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}"); continue
        if len(ep) == 0:
            continue
        d14 = ep14.get_data(copy=False); dF = ep.get_data(copy=False)
        a = acc[fam]; n = len(ep)
        a["p14"].append(bandpower(ep14)); a["c14"].append(cov14.fit_transform(d14 * 1e6).astype(np.float32))
        a["pF"].append(bandpower(ep)); a["cF"].append(covF.fit_transform(dF * 1e6).astype(np.float32))
        a["y"] += [lvl] * n; a["subj"] += [sub] * n; a["sess"] += [ses] * n
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(sel)}", flush=True)

    for fam in acc:
        a = acc[fam]
        if not a["y"]:
            continue
        fn = OUT / f"COG-BCI_{fam}_sess.npz"
        np.savez_compressed(
            fn, X_psd14=np.concatenate(a["p14"]), X_cov14=np.concatenate(a["c14"]),
            X_psdF=np.concatenate(a["pF"]), X_covF=np.concatenate(a["cF"]),
            y=np.array(a["y"], int), subjects=np.array(a["subj"]), sessions=np.array(a["sess"]),
            full_channels=np.array(full))
        print(f"[save] {fn.name}: {len(a['y'])} epochs, full={len(full)}ch, "
              f"sessions={sorted(set(a['sess']))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

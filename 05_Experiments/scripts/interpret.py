#!/usr/bin/env python3
"""Interpretabilitas neurofisiologis (reproducible) — Cohen's d beban kerja low vs high.

Konvensi d: dihitung pada daya pita yang DIRATA-RATA antar-kanal lebih dulu (band_d), dan
per-kanal untuk pita alpha (utk topomap). FULL-cohort (tanpa subsample) agar angka stabil &
dapat direplikasi. Output -> results/interpretability.json (dipakai naskah + figures).

Penggunaan: python interpret.py --config ../config/experiments.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
COMMON_CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def d_pooled(lo, hi):
    p = np.sqrt(((lo.var(0, ddof=1) * (len(lo) - 1)) +
                 (hi.var(0, ddof=1) * (len(hi) - 1))) / (len(lo) + len(hi) - 2))
    return (hi.mean(0) - lo.mean(0)) / np.where(p == 0, np.nan, p)


def analyze(npz):
    d = np.load(npz, allow_pickle=True)
    X = d["X_psd"].astype(float).reshape(-1, 5, 14); y = d["y"].astype(int)
    lo, hi = X[y == 0], X[y == 1]
    band = d_pooled(lo.mean(2), hi.mean(2))                          # d per band (kanal dirata2)
    perch = d_pooled(lo.reshape(len(lo), -1), hi.reshape(len(hi), -1)).reshape(5, 14)
    alpha_ch = perch[BANDS.index("alpha")]
    top = sorted(zip(COMMON_CH, alpha_ch), key=lambda t: t[1])       # paling negatif dulu
    return {
        "n_low": int((y == 0).sum()), "n_high": int((y == 1).sum()),
        "band_d": {b: round(float(v), 2) for b, v in zip(BANDS, band)},
        "alpha_channel_d": {c: round(float(v), 2) for c, v in zip(COMMON_CH, alpha_ch)},
        "alpha_top_channels": [[c, round(float(v), 2)] for c, v in top[:4]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    feat = (base / cfg["features_root"]).resolve()
    out = (base / cfg["output_root"]).resolve(); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for label, f in [("COG-BCI workload(extreme)", "COG-BCI_workload_extreme.npz"),
                     ("STEW workload", "STEW_workload_binary.npz")]:
        npz = feat / f
        if not npz.exists():
            print(f"[skip] {label}"); continue
        res[label] = analyze(npz)
        print(f"{label}: {res[label]['band_d']}")
        print(f"  alpha top channels: {res[label]['alpha_top_channels']}")
    (out / "interpretability.json").write_text(json.dumps(res, indent=2))
    print(f"\nHasil -> {out / 'interpretability.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

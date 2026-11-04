#!/usr/bin/env python3
"""Uji signifikansi klaim transfer (PSD, cross-subject GroupKFold).

Per-fold macro-F1 untuk SINGLE / MULTI(naive) / MULTI+ccDA(mean per-kelas) pada skema
workload (binary & extreme), lalu:
  - one-sample test gain (multi-single) vs 0  -> apakah transfer signifikan (negatif),
  - paired test ccDA vs naive                 -> apakah class-conditional DA membantu.
Hasil -> results/transfer_stats.json.

Penggunaan: python transfer_stats.py --config ../config/experiments.json
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


def load(npz, cap, seed):
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]; n = len(y)
    idx = np.random.default_rng(seed).choice(n, cap, replace=False) if cap and n > cap else np.arange(n)
    return d["X_psd"][idx], y[idx], g[idx]


def ccda_mean(Xs, ys, Xt, yt):
    Xa = Xs.astype(float).copy()
    for c in np.unique(yt):
        if (ys == c).sum() and (yt == c).sum():
            Xa[ys == c] += Xt[yt == c].mean(0) - Xs[ys == c].mean(0)
    return Xa


def probe(Xtr, ytr, Xte, cw):
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight=cw))
    return p.fit(Xtr, ytr).predict(Xte)


def per_fold(target, source, scheme, cfg, feat, k, cap):
    cw = cfg.get("class_weight"); seed = cfg["cv"].get("random_state", 42)
    Xt, yt, gt = load(feat / f"{target}_{scheme}.npz", cap, seed)
    Xs, ys, _ = load(feat / f"{source}_{scheme}.npz", cap, seed)
    R = {"single": [], "multi": [], "ccda": []}
    for tr, te in GroupKFold(min(k, len(np.unique(gt)))).split(Xt, yt, gt):
        R["single"].append(f1_score(yt[te], probe(Xt[tr], yt[tr], Xt[te], cw), average="macro"))
        Xm, ym = np.vstack([Xt[tr], Xs]), np.concatenate([yt[tr], ys])
        R["multi"].append(f1_score(yt[te], probe(Xm, ym, Xt[te], cw), average="macro"))
        Xa = np.vstack([Xt[tr], ccda_mean(Xs, ys, Xt[tr], yt[tr])])
        R["ccda"].append(f1_score(yt[te], probe(Xa, ym, Xt[te], cw), average="macro"))
    return {k2: np.array(v) for k2, v in R.items()}


def tests(a, b):
    try:
        w = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w = float("nan")
    d = a - b; k = len(d); var = float(d.var(ddof=1))
    if var > 0:                                  # corrected resampled t-test (Nadeau & Bengio 2003)
        nb_t = float(d.mean()) / np.sqrt(var * (1.0 / k + 1.0 / (k - 1)))
        nb_p = round(2.0 * float(stats.t.sf(abs(nb_t), k - 1)), 4)
    else:
        nb_p = 0.0
    return {"mean_diff": round(float((a - b).mean()), 3),
            "paired_t_p": round(float(stats.ttest_rel(a, b).pvalue), 4),
            "nadeau_bengio_p": nb_p,
            "wilcoxon_p": round(w, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cap", type=int, default=8000)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    feat = (base / cfg["features_root"]).resolve()
    out = (base / cfg["output_root"]).resolve(); out.mkdir(parents=True, exist_ok=True)
    zero = np.zeros(args.k)
    res = {}
    for target, source in [("STEW", "COG-BCI"), ("COG-BCI", "STEW")]:
        for scheme in ["workload_binary", "workload_extreme"]:
            print(f"\n=== {target} (+{source}), {scheme} ===", flush=True)
            R = per_fold(target, source, scheme, cfg, feat, args.k, args.cap)
            gain = R["multi"] - R["single"]
            row = {
                "n_folds": len(R["single"]),
                "single": round(float(R["single"].mean()), 3),
                "multi": round(float(R["multi"].mean()), 3),
                "ccda": round(float(R["ccda"].mean()), 3),
                "gain_vs0": tests(gain, zero[:len(gain)]),       # apakah naive transfer != 0
                "ccda_vs_naive": tests(R["ccda"], R["multi"]),    # apakah ccDA > naive
            }
            res[f"{target}_{scheme}"] = row
            print(f"  single={row['single']} multi={row['multi']} ccda={row['ccda']}")
            print(f"  gain(multi-single) {row['gain_vs0']['mean_diff']:+} vs0: t-p={row['gain_vs0']['paired_t_p']} W-p={row['gain_vs0']['wilcoxon_p']}")
            print(f"  ccDA-naive {row['ccda_vs_naive']['mean_diff']:+}: t-p={row['ccda_vs_naive']['paired_t_p']} W-p={row['ccda_vs_naive']['wilcoxon_p']}")
    (out / "transfer_stats.json").write_text(json.dumps(res, indent=2))
    print(f"\nHasil -> {out / 'transfer_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

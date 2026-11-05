#!/usr/bin/env python3
"""TINE strengthening — analyses on existing 14-ch features (no re-extraction).

T1.2 Temporal dynamics: how does smoothing window length trade off separation vs variance?
T1.4 Channel ablation: how few channels (band-power view) still track graded load?
T1.5 In-silico adaptive loop: does a simple hysteresis controller on the index behave stably?
Output -> results/strengthen_light.json
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
from agents import build_decoder  # noqa: E402
FEAT = BASE / "03_Features" / "features_v3"
RES = Path(__file__).resolve().parent.parent / "results_v3"
COMMON = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
CFG = {"specialists": {"spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
                       "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
                       "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6}},
       "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3}}
GRADED = {"COG-BCI_nback_graded": "n-back", "COG-BCI_matb_graded": "MATB"}


def load(name, cap=8000, seed=42):
    d = np.load(FEAT / f"{name}.npz", allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]; sub = d["subjects"]
    if cap and len(y) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(len(y), cap, replace=False))
        return {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx], sub[idx]
    return {"psd": d["X_psd"], "cov": d["X_cov"]}, y, g, sub


def oof_index(views, y, g, k=8):
    phi = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(k, len(np.unique(g)))).split(views["psd"], y, g):
        vtr = {kk: v[tr] for kk, v in views.items()}; vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(CFG, "balanced"); dec.fit(vtr, y[tr], groups=g[tr])
        phi[te] = dec.predict_proba(vte) @ np.array(dec.classes_)
    return phi


# ---------- T1.2 temporal window ----------
def temporal(name):
    views, y, g, sub = load(name)
    phi = oof_index(views, y, g)
    out = {}
    for W in [1, 2, 4, 10]:                       # epochs; x2 s = 2,4,8,20 s
        wi, wl = [], []
        for s in np.unique(sub):
            for l in np.unique(y):
                v = phi[(sub == s) & (y == l)]
                nb = len(v) // W
                if nb == 0:
                    continue
                blocks = v[:nb * W].reshape(nb, W).mean(1)
                wi += blocks.tolist(); wl += [l] * nb
        wi = np.array(wi); wl = np.array(wl)
        rho = stats.spearmanr(wi, wl).correlation
        within_sd = float(np.mean([wi[wl == l].std() for l in np.unique(wl)]))
        out[W * 2] = {"seconds": W * 2, "spearman": round(float(rho), 3),
                      "within_level_sd": round(within_sd, 3), "n_windows": int(len(wi))}
    return out


# ---------- T1.4 channel ablation (band-power view) ----------
def psd_cols(channels):
    idx = [COMMON.index(c) for c in channels]
    return [b * 14 + c for b in range(5) for c in idx]


def ablation(name, k=8):
    d = np.load(FEAT / f"{name}.npz", allow_pickle=True)
    X = d["X_psd"]; y = d["y"].astype(int); g = d["groups"]
    cap = 8000
    if len(y) > cap:
        ix = np.sort(np.random.default_rng(42).choice(len(y), cap, replace=False))
        X, y, g = X[ix], y[ix], g[ix]
    subsets = {
        "2 (O1,O2)": ["O1", "O2"],
        "4 posterior (O1,O2,P7,P8)": ["O1", "O2", "P7", "P8"],
        "6 (+T7,T8)": ["O1", "O2", "P7", "P8", "T7", "T8"],
        "10 (+frontal)": ["O1", "O2", "P7", "P8", "T7", "T8", "F3", "F4", "F7", "F8"],
        "14 (all)": COMMON,
    }
    res = {}
    for label, chans in subsets.items():
        cols = psd_cols(chans)
        phi = np.full(len(y), np.nan)
        for tr, te in GroupKFold(min(k, len(np.unique(g)))).split(X, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(X[tr][:, cols], y[tr])
            phi[te] = clf.predict_proba(X[te][:, cols]) @ np.array(clf.classes_)
        res[label] = {"n_channels": len(chans), "spearman": round(float(stats.spearmanr(phi, y).correlation), 3)}
    return res


# ---------- T1.5 in-silico hysteresis controller ----------
def insilico(name):
    views, y, g, sub = load(name)
    phi = oof_index(views, y, g)
    # smooth with 4-epoch (8 s) window per subject (time order preserved within subject)
    lo_q, hi_q = np.nanpercentile(phi, 33), np.nanpercentile(phi, 67)
    margins = {"no hysteresis": 0.0, "hysteresis (0.5 SD)": 0.5 * np.nanstd(phi)}
    out = {}
    for mlabel, m in margins.items():
        switches, minutes, track = 0, 0.0, []
        for s in np.unique(sub):
            v = phi[sub == s]
            if len(v) < 8:
                continue
            sm = np.convolve(v, np.ones(4) / 4, mode="valid")     # 8 s smoothing
            state = 1                                              # 0 lower,1 hold,2 raise
            seq = []
            for x in sm:
                if x > hi_q + m and state != 2:
                    state = 2; switches += 1
                elif x < lo_q - m and state != 0:
                    state = 0; switches += 1
                elif lo_q <= x <= hi_q and state != 1:
                    state = 1; switches += 1
                seq.append(state)
            minutes += len(sm) * 2.0 / 60.0                       # 2 s/epoch
        out[mlabel] = {"switches_per_min": round(switches / max(minutes, 1e-9), 2)}
    out["thresholds"] = {"low_q33": round(float(lo_q), 3), "high_q67": round(float(hi_q), 3)}
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = {"temporal": {}, "ablation": {}, "insilico": {}}
    for name, lab in GRADED.items():
        print(f"=== {lab}: temporal ===", flush=True)
        out["temporal"][lab] = temporal(name)
        for w, r in out["temporal"][lab].items():
            print(f"  {r['seconds']}s: Spearman={r['spearman']} within-level SD={r['within_level_sd']}", flush=True)
        print(f"=== {lab}: ablation ===", flush=True)
        out["ablation"][lab] = ablation(name)
        for l, r in out["ablation"][lab].items():
            print(f"  {l}: rho={r['spearman']}", flush=True)
        print(f"=== {lab}: in-silico loop ===", flush=True)
        out["insilico"][lab] = insilico(name)
        print(f"  {out['insilico'][lab]}", flush=True)
    (RES / "strengthen_light.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RES/'strengthen_light.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

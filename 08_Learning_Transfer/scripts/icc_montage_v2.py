#!/usr/bin/env python3
"""TINE strengthening (heavy): uses COG-BCI_{nback,matb}_sess.npz from extract_sessions.py.

T1.1 Test-retest reliability: ICC(2,1) of the per-(subject,session) mean load index across
     COG-BCI sessions -- does the index behave as a stable measurement instrument day to day?
T1.3 Hardware cost: same decoder/index on the full research montage vs the 14-ch consumer montage
     (macro-F1, Spearman, ordinal MAE) -- what is sacrificed for cheap hardware?
Output -> results/icc_montage.json
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
from agents import build_decoder  # noqa: E402
FEAT = BASE / "03_Features" / "features_v2"
RES = Path(__file__).resolve().parent.parent / "results_v2"
CFG = {"specialists": {"spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
                       "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
                       "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6}},
       "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3}}


def icc21(M):
    """ICC(2,1) two-way random, single measures. M: rows=subjects, cols=sessions (no NaN)."""
    n, k = M.shape
    gm = M.mean()
    MSR = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    MSC = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    resid = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm
    MSE = (resid ** 2).sum() / ((n - 1) * (k - 1))
    return float((MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n))


def oof_index(views, y, g, cap=8000, k=8):
    if cap and len(y) > cap:
        ix = np.sort(np.random.default_rng(42).choice(len(y), cap, replace=False))
        views = {kk: v[ix] for kk, v in views.items()}; y = y[ix]; g = g[ix]; sub_idx = ix
    else:
        sub_idx = np.arange(len(y))
    phi = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(k, len(np.unique(g)))).split(views["psd"], y, g):
        vtr = {kk: v[tr] for kk, v in views.items()}; vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(CFG, "balanced"); dec.fit(vtr, y[tr], groups=g[tr])
        phi[te] = dec.predict_proba(vte) @ np.array(dec.classes_)
    return phi, y, g, sub_idx


def decode_metrics(views, y, g, k=8, cap=8000):
    if cap and len(y) > cap:
        ix = np.sort(np.random.default_rng(42).choice(len(y), cap, replace=False))
        views = {kk: v[ix] for kk, v in views.items()}; y = y[ix]; g = g[ix]
    f1, phi = [], np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(k, len(np.unique(g)))).split(views["psd"], y, g):
        vtr = {kk: v[tr] for kk, v in views.items()}; vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(CFG, "balanced"); dec.fit(vtr, y[tr], groups=g[tr])
        cls = np.array(dec.classes_); pr = dec.predict_proba(vte)
        phi[te] = pr @ cls
        f1.append(f1_score(y[te], cls[np.argmax(pr, 1)], average="macro"))
    return {"macro_f1": round(float(np.mean(f1)), 3),
            "spearman": round(float(stats.spearmanr(phi, y).correlation), 3),
            "ordinal_mae": round(float(np.mean(np.abs(np.rint(phi) - y))), 3)}


def run(fam):
    d = np.load(FEAT / f"COG-BCI_{fam}_sess.npz", allow_pickle=True)
    y = d["y"].astype(int); subj = d["subjects"]; sess = d["sessions"]
    g = np.array([f"s{s}" for s in subj])
    full_ch = list(d["full_channels"])
    out = {"n_full_channels": len(full_ch)}

    # ---- T1.3: 14-ch vs full montage ----
    v14 = {"psd": d["X_psd14"], "cov": d["X_cov14"]}
    vF = {"psd": d["X_psdF"], "cov": d["X_covF"]}
    out["consumer_14ch"] = decode_metrics(v14, y, g)
    out["full_montage"] = decode_metrics(vF, y, g)

    # ---- T1.1: test-retest ICC of per-(subject,session) mean index (14-ch) ----
    phi, y2, g2, ix = oof_index(v14, y, g)
    subj2, sess2 = subj[ix], sess[ix]
    sessions = sorted(set(sess2))
    subs = sorted(set(subj2))
    # per (subject, session) mean index, and the high-minus-low contrast
    def mat(metric):
        M = np.full((len(subs), len(sessions)), np.nan)
        for i, s in enumerate(subs):
            for j, se in enumerate(sessions):
                m = (subj2 == s) & (sess2 == se)
                if m.sum() == 0:
                    continue
                if metric == "mean":
                    M[i, j] = phi[m].mean()
                else:  # high-low contrast
                    hi = phi[m & (y2 == y2.max())]; lo = phi[m & (y2 == y2.min())]
                    if len(hi) and len(lo):
                        M[i, j] = hi.mean() - lo.mean()
        return M
    res_icc = {}
    for metric in ("mean", "contrast"):
        M = mat(metric)
        keep = ~np.isnan(M).any(1)                       # subjects present in all sessions
        if keep.sum() >= 5:
            res_icc[metric] = {"icc21": round(icc21(M[keep]), 3),
                               "n_subjects": int(keep.sum()), "n_sessions": len(sessions)}
    out["test_retest"] = res_icc
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    for fam in ("nback", "matb"):
        f = FEAT / f"COG-BCI_{fam}_sess.npz"
        if not f.exists():
            print(f"[skip] {fam}: {f.name} missing"); continue
        print(f"=== {fam} ===", flush=True)
        out[fam] = run(fam)
        o = out[fam]
        print(f"  14-ch: {o['consumer_14ch']} | full({o['n_full_channels']}ch): {o['full_montage']}", flush=True)
        print(f"  test-retest ICC: {o['test_retest']}", flush=True)
    (RES / "icc_montage.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RES/'icc_montage.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

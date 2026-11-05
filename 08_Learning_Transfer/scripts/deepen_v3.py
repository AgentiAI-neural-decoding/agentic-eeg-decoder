#!/usr/bin/env python3
"""Plan B / TINE paper — deepening analyses (existing data, lightweight).

(1) Per-participant reliability of the continuous load index (distribution of within-subject
    Spearman; fraction monotonic) -- robustness beyond the pooled correlation.
(2) Alpha-only baseline: does the multi-specialist index beat a textbook posterior-alpha index
    at tracking graded load? (the obvious reviewer question)
(3) Cross-task generalisation: train a binary low/high-load decoder on one graded task, test on the
    other (n-back <-> MATB) -- evidence for a task-general, deployable load decoder.
(4) Neurophysiological dose-response: posterior alpha and frontal theta vs load level (CLT mechanism).
Output -> results/deepen.json
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
FEAT = BASE / "03_Features" / "features_v3"
RES = Path(__file__).resolve().parent.parent / "results_v3"
CFG = {"specialists": {"spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
                       "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
                       "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6}},
       "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3}}
GRADED = {"COG-BCI_nback_graded": "n-back", "COG-BCI_matb_graded": "MATB"}
# COMMON channel order; posterior alpha + frontal theta indices
COMMON = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
POST = [COMMON.index(c) for c in ["P7", "P8", "O1", "O2"]]
FRONT = [COMMON.index(c) for c in ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6"]]


def load(name, cap=8000, seed=42):
    d = np.load(FEAT / f"{name}.npz", allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]
    if cap and len(y) > cap:
        idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
        return ({"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx])
    return ({"psd": d["X_psd"], "cov": d["X_cov"]}, y, g)


def oof_index(views, y, g, k=8):
    phi = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(k, len(np.unique(g)))).split(views["psd"], y, g):
        vtr = {kk: v[tr] for kk, v in views.items()}; vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(CFG, "balanced"); dec.fit(vtr, y[tr], groups=g[tr])
        cls = np.array(dec.classes_)
        phi[te] = dec.predict_proba(vte) @ cls
    return phi


def reliability_and_baseline(name):
    views, y, g = load(name)
    phi = oof_index(views, y, g)                       # decoder continuous index
    Xa = views["psd"].reshape(-1, 5, 14)
    alpha_post = Xa[:, 2, :][:, POST].mean(1)          # log-rel posterior alpha (decreases w/ load)
    alpha_idx = -alpha_post                            # flip so higher = more load
    # per-subject within-subject Spearman(index, level)
    rhos, monos = [], []
    for s in np.unique(g):
        m = g == s
        if len(np.unique(y[m])) < 2:
            continue
        rhos.append(float(stats.spearmanr(phi[m], y[m]).correlation))
        per = [phi[m & (y == l)].mean() for l in [0, 1, 2] if (m & (y == l)).any()]
        monos.append(bool(np.all(np.diff(per) > 0)))
    rhos = np.array(rhos)
    return {
        "n_subjects": int(len(rhos)),
        "per_subject_rho_median": round(float(np.median(rhos)), 3),
        "per_subject_rho_iqr": [round(float(np.percentile(rhos, 25)), 3),
                                round(float(np.percentile(rhos, 75)), 3)],
        "pct_subjects_positive": round(float((rhos > 0).mean() * 100), 1),
        "pct_subjects_monotonic": round(float(np.mean(monos) * 100), 1),
        "decoder_index_rho_pooled": round(float(stats.spearmanr(phi, y).correlation), 3),
        "alpha_only_rho_pooled": round(float(stats.spearmanr(alpha_idx, y).correlation), 3),
    }


def dose_response(name):
    views, y, _ = load(name)
    Xa = views["psd"].reshape(-1, 5, 14)
    out = {}
    for l in [0, 1, 2]:
        m = y == l
        out[l] = {"posterior_alpha": round(float(Xa[m, 2, :][:, POST].mean()), 3),
                  "frontal_theta": round(float(Xa[m, 1, :][:, FRONT].mean()), 3)}
    return out


def cross_task(src, tgt, cap=8000):
    """Binary low(level0)/high(level2) load: train on src, test on tgt (different task + subjects)."""
    vs, ys, gs = load(src, cap=cap)
    vt, yt, gt = load(tgt, cap=cap)
    def binarize(v, y):
        keep = np.isin(y, [0, 2]); yb = (y[keep] == 2).astype(int)
        return {k: a[keep] for k, a in v.items()}, yb
    vs2, ys2 = binarize(vs, ys); vt2, yt2 = binarize(vt, yt)
    dec = build_decoder(CFG, "balanced"); dec.fit(vs2, ys2)
    proba = dec.predict_proba(vt2); hi = list(dec.classes_).index(1)
    pred = dec.classes_[np.argmax(proba, 1)]
    return {"macro_f1": round(float(f1_score(yt2, pred, average="macro")), 3),
            "index_spearman_vs_load": round(float(stats.spearmanr(proba[:, hi], yt2).correlation), 3),
            "n_test": int(len(yt2))}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = {"reliability_baseline": {}, "dose_response": {}, "cross_task": {}}
    for name, lab in GRADED.items():
        print(f"=== reliability/baseline: {lab} ===", flush=True)
        r = reliability_and_baseline(name); out["reliability_baseline"][lab] = r
        print(f"  per-subj rho median={r['per_subject_rho_median']} IQR={r['per_subject_rho_iqr']} "
              f"| {r['pct_subjects_positive']}% positive, {r['pct_subjects_monotonic']}% monotonic", flush=True)
        print(f"  decoder index rho={r['decoder_index_rho_pooled']} vs alpha-only rho={r['alpha_only_rho_pooled']}", flush=True)
        out["dose_response"][lab] = dose_response(name)
        print(f"  dose-response (alpha/theta by level): {out['dose_response'][lab]}", flush=True)
    print("=== cross-task generalisation (binary low/high load) ===", flush=True)
    out["cross_task"]["nback->MATB"] = cross_task("COG-BCI_nback_graded", "COG-BCI_matb_graded")
    out["cross_task"]["MATB->nback"] = cross_task("COG-BCI_matb_graded", "COG-BCI_nback_graded")
    for k, v in out["cross_task"].items():
        print(f"  {k}: macro-F1={v['macro_f1']} index-Spearman={v['index_spearman_vs_load']}", flush=True)
    (RES / "deepen.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RES / 'deepen.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

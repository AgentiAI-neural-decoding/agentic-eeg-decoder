#!/usr/bin/env python3
"""Plan B / Stage 08 — FULL learning-context analysis on EEGMAT (orchestrator + neurophysiology).

(1) Within-task load decode with the real toolkit (spectral/Riemannian/CSP + orchestrator),
    cross-subject GroupKFold -> best macro-F1 (vs the weak PSD+LogReg probe).
(2) Neurophysiological grounding in the LEARNING task: Cohen's d (arithmetic - rest) per band
    (channel-averaged), plus frontal theta-increase and parieto-occipital alpha-desync.
(3) Level-b actionability (proper): per-subject indices vs performance (Count quality G/B and
    Number of subtractions):
      - decoded-load index = OOF orchestrator P(high) on arithmetic epochs,
      - neuro load index = frontal theta increase, parieto-occipital alpha decrease.
Output -> results/eegmat_full.json
"""
from __future__ import annotations
import csv, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
from agents import build_decoder  # noqa: E402

FEAT = BASE / "03_Features" / "features"
RES = Path(__file__).resolve().parent.parent / "results"
RAW = BASE / "01_Akuisisi_Data" / "raw" / "EEGMAT" / "eeg-during-mental-arithmetic-tasks-1.0.0"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
COMMON = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
FRONTAL = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6"]
PARIETO_OCC = ["P7", "P8", "O1", "O2"]
CFG = {"specialists": {"spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
                       "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
                       "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6}},
       "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3}}


def load():
    d = np.load(FEAT / "EEGMAT_workload_binary.npz", allow_pickle=True)
    return {"psd": d["X_psd"], "cov": d["X_cov"]}, d["y"].astype(int), d["groups"], d["subjects"]


def perf_tables():
    g, subt = {}, {}
    f = RAW / "subject-info.csv"
    if f.exists():
        for r in csv.DictReader(f.open()):
            s = r["Subject"].strip()
            g[s] = "G" if (r.get("Count quality", "").strip() in ("1", "1.0")) else "B"
            try:
                subt[s] = float(r["Number of subtractions"])
            except (KeyError, ValueError):
                pass
    return g, subt


def within_decode(views, y, groups, k=10):
    gkf = GroupKFold(min(k, len(np.unique(groups))))
    rows = {f"spec:{n}": [] for n in CFG["specialists"]}; rows["orchestrator"] = []
    for tr, te in gkf.split(views["psd"], y, groups):
        vtr = {kk: v[tr] for kk, v in views.items()}; vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(CFG, "balanced"); dec.fit(vtr, y[tr], groups=groups[tr])
        for s in dec.specialists:
            pred = s.classes_[np.argmax(s.predict_proba(vte), axis=1)]
            rows[f"spec:{s.name}"].append(f1_score(y[te], pred, average="macro"))
        rows["orchestrator"].append(f1_score(y[te], dec.predict(vte), average="macro"))
    return {kk: round(float(np.mean(v)), 3) for kk, v in rows.items()}


def neuro(views, y, subjects):
    """Cohen's d (arith - rest) on log band-power averaged over channels; per-subject load indices."""
    X = views["psd"].reshape(-1, 5, 14)
    lo, hi = X[y == 0], X[y == 1]

    def d_pooled(a, b):
        p = np.sqrt(((a.var(0, ddof=1) * (len(a) - 1)) + (b.var(0, ddof=1) * (len(b) - 1))) /
                    (len(a) + len(b) - 2))
        return (b.mean(0) - a.mean(0)) / np.where(p == 0, np.nan, p)

    band_d = d_pooled(lo.mean(2), hi.mean(2))           # per band (channel-averaged)
    fi = [COMMON.index(c) for c in FRONTAL]; pi = [COMMON.index(c) for c in PARIETO_OCC]
    ti = BANDS.index("theta"); ai = BANDS.index("alpha")
    # per-subject load-increase indices (arith - rest)
    idx = {}
    for s in np.unique(subjects):
        r = X[(subjects == s) & (y == 0)]; a = X[(subjects == s) & (y == 1)]
        if len(r) == 0 or len(a) == 0:
            continue
        theta_inc = float(a[:, ti, :][:, fi].mean() - r[:, ti, :][:, fi].mean())
        alpha_dec = float(a[:, ai, :][:, pi].mean() - r[:, ai, :][:, pi].mean())  # expect <0
        idx[s] = (theta_inc, alpha_dec)
    return {b: round(float(v), 2) for b, v in zip(BANDS, band_d)}, idx


def oof_load(views, y, groups, subjects, k=10):
    gkf = GroupKFold(min(k, len(np.unique(groups))))
    phi = np.zeros(len(y))
    for tr, te in gkf.split(views["psd"], y, groups):
        vtr = {kk: v[tr] for kk, v in views.items()}; vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(CFG, "balanced"); dec.fit(vtr, y[tr], groups=groups[tr])
        pr = dec.predict_proba(vte); hi = list(dec.classes_).index(1)
        phi[te] = pr[:, hi]
    out = {}
    for s in np.unique(subjects):
        m = (subjects == s) & (y == 1)
        if m.sum():
            out[s] = float(phi[m].mean())
    return out


def assoc(index_by_subj, g, subt, label):
    subs = [s for s in index_by_subj]
    val = np.array([index_by_subj[s] for s in subs])
    grp = np.array([g.get(s, "?") for s in subs])
    G, B = val[grp == "G"], val[grp == "B"]
    sc = np.array([subt.get(s, np.nan) for s in subs]); ok = ~np.isnan(sc)
    res = {"label": label, "n": len(subs),
           "G_mean": round(float(G.mean()), 3) if len(G) else None,
           "B_mean": round(float(B.mean()), 3) if len(B) else None,
           "GvsB_mwu_p": round(float(stats.mannwhitneyu(G, B).pvalue), 4) if len(G) and len(B) else None}
    if ok.sum() > 5:
        sp = stats.spearmanr(val[ok], sc[ok])
        res["spearman_subtractions"] = round(float(sp.correlation), 3)
        res["spearman_p"] = round(float(sp.pvalue), 4)
    return res


def main():
    RES.mkdir(parents=True, exist_ok=True)
    views, y, groups, subjects = load()
    g, subt = perf_tables()
    out = {}

    print("=== (1) within-task load decode (full toolkit) ===", flush=True)
    out["within_decode"] = within_decode(views, y, groups)
    print("  ", out["within_decode"], flush=True)

    print("=== (2) neurophysiology in the learning task (Cohen's d arith-rest) ===", flush=True)
    band_d, neuro_idx = neuro(views, y, subjects)
    out["band_d_arith_minus_rest"] = band_d
    print("  band d:", band_d, flush=True)

    print("=== (3) level-b: per-subject indices vs performance ===", flush=True)
    dl = oof_load(views, y, groups, subjects)
    out["assoc_decoded_load"] = assoc(dl, g, subt, "OOF orchestrator P(high) on arithmetic")
    out["assoc_theta_increase"] = assoc({s: v[0] for s, v in neuro_idx.items()}, g, subt, "frontal theta increase")
    out["assoc_alpha_desync"] = assoc({s: v[1] for s, v in neuro_idx.items()}, g, subt, "parieto-occipital alpha change")
    for k in ("assoc_decoded_load", "assoc_theta_increase", "assoc_alpha_desync"):
        r = out[k]
        print(f"  {r['label']}: G={r['G_mean']} B={r['B_mean']} MWU-p={r['GvsB_mwu_p']} | "
              f"Spearman(#subtr) rho={r.get('spearman_subtractions')} p={r.get('spearman_p')}", flush=True)

    (RES / "eegmat_full.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RES / 'eegmat_full.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

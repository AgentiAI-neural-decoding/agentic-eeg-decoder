"""stew_rating_v2.py - does the graded MATB index transfer to a consumer headset?

Train the graded MATB decoder on clean COG-BCI (20 unique subjects, research-grade,
14-channel harmonised features), apply it unchanged to STEW (Emotiv EPOC, 47 subjects,
rest vs SIMKAP multitasking), and test the resulting continuous index against
(a) the task contrast within each subject and (b) the subjective workload ratings (1-9)
that STEW participants gave for each condition.

Comparators on the same STEW epochs:
  alpha_only   : negative posterior alpha log-relative power (no training at all).
  stew_matched : the same decoder trained within STEW (GroupKFold-8 out-of-fold, binary
                 rest vs task), i.e. what a device-matched model achieves.

Family of tests fixed before running (per comparator):
  T1 hi > lo within subject (Wilcoxon signed-rank, one-sided, subject means)
  T2 Spearman of delta-index (hi - lo) with delta-rating (hi - lo) across subjects
  T3 Spearman of index with rating over subject-condition means (2 per subject; descriptive)
Inputs : 03_Features/features_v3/{COG-BCI_matb_graded,STEW_workload_binary}.npz,
         01_Akuisisi_Data/raw/STEW/ratings.txt (subject, rest rating, task rating)
Output : 08_Learning_Transfer/results_v3/stew_rating.json
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
from agents import build_decoder  # noqa: E402

warnings.filterwarnings("ignore")
FEAT = BASE / "03_Features" / "features_v3"
RATINGS = BASE / "01_Akuisisi_Data" / "raw" / "STEW" / "ratings.txt"
RES = Path(__file__).resolve().parent.parent / "results_v3"
CFG = {"specialists": {"spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
                       "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
                       "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6}},
       "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3}}
COMMON = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
POST = [COMMON.index(c) for c in ["P7", "P8", "O1", "O2"]]
CAP, SEED = 9000, 42


def load(name, cap=None):
    d = np.load(FEAT / f"{name}.npz", allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]; t = d["tasks"]
    idx = np.arange(len(y))
    if cap and len(y) > cap:
        idx = np.random.default_rng(SEED).choice(len(y), cap, replace=False)
    return {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx], t[idx]


def read_ratings():
    out = {}
    for line in RATINGS.read_text().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) == 3:
            s, lo, hi = parts
            out[f"STEW-{int(s):02d}"] = {"lo": float(lo), "hi": float(hi)}
    return out


def evaluate(index, is_hi, g, ratings, label):
    subs = sorted(set(g))
    rows = []
    for s in subs:
        m = g == s
        if not (m & is_hi).any() or not (m & ~is_hi).any():
            continue
        rows.append({"subject": s, "idx_lo": float(index[m & ~is_hi].mean()),
                     "idx_hi": float(index[m & is_hi].mean())})
    d_idx = np.array([r["idx_hi"] - r["idx_lo"] for r in rows])
    w = stats.wilcoxon(d_idx, alternative="greater")
    rated = [r for r in rows if r["subject"] in ratings]
    d_rate = np.array([ratings[r["subject"]]["hi"] - ratings[r["subject"]]["lo"] for r in rated])
    d_idx_r = np.array([r["idx_hi"] - r["idx_lo"] for r in rated])
    t2 = stats.spearmanr(d_idx_r, d_rate)
    pooled_idx = np.array([v for r in rated for v in (r["idx_lo"], r["idx_hi"])])
    pooled_rate = np.array([v for r in rated for v in (ratings[r["subject"]]["lo"], ratings[r["subject"]]["hi"])])
    t3 = stats.spearmanr(pooled_idx, pooled_rate)
    auc = roc_auc_score(is_hi.astype(int), index)
    res = {"n_subjects": len(rows), "n_rated": len(rated),
           "epoch_auc_hi_vs_lo": round(float(auc), 3),
           "T1_pct_subjects_hi_gt_lo": round(float(100 * np.mean(d_idx > 0)), 1),
           "T1_median_delta_index": round(float(np.median(d_idx)), 4),
           "T1_wilcoxon_p_greater": float(w.pvalue),
           "T2_spearman_delta_index_vs_delta_rating": {"rho": round(float(t2.correlation), 3), "p": float(t2.pvalue)},
           "T3_spearman_index_vs_rating_pooled": {"rho": round(float(t3.correlation), 3), "p": float(t3.pvalue)}}
    print(f"  {label:14s} AUC={res['epoch_auc_hi_vs_lo']} hi>lo {res['T1_pct_subjects_hi_gt_lo']}% "
          f"p={res['T1_wilcoxon_p_greater']:.2g} | d-index vs d-rating rho={res['T2_spearman_delta_index_vs_delta_rating']['rho']} "
          f"p={res['T2_spearman_delta_index_vs_delta_rating']['p']:.2g} | pooled rho={res['T3_spearman_index_vs_rating_pooled']['rho']}", flush=True)
    return res


def main():
    RES.mkdir(parents=True, exist_ok=True)
    ratings = read_ratings()
    v_cog, y_cog, g_cog, _ = load("COG-BCI_matb_graded", cap=CAP)
    v_st, y_st, g_st, t_st = load("STEW_workload_binary")
    is_hi = np.array([str(t) == "hi" for t in t_st])
    print(f"COG-BCI MATB train: {len(y_cog)} epochs, {len(set(g_cog))} subjects | "
          f"STEW: {len(y_st)} epochs, {len(set(g_st))} subjects, ratings for {len(ratings)}", flush=True)
    out = {"method": __doc__, "n_ratings": len(ratings)}

    dec = build_decoder(CFG, "balanced"); dec.fit(v_cog, y_cog, groups=g_cog)
    proba = dec.predict_proba(v_st); cls = np.array(dec.classes_)
    out["matb_index_transfer"] = evaluate(proba @ cls, is_hi, g_st, ratings, "MATB index")

    alpha = -v_st["psd"].reshape(-1, 5, 14)[:, 2, :][:, POST].mean(1)
    out["alpha_only"] = evaluate(alpha, is_hi, g_st, ratings, "alpha only")

    oof = np.full(len(y_st), np.nan)
    for tr, te in GroupKFold(8).split(v_st["psd"], y_st, g_st):
        d2 = build_decoder(CFG, "balanced")
        d2.fit({k: v[tr] for k, v in v_st.items()}, y_st[tr], groups=g_st[tr])
        p2 = d2.predict_proba({k: v[te] for k, v in v_st.items()})
        hi_col = list(d2.classes_).index(int(np.unique(y_st[is_hi])[0]))
        oof[te] = p2[:, hi_col]
    out["stew_matched"] = evaluate(oof, is_hi, g_st, ratings, "STEW matched")

    (RES / "stew_rating.json").write_text(json.dumps(out, indent=2))
    print("-> " + str(RES / "stew_rating.json"), flush=True)


if __name__ == "__main__":
    main()

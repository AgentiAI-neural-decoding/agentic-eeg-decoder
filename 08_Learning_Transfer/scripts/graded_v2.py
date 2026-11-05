#!/usr/bin/env python3
"""Plan B applied paper — GRADED cognitive-load decoding (the adaptive-learning core).

For COG-BCI n-back (0/1/2-back) and MATB (easy/med/diff) -- ordinal load levels:
  (1) 3-class decode (orchestrator + specialists), cross-subject GroupKFold, macro-F1.
  (2) CONTINUOUS load index = E[level] = sum_k k * P(level=k) from OOF probabilities.
      - does it track difficulty MONOTONICALLY? mean index per true level (expect increasing),
      - Spearman(continuous index, true level) at epoch level and subject-mean level,
      - ordinal error = MAE(round(index), true level).
This is the actionable adaptive-learning signal: a continuous load estimate that rises with demand,
decoded by a LIGHTWEIGHT classical orchestrator (no deep training).
Output -> results/graded.json
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
LEVELS = {"COG-BCI_nback_graded": ["0-back", "1-back", "2-back"],
          "COG-BCI_matb_graded": ["easy", "med", "diff"]}


def load(name, cap=9000, seed=42):
    d = np.load(FEAT / f"{name}.npz", allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]
    if cap and len(y) > cap:
        idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
        return {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx]
    return {"psd": d["X_psd"], "cov": d["X_cov"]}, y, g


def run(name, k=8):
    views, y, g = load(name)
    gkf = GroupKFold(min(k, len(np.unique(g))))
    f1_orch, f1_spec = [], {f"spec:{n}": [] for n in CFG["specialists"]}
    idx_cont = np.full(len(y), np.nan)          # OOF continuous load index E[level]
    pred_lvl = np.full(len(y), -1)
    for tr, te in gkf.split(views["psd"], y, g):
        vtr = {kk: v[tr] for kk, v in views.items()}; vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(CFG, "balanced"); dec.fit(vtr, y[tr], groups=g[tr])
        proba = dec.predict_proba(vte)                      # (n, 3) over dec.classes_
        cls = np.array(dec.classes_)
        idx_cont[te] = proba @ cls                          # E[level]
        pred_lvl[te] = cls[np.argmax(proba, 1)]
        f1_orch.append(f1_score(y[te], pred_lvl[te], average="macro"))
        for s in dec.specialists:
            sp = s.classes_[np.argmax(s.predict_proba(vte), 1)]
            f1_spec[f"spec:{s.name}"].append(f1_score(y[te], sp, average="macro"))

    # monotonicity of the continuous index across true levels
    per_level = {int(l): round(float(idx_cont[y == l].mean()), 3) for l in np.unique(y)}
    sp_ep = stats.spearmanr(idx_cont, y)
    # subject-level: mean index per (subject, true level) -> Spearman pooled
    subj = g
    rows = []
    for s in np.unique(subj):
        for l in np.unique(y):
            m = (subj == s) & (y == l)
            if m.sum():
                rows.append((idx_cont[m].mean(), l))
    si = np.array([r[0] for r in rows]); sl = np.array([r[1] for r in rows])
    sp_subj = stats.spearmanr(si, sl)
    mae = float(np.mean(np.abs(np.rint(idx_cont) - y)))
    return {
        "n_epochs": int(len(y)), "n_subjects": int(len(np.unique(g))), "n_folds": len(f1_orch),
        "macro_f1_orchestrator": round(float(np.mean(f1_orch)), 3),
        "macro_f1_specialists": {k2: round(float(np.mean(v)), 3) for k2, v in f1_spec.items()},
        "chance_macro_f1": round(1.0 / len(np.unique(y)), 3),
        "load_index_per_true_level": per_level,
        "monotonic_increasing": bool(all(np.diff(list(per_level.values())) > 0)),
        "spearman_index_vs_level_epoch": {"rho": round(float(sp_ep.correlation), 3),
                                          "p": float(sp_ep.pvalue)},
        "spearman_index_vs_level_subjectmean": {"rho": round(float(sp_subj.correlation), 3),
                                                "p": round(float(sp_subj.pvalue), 4)},
        "ordinal_mae_levels": round(mae, 3),
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in LEVELS:
        print(f"=== {name} ({'/'.join(LEVELS[name])}) ===", flush=True)
        r = run(name)
        out[name] = r
        print(f"  3-class macro-F1: orch={r['macro_f1_orchestrator']} "
              f"specs={r['macro_f1_specialists']} (chance {r['chance_macro_f1']})", flush=True)
        print(f"  continuous load index per level: {r['load_index_per_true_level']} "
              f"monotonic={r['monotonic_increasing']}", flush=True)
        print(f"  Spearman(index,level) epoch rho={r['spearman_index_vs_level_epoch']['rho']} | "
              f"subject-mean rho={r['spearman_index_vs_level_subjectmean']['rho']} "
              f"p={r['spearman_index_vs_level_subjectmean']['p']} | ordinal MAE={r['ordinal_mae_levels']}", flush=True)
    (RES / "graded.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RES / 'graded.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

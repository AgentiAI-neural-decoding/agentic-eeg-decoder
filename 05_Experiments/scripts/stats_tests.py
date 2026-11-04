#!/usr/bin/env python3
"""Uji signifikansi statistik klaim utama: orkestrator > spesialis terbaik (within-dataset).

Per-fold macro-F1 (cross-subject GroupKFold), lalu paired t-test + Wilcoxon signed-rank
(orkestrator vs spesialis terbaik). k diperbesar utk daya uji. Hasil -> results/stats_results.json.

Penggunaan: python stats_tests.py --config ../config/experiments.json
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
from agents import build_decoder  # noqa: E402
warnings.filterwarnings("ignore")

CFG_AGENTIC = {
    "specialists": {
        "spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
        "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
        "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6},
    },
    "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3},
}
TARGETS = ["STEW_workload_binary", "COG-BCI_matb_graded"]


def load(npz, cap, seed):
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]; n = len(y)
    idx = np.random.default_rng(seed).choice(n, cap, replace=False) if cap and n > cap else np.arange(n)
    return {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx]


def per_fold_f1(npz, cw, n_splits, cap, seed):
    views, y, g = load(npz, cap, seed)
    names = [s.name for s in build_decoder(CFG_AGENTIC, cw).specialists]
    rows = {f"specialist:{n}": [] for n in names}; rows["orchestrator"] = []
    gkf = GroupKFold(min(n_splits, len(np.unique(g))))
    for tr, te in gkf.split(views["psd"], y, g):
        v_tr = {k: v[tr] for k, v in views.items()}; v_te = {k: v[te] for k, v in views.items()}
        dec = build_decoder(CFG_AGENTIC, cw); dec.fit(v_tr, y[tr], groups=g[tr])
        for s in dec.specialists:
            pred = s.classes_[np.argmax(s.predict_proba(v_te), axis=1)]
            rows[f"specialist:{s.name}"].append(f1_score(y[te], pred, average="macro"))
        rows["orchestrator"].append(f1_score(y[te], dec.predict(v_te), average="macro"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n_splits", type=int, default=8)
    ap.add_argument("--cap", type=int, default=5000)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    feat = (base / cfg["features_root"]).resolve()
    out = (base / cfg["output_root"]).resolve(); out.mkdir(parents=True, exist_ok=True)
    cw = cfg.get("class_weight"); seed = cfg["cv"].get("random_state", 42)

    results = {}
    for t in TARGETS:
        npz = feat / f"{t}.npz"
        if not npz.exists():
            print(f"[skip] {t}"); continue
        print(f"\n=== {t} (k={args.n_splits}) ===", flush=True)
        rows = per_fold_f1(npz, cw, args.n_splits, args.cap, seed)
        orch = np.array(rows["orchestrator"])
        specs = {k: np.array(v) for k, v in rows.items() if k.startswith("specialist")}
        best = max(specs, key=lambda k: specs[k].mean())
        diff = orch - specs[best]
        tt = stats.ttest_rel(orch, specs[best])
        try:
            w = stats.wilcoxon(orch, specs[best])
            wp = float(w.pvalue)
        except ValueError:
            wp = float("nan")
        results[t] = {
            "n_folds": len(orch),
            "orchestrator_mean": round(float(orch.mean()), 3),
            "best_specialist": best.split(":")[1],
            "best_specialist_mean": round(float(specs[best].mean()), 3),
            "mean_diff": round(float(diff.mean()), 3),
            "paired_t_p": round(float(tt.pvalue), 4),
            "wilcoxon_p": round(wp, 4),
        }
        r = results[t]
        print(f"  orchestrator {r['orchestrator_mean']} vs {r['best_specialist']} "
              f"{r['best_specialist_mean']} | Δ={r['mean_diff']:+} | "
              f"paired-t p={r['paired_t_p']} | Wilcoxon p={r['wilcoxon_p']}", flush=True)

    (out / "stats_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nHasil -> {out / 'stats_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

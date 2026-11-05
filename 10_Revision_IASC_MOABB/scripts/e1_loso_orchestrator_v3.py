#!/usr/bin/env python3
"""E1-LOSO (IASC revision R2) — within-dataset claim under the MOABB protocol.

Claim: multi-specialist stacking decoder > best single specialist, evaluated
with **outer Leave-One-Subject-Out** (MOABB CrossSubject) and **inner 10-fold**
subject-grouped CV for the stacking meta-learner (nested CV).

Per held-out subject we record ROC-AUC (primary) and macro-F1 (secondary) for
the orchestrator and every specialist. Across subjects we compare orchestrator
vs best specialist with a paired Wilcoxon signed-rank test + SMD (Cohen's d_z).

Pilot usage (time it):
    python e1_loso_orchestrator.py --targets STEW_workload_binary --cap 200
    python e1_loso_orchestrator.py --targets STEW_workload_binary COG-BCI_matb_graded

Non-destructive: reads existing 03_Features; reuses agents.build_decoder; writes
ONLY to 10_Revision_IASC_MOABB/results/. Does not touch 09_Revision_IASC or any
original script.
"""
from __future__ import annotations
import os
# Pin BLAS/OMP to 1 thread per worker so joblib can parallelize outer LOSO folds
# across cores without oversubscription (set before numpy import).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents import build_decoder            # noqa: E402
from moabb_protocol import (                # noqa: E402
    per_subject_cap, loso_splits, score_subject, paired_stats, bh_fdr,
)
warnings.filterwarnings("ignore")

# Inner CV = 10 (nested): stacking OOF meta-features + (later) hyperparameter grids.
CFG = {
    "specialists": {
        "spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
        "riemann":  {"enabled": True, "view": "cov", "classifier": "logreg"},
        "csp":      {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6},
    },
    "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 10},
}
TARGETS = ["STEW_workload_binary", "COG-BCI_matb_graded",
           "COG-BCI_workload_binary", "COG-BCI_nback_graded"]


def load(npz, cap, seed):
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]
    idx = per_subject_cap(y, g, cap, seed)
    return ({"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], np.asarray(g)[idx])


def _cfg(inner_cv):
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CFG.items()}
    c["specialists"] = {k: dict(v) for k, v in CFG["specialists"].items()}
    c["orchestrator"] = dict(CFG["orchestrator"]); c["orchestrator"]["internal_cv_folds"] = inner_cv
    return c


def _eval_fold(v_tr, v_te, y_tr, y_te, g_tr, classes, inner_cv):
    """One outer LOSO fold (runs in a worker process). Returns per-subject
    AUC + macro-F1 dicts for orchestrator and each specialist."""
    dec = build_decoder(_cfg(inner_cv), "balanced")
    dec.fit(v_tr, y_tr, groups=g_tr)               # inner 10-fold subject-grouped OOF
    res_auc = {"orchestrator": score_subject(y_te, dec.predict_proba(v_te),
                                             dec.predict(v_te), classes)}
    o = res_auc["orchestrator"]
    auc = {"orchestrator": o["auc"]}; f1 = {"orchestrator": o["macro_f1"]}
    for sp in dec.specialists:
        p = sp.predict_proba(v_te)
        pred = sp.classes_[np.argmax(p, axis=1)]
        proba_full = np.zeros((len(y_te), len(classes)))
        for j, c in enumerate(classes):
            if c in sp.classes_:
                proba_full[:, j] = p[:, list(sp.classes_).index(c)]
        sm = score_subject(y_te, proba_full, pred, classes)
        auc[f"spec:{sp.name}"] = sm["auc"]; f1[f"spec:{sp.name}"] = sm["macro_f1"]
    return auc, f1


def evaluate_target(npz, cap, seed, n_jobs, inner_cv):
    views, y, g = load(npz, cap, seed)
    classes = np.unique(y)
    spec_names = [s.name for s in build_decoder(_cfg(inner_cv), "balanced").specialists]
    folds = list(loso_splits(g))
    tasks = [(({k: v[tr] for k, v in views.items()},
               {k: v[te] for k, v in views.items()},
               y[tr], y[te], g[tr]), (int(s) if str(s).isdigit() else str(s)))
             for tr, te, s in folds]
    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_eval_fold)(*args, classes, inner_cv) for args, _ in tasks)
    subjects = [sid for _, sid in tasks]
    subj_auc = {"orchestrator": []} | {f"spec:{n}": [] for n in spec_names}
    subj_f1  = {"orchestrator": []} | {f"spec:{n}": [] for n in spec_names}
    for auc, f1 in out:
        for k in subj_auc:
            subj_auc[k].append(auc[k]); subj_f1[k].append(f1[k])
    return spec_names, subjects, subj_auc, subj_f1, len(folds), len(y)


def summarize(metric_name, subj_scores, spec_names):
    """orchestrator vs best specialist (best = highest mean over subjects)."""
    orch = np.array(subj_scores["orchestrator"], float)
    spec_means = {n: np.nanmean(subj_scores[f"spec:{n}"]) for n in spec_names}
    best = max(spec_means, key=spec_means.get)
    ps = paired_stats(orch, np.array(subj_scores[f"spec:{best}"], float))
    return {
        "metric": metric_name,
        "orch_mean": float(np.nanmean(orch)), "orch_sd": float(np.nanstd(orch, ddof=1)),
        "spec_means": {k: float(v) for k, v in spec_means.items()},
        "best_spec": best, "best_spec_mean": float(spec_means[best]),
        "paired": ps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--cap", type=int, default=200, help="max epochs per subject (0=all)")
    ap.add_argument("--seed", type=int, default=0, help="subsample seed only (LOSO is deterministic)")
    ap.add_argument("--n-jobs", type=int, default=10, help="parallel outer LOSO folds")
    ap.add_argument("--inner-cv", type=int, default=5, help="inner subject-grouped folds for stacking OOF")
    ap.add_argument("--out", type=Path,
                    default=BASE / "10_Revision_IASC_MOABB" / "results" / "e1_loso_orchestrator.json")
    args = ap.parse_args()
    feat = BASE / "03_Features" / "features_v3"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    results = {"config": {"cap_per_subject": args.cap, "inner_cv": args.inner_cv,
                          "outer": "LOSO", "primary_metric": "roc_auc", "seed": args.seed}, "targets": {}}
    for t in args.targets:
        npz = feat / f"{t}.npz"
        if not npz.exists():
            print(f"[skip] {t}", flush=True); continue
        t0 = time.time()
        print(f"\n=== {t}  (outer=LOSO, inner={args.inner_cv}, cap/subj={args.cap}) ===", flush=True)
        spec_names, subjects, subj_auc, subj_f1, n_folds, n_epochs = evaluate_target(
            npz, args.cap, args.seed, args.n_jobs, args.inner_cv)
        auc_sum = summarize("roc_auc", subj_auc, spec_names)
        f1_sum = summarize("macro_f1", subj_f1, spec_names)
        dt = time.time() - t0
        results["targets"][t] = {
            "n_subjects": n_folds, "n_epochs_used": n_epochs, "wall_sec": round(dt, 1),
            "auc": auc_sum, "macro_f1": f1_sum,
            "per_subject": {"subjects": subjects, "auc": subj_auc, "macro_f1": subj_f1},
        }
        a, f = auc_sum, f1_sum
        print(f"  N={n_folds} subj, {n_epochs} epochs, {dt:.1f}s", flush=True)
        print(f"  AUC : orch {a['orch_mean']:.3f}±{a['orch_sd']:.3f} vs {a['best_spec']} "
              f"{a['best_spec_mean']:.3f} | Δ={a['paired']['mean_diff']:+.3f} "
              f"| dz={a['paired']['smd_dz']:.2f} | pos {a['paired']['n_pos']}/{a['paired']['n_subjects']} "
              f"| Wilcoxon(greater) p={a['paired']['wilcoxon_p_greater']:.4f}", flush=True)
        print(f"  F1  : orch {f['orch_mean']:.3f}±{f['orch_sd']:.3f} vs {f['best_spec']} "
              f"{f['best_spec_mean']:.3f} | Δ={f['paired']['mean_diff']:+.3f} "
              f"| Wilcoxon(greater) p={f['paired']['wilcoxon_p_greater']:.4f}", flush=True)
        args.out.write_text(json.dumps(results, indent=2))  # checkpoint per target

    # family-wide FDR across targets (AUC, one-sided greater) — illustrative for the pilot
    tnames = list(results["targets"].keys())
    if tnames:
        raw = [results["targets"][t]["auc"]["paired"]["wilcoxon_p_greater"] for t in tnames]
        q = bh_fdr(raw)
        for t, qq in zip(tnames, q):
            results["targets"][t]["auc"]["paired"]["fdr_q_across_targets"] = float(qq)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nDone -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

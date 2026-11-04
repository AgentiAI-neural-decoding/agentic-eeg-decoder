#!/usr/bin/env python3
"""Evaluasi terpadu (SATU sumber kebenaran) — menghapus inkonsistensi angka antar-tabel.

Satu run, satu cap, satu seed, satu k: menghasilkan PER-FOLD macro-F1 untuk semua kondisi
yang dipakai di Tabel 1 (orkestrator vs spesialis), Tabel 2 (ablation), dan uji signifikansi
(orkestrator vs spesialis terbaik). Karena semuanya dari fold yang sama, angka teks/tabel/abstrak
dijamin konsisten.

Uji signifikansi memakai **corrected resampled t-test (Nadeau & Bengio, 2003)** yang mengoreksi
pelanggaran independensi antar-fold CV (fold berbagi data latih), plus Wilcoxon (bebas-asumsi).
Tanpa koreksi multiple-comparison (dilaporkan apa adanya di naskah).

Output -> results/final_eval.json.
Penggunaan: python final_eval.py --config ../config/experiments.json --n_splits 8 --cap 8000
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

SPEC = {
    "spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
    "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
    "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6},
}
ORCH = {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3}


def cfg_with(specs):
    return {"specialists": {k: dict(SPEC[k]) for k in specs}, "orchestrator": dict(ORCH)}


# target -> daftar kondisi yang dibutuhkan
T1 = ["COG-BCI_nback_graded", "COG-BCI_matb_graded", "COG-BCI_workload_binary", "STEW_workload_binary"]
ABLATION = ["COG-BCI_matb_graded", "STEW_workload_binary"]   # juga butuh softvote + leave-one-out
STATS = ["STEW_workload_binary", "COG-BCI_matb_graded"]      # uji signifikansi orch vs best-spec


def load(npz, cap, seed):
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]; n = len(y)
    idx = np.random.default_rng(seed).choice(n, cap, replace=False) if cap and n > cap else np.arange(n)
    return {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx]


def softvote_f1(dec, v_te, y_te):
    probas = [s.predict_proba(v_te) for s in dec.specialists]
    pred = dec.classes_[np.argmax(np.mean(probas, axis=0), axis=1)]
    return f1_score(y_te, pred, average="macro")


def per_fold(npz, cw, k, cap, seed, want_ablation):
    views, y, g = load(npz, cap, seed)
    rows = {f"spec:{n}": [] for n in SPEC}
    rows["full_stacking"] = []
    if want_ablation:
        rows["full_softvote"] = []
        for miss in SPEC:
            rows[f"minus_{miss}"] = []
    gkf = GroupKFold(min(k, len(np.unique(g))))
    for tr, te in gkf.split(views["psd"], y, g):
        v_tr = {kk: v[tr] for kk, v in views.items()}
        v_te = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(cfg_with(list(SPEC)), cw); dec.fit(v_tr, y[tr], groups=g[tr])
        for s in dec.specialists:
            pred = s.classes_[np.argmax(s.predict_proba(v_te), axis=1)]
            rows[f"spec:{s.name}"].append(f1_score(y[te], pred, average="macro"))
        rows["full_stacking"].append(f1_score(y[te], dec.predict(v_te), average="macro"))
        if want_ablation:
            rows["full_softvote"].append(softvote_f1(dec, v_te, y[te]))
            for miss in SPEC:
                keep = [s for s in SPEC if s != miss]
                d2 = build_decoder(cfg_with(keep), cw); d2.fit(v_tr, y[tr], groups=g[tr])
                rows[f"minus_{miss}"].append(f1_score(y[te], d2.predict(v_te), average="macro"))
    return {k2: np.array(v) for k2, v in rows.items()}, int(len(np.unique(g)))


def nb_ttest(diff, k):
    """Corrected resampled t-test (Nadeau & Bengio 2003) untuk k-fold CV.
    Faktor varian terkoreksi = 1/k + rho/(1-rho), rho = frac uji = 1/k => 1/k + 1/(k-1)."""
    k = len(diff); m = float(diff.mean()); var = float(diff.var(ddof=1))
    if var == 0:
        return float("inf"), 0.0
    factor = 1.0 / k + (1.0 / (k - 1))
    t = m / np.sqrt(var * factor)
    p = 2.0 * stats.t.sf(abs(t), k - 1)
    return round(float(t), 3), round(float(p), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n_splits", type=int, default=8)
    ap.add_argument("--cap", type=int, default=8000)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    feat = (base / cfg["features_root"]).resolve()
    out = (base / cfg["output_root"]).resolve(); out.mkdir(parents=True, exist_ok=True)
    cw = cfg.get("class_weight"); seed = cfg["cv"].get("random_state", 42)
    k = args.n_splits

    res = {"_meta": {"n_splits": k, "cap": args.cap, "seed": seed,
                     "significance": "Nadeau-Bengio corrected resampled t-test + Wilcoxon; no multiple-comparison correction"}}
    for t in T1:
        npz = feat / f"{t}.npz"
        if not npz.exists():
            print(f"[skip] {t}"); continue
        want_abl = t in ABLATION
        print(f"\n=== {t} (k={k}, cap={args.cap}, ablation={want_abl}) ===", flush=True)
        rows, nsubj = per_fold(npz, cw, k, args.cap, seed, want_abl)
        means = {kk: round(float(v.mean()), 3) for kk, v in rows.items()}
        stds = {kk: round(float(v.std()), 3) for kk, v in rows.items()}
        entry = {"n_subjects": nsubj, "n_folds": len(rows["full_stacking"]),
                 "means": means, "stds": stds}
        if t in STATS:
            specs = {kk: v for kk, v in rows.items() if kk.startswith("spec:")}
            best = max(specs, key=lambda kk: specs[kk].mean())
            diff = rows["full_stacking"] - specs[best]
            nb_t, nb_p = nb_ttest(diff, k)
            t_uncorr = round(float(stats.ttest_rel(rows["full_stacking"], specs[best]).pvalue), 4)
            try:
                wp = round(float(stats.wilcoxon(rows["full_stacking"], specs[best]).pvalue), 4)
            except ValueError:
                wp = float("nan")
            entry["sig_orch_vs_best"] = {
                "best_specialist": best.split(":")[1],
                "orchestrator_mean": means["full_stacking"],
                "best_specialist_mean": means[best],
                "mean_diff": round(float(diff.mean()), 3),
                "paired_t_p_uncorrected": t_uncorr,
                "nadeau_bengio_p": nb_p,
                "wilcoxon_p": wp,
            }
            s = entry["sig_orch_vs_best"]
            print(f"  orch {s['orchestrator_mean']} vs {s['best_specialist']} {s['best_specialist_mean']} "
                  f"Δ={s['mean_diff']:+} | t-p(uncorr)={s['paired_t_p_uncorrected']} "
                  f"NB-p={s['nadeau_bengio_p']} W-p={s['wilcoxon_p']}", flush=True)
        else:
            print(f"  stacking={means['full_stacking']} specs=" +
                  " ".join(f"{kk.split(':')[1]}={means[kk]}" for kk in means if kk.startswith("spec:")), flush=True)
        res[t] = entry
    (out / "final_eval.json").write_text(json.dumps(res, indent=2))
    print(f"\nHasil -> {out / 'final_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

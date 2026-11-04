#!/usr/bin/env python3
"""Tahap 05 — Ablation study arsitektur agentic.

Mengukur kontribusi tiap komponen pada decoder (Tahap 04):
  - tiap spesialis sendiri (spectral / riemann / csp),
  - leave-one-specialist-out (full minus satu),
  - full ensemble dengan orkestrator `stacking` vs `soft_vote`.

Cross-subject GroupKFold, F1-macro sebagai metrik utama (robust terhadap imbalans).

Penggunaan:
    python ablation.py --config ../config/experiments.json --target COG-BCI_matb_graded
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
from agents import build_decoder  # noqa: E402

warnings.filterwarnings("ignore")

AGENTIC_CFG = {
    "specialists": {
        "spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
        "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
        "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6},
    },
    "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3},
}

DEFAULT_TARGETS = ["COG-BCI_matb_graded", "STEW_workload_binary"]


def make_cfg(enabled: list[str], orch_type: str = "stacking") -> dict:
    c = copy.deepcopy(AGENTIC_CFG)
    for name in c["specialists"]:
        c["specialists"][name]["enabled"] = name in enabled
    c["orchestrator"]["type"] = orch_type
    return c


def load(npz, cap, seed):
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]; n = len(y)
    idx = np.random.default_rng(seed).choice(n, cap, replace=False) if cap and n > cap else np.arange(n)
    return {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx]


def score(cfg_variant, views, y, groups, n_splits, cw):
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s = [], []
    for tr, te in gkf.split(views["psd"], y, groups):
        v_tr = {k: v[tr] for k, v in views.items()}
        v_te = {k: v[te] for k, v in views.items()}
        dec = build_decoder(cfg_variant, cw)
        dec.fit(v_tr, y[tr], groups=groups[tr])
        p = dec.predict(v_te)
        accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
    return {"acc": round(float(np.mean(accs)), 3), "f1": round(float(np.mean(f1s)), 3)}


def ablate(npz, cfg, log):
    cw = cfg.get("class_weight"); cvc = cfg["cv"]
    views, y, g = load(npz, cvc.get("subsample_per_run"), cvc.get("random_state", 42))
    n_splits = min(cvc["n_splits"], len(np.unique(g)))
    variants = {
        "only_spectral": make_cfg(["spectral"]),
        "only_riemann": make_cfg(["riemann"]),
        "only_csp": make_cfg(["csp"]),
        "minus_spectral": make_cfg(["riemann", "csp"]),
        "minus_riemann": make_cfg(["spectral", "csp"]),
        "minus_csp": make_cfg(["spectral", "riemann"]),
        "full_softvote": make_cfg(["spectral", "riemann", "csp"], "soft_vote"),
        "full_stacking": make_cfg(["spectral", "riemann", "csp"], "stacking"),
    }
    out = {"n_subjects": int(len(np.unique(g))), "n_splits": n_splits, "variants": {}}
    for name, cv in variants.items():
        out["variants"][name] = score(cv, views, y, g, n_splits, cw)
        log(f"    {name:<16} acc={out['variants'][name]['acc']} F1={out['variants'][name]['f1']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--targets", nargs="*")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    feat_root = (base / cfg["features_root"]).resolve()
    out_root = (base / cfg["output_root"]).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    results = {}
    for t in (args.targets or DEFAULT_TARGETS):
        npz = feat_root / f"{t}.npz"
        if not npz.exists():
            log(f"[skip] {t}"); continue
        log(f"\n=== Ablation: {t} ===")
        results[t] = ablate(npz, cfg, log)
    (out_root / "ablation_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nSelesai. Hasil -> {out_root / 'ablation_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

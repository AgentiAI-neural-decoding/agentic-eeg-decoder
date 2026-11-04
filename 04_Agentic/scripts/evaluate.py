#!/usr/bin/env python3
"""Evaluasi arsitektur Agentic — spesialis vs orkestrator (cross-subject, anti-leakage).

Untuk tiap target decoding (file fitur Tahap 03), jalankan GroupKFold per-subjek:
di tiap fold latih AgenticDecoder (yang melatih semua spesialis + orkestrator atas
out-of-fold proba), lalu bandingkan akurasi/F1 tiap spesialis sendiri vs orkestrator.
Membuktikan nilai koordinasi agentic (orkestrator >= spesialis terbaik).

Penggunaan:
    python evaluate.py --config ../config/agentic.json
    python evaluate.py --config ../config/agentic.json --targets COG-BCI_matb_graded STEW_workload_binary
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents import build_decoder

warnings.filterwarnings("ignore")

DEFAULT_TARGETS = [
    "COG-BCI_nback_graded", "COG-BCI_matb_graded",
    "COG-BCI_workload_binary", "STEW_workload_binary",
]


def load_views(npz_path: Path, cap: int, seed: int):
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"].astype(int)
    groups = d["groups"]
    n = len(y)
    if cap and n > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, cap, replace=False)
    else:
        idx = np.arange(n)
    views = {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}
    return views, y[idx], groups[idx]


def eval_target(npz_path: Path, cfg: dict, log) -> dict:
    cw = cfg.get("class_weight")
    cvc = cfg["cv"]
    views, y, groups = load_views(npz_path, cvc.get("subsample_per_run"), cvc.get("random_state", 42))
    n_splits = min(cvc["n_splits"], len(np.unique(groups)))
    chance = np.bincount(y).max() / len(y)

    names = [s.name for s in build_decoder(cfg, cw).specialists]
    rows = {f"specialist:{n}": {"acc": [], "f1": []} for n in names}
    rows["orchestrator"] = {"acc": [], "f1": []}

    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(views["psd"], y, groups)):
        v_tr = {kk: vv[tr] for kk, vv in views.items()}
        v_te = {kk: vv[te] for kk, vv in views.items()}
        dec = build_decoder(cfg, cw)
        dec.fit(v_tr, y[tr], groups=groups[tr])
        # spesialis individual (sudah ter-refit di dalam decoder)
        for s in dec.specialists:
            pred = s.classes_[np.argmax(s.predict_proba(v_te), axis=1)]
            rows[f"specialist:{s.name}"]["acc"].append(accuracy_score(y[te], pred))
            rows[f"specialist:{s.name}"]["f1"].append(f1_score(y[te], pred, average="macro"))
        # orkestrator
        op = dec.predict(v_te)
        rows["orchestrator"]["acc"].append(accuracy_score(y[te], op))
        rows["orchestrator"]["f1"].append(f1_score(y[te], op, average="macro"))
        log(f"    fold {k+1}/{n_splits} selesai")

    out = {"n_epochs": int(len(y)), "n_subjects": int(len(np.unique(groups))),
           "n_classes": int(len(np.unique(y))), "chance": round(float(chance), 3),
           "n_splits": n_splits, "scores": {}}
    for name, sc in rows.items():
        out["scores"][name] = {
            "acc_mean": round(float(np.mean(sc["acc"])), 3),
            "acc_std": round(float(np.std(sc["acc"])), 3),
            "f1_mean": round(float(np.mean(sc["f1"])), 3),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--targets", nargs="*")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    feat_root = (base / cfg["features_root"]).resolve()
    out_root = (base / cfg["output_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    targets = args.targets or DEFAULT_TARGETS
    results = {}
    for t in targets:
        npz = feat_root / f"{t}.npz"
        if not npz.exists():
            log(f"[skip] {t}: {npz} tidak ada")
            continue
        log(f"\n=== {t} ===")
        results[t] = eval_target(npz, cfg, log)
        sc = results[t]["scores"]
        best_spec = max((k for k in sc if k.startswith("specialist")), key=lambda k: sc[k]["acc_mean"])
        orch = sc["orchestrator"]["acc_mean"]
        log(f"  chance={results[t]['chance']} | spesialis terbaik={best_spec.split(':')[1]} "
            f"({sc[best_spec]['acc_mean']}) | ORKESTRATOR={orch} "
            f"({'+' if orch>=sc[best_spec]['acc_mean'] else ''}{round(orch-sc[best_spec]['acc_mean'],3)})")

    (out_root / "agentic_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nSelesai. Hasil -> {out_root / 'agentic_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

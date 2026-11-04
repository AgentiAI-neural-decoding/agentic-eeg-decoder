#!/usr/bin/env python3
"""Uji transfer lintas-dataset — hipotesis inti penelitian.

Membandingkan decoding beban kognitif (workload biner) pada dataset TARGET ketika
model dilatih:
  - SINGLE  : hanya subjek-latih TARGET (cross-subject).
  - MULTI   : subjek-latih TARGET + SELURUH dataset SUMBER (transfer/augmentasi).

Diuji pada subjek TARGET yang di-hold-out (GroupKFold), sehingga peningkatan MULTI
atas SINGLE = bukti manfaat integrasi multi-dataset (target proposal: +40% vs single).

Penggunaan:
    python transfer.py --config ../config/agentic.json --target STEW --source COG-BCI
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


def load(npz_path: Path, cap, seed):
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]
    n = len(y)
    idx = np.random.default_rng(seed).choice(n, cap, replace=False) if cap and n > cap else np.arange(n)
    return {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx]


def concat_views(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def run(target, source, cfg, feat_root, log):
    cw = cfg.get("class_weight"); cvc = cfg["cv"]; seed = cvc.get("random_state", 42)
    cap = cvc.get("subsample_per_run")
    vt, yt, gt = load(feat_root / f"{target}_workload_binary.npz", cap, seed)
    vs, ys, gs = load(feat_root / f"{source}_workload_binary.npz", cap, seed)
    n_splits = min(cvc["n_splits"], len(np.unique(gt)))
    gkf = GroupKFold(n_splits=n_splits)
    res = {"single": {"acc": [], "f1": []}, "multi": {"acc": [], "f1": []}}

    for k, (tr, te) in enumerate(gkf.split(vt["psd"], yt, gt)):
        v_tr = {kk: vv[tr] for kk, vv in vt.items()}
        v_te = {kk: vv[te] for kk, vv in vt.items()}
        # SINGLE
        d1 = build_decoder(cfg, cw); d1.fit(v_tr, yt[tr], groups=gt[tr])
        p1 = d1.predict(v_te)
        res["single"]["acc"].append(accuracy_score(yt[te], p1))
        res["single"]["f1"].append(f1_score(yt[te], p1, average="macro"))
        # MULTI (+ seluruh sumber)
        v_multi = concat_views(v_tr, vs)
        y_multi = np.concatenate([yt[tr], ys])
        g_multi = np.concatenate([gt[tr], gs])
        d2 = build_decoder(cfg, cw); d2.fit(v_multi, y_multi, groups=g_multi)
        p2 = d2.predict(v_te)
        res["multi"]["acc"].append(accuracy_score(yt[te], p2))
        res["multi"]["f1"].append(f1_score(yt[te], p2, average="macro"))
        log(f"    fold {k+1}/{n_splits}: single={res['single']['acc'][-1]:.3f} multi={res['multi']['acc'][-1]:.3f}")

    s_acc = float(np.mean(res["single"]["acc"])); m_acc = float(np.mean(res["multi"]["acc"]))
    s_f1 = float(np.mean(res["single"]["f1"])); m_f1 = float(np.mean(res["multi"]["f1"]))
    rel = (m_acc - s_acc) / s_acc * 100 if s_acc else 0.0
    return {"target": target, "source": source, "n_splits": n_splits,
            "single_acc": round(s_acc, 3), "multi_acc": round(m_acc, 3),
            "single_f1": round(s_f1, 3), "multi_f1": round(m_f1, 3),
            "abs_gain_acc": round(m_acc - s_acc, 3), "rel_gain_acc_pct": round(rel, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target", default="STEW")
    ap.add_argument("--source", default="COG-BCI")
    ap.add_argument("--both", action="store_true", help="Uji kedua arah (target<->source)")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    feat_root = (base / cfg["features_root"]).resolve()
    out_root = (base / cfg["output_root"]).resolve()
    log = lambda m: print(m, flush=True)  # noqa: E731

    pairs = [(args.target, args.source)]
    if args.both:
        pairs.append((args.source, args.target))
    results = []
    for tgt, src in pairs:
        log(f"\n=== Transfer: target={tgt} + sumber={src} ===")
        r = run(tgt, src, cfg, feat_root, log)
        results.append(r)
        log(f"  SINGLE={r['single_acc']} | MULTI={r['multi_acc']} | "
            f"gain={r['abs_gain_acc']:+} ({r['rel_gain_acc_pct']:+}%)")

    (out_root / "transfer_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nSelesai. Hasil -> {out_root / 'transfer_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

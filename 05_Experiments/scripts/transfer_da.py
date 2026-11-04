#!/usr/bin/env python3
"""Tahap 05 — Transfer lintas-dataset dengan Domain Adaptation Riemannian.

Tahap 04 menemukan: pooling NAIF lintas-dataset = negative transfer (domain shift).
Skrip ini menguji apakah **re-centering Riemannian per-domain** (TLCenter; inti
Riemannian Procrustes Analysis, Zanini 2018 / Rodrigues 2019) memperbaikinya.

Tiga kondisi diuji pada subjek TARGET yang di-hold-out (GroupKFold cross-subject),
semua memakai decoder Riemannian (TangentSpace + LogReg) pada view kovarians:

  1. SINGLE      : latih hanya subjek-latih TARGET.
  2. MULTI-naif  : latih TARGET-train + SUMBER (pooling mentah, tanpa DA).
  3. MULTI-DA    : idem + TLCenter me-recenter tiap domain ke identitas sebelum decode.

Catatan transduktif: TLCenter me-recenter domain TARGET memakai rerata geometrik
kovarians target (termasuk test yang TAK berlabel) — asumsi standar transfer BCI
(data target unlabeled tersedia). Tidak ada kebocoran LABEL (TLCenter tak pakai y;
classifier hanya dilatih pada label train).

Penggunaan:
    python transfer_da.py --config ../config/experiments.json --both
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import TLCenter, encode_domains
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_cov(npz_path: Path, cap, seed):
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]
    n = len(y)
    idx = np.random.default_rng(seed).choice(n, cap, replace=False) if cap and n > cap else np.arange(n)
    return d["X_cov"][idx], y[idx], g[idx]


def riemann_clf(cw):
    return make_pipeline(TangentSpace(metric="riemann"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight=cw))


def recenter(Xs, Xt_tr, Xt_te, ys, yt_tr):
    """TLCenter: recenter source & target (train+test) ke identitas per-domain."""
    X = np.concatenate([Xs, Xt_tr, Xt_te])
    dummy_te = np.zeros(len(Xt_te), dtype=int)
    y = np.concatenate([ys, yt_tr, dummy_te])
    dom = np.array(["source"] * len(Xs) + ["target"] * (len(Xt_tr) + len(Xt_te)))
    Xenc, yenc = encode_domains(X, y, dom)
    Xr = TLCenter(target_domain="target").fit_transform(Xenc, yenc)
    a, b = len(Xs), len(Xs) + len(Xt_tr)
    return Xr[:a], Xr[a:b], Xr[b:]


def run(target, source, cfg, feat_root, log):
    cw = cfg.get("class_weight"); cvc = cfg["cv"]; seed = cvc.get("random_state", 42)
    cap = cvc.get("subsample_per_run")
    Xt, yt, gt = load_cov(feat_root / f"{target}_workload_binary.npz", cap, seed)
    Xs, ys, gs = load_cov(feat_root / f"{source}_workload_binary.npz", cap, seed)
    n_splits = min(cvc["n_splits"], len(np.unique(gt)))
    gkf = GroupKFold(n_splits=n_splits)
    R = {k: {"acc": [], "f1": []} for k in ("single", "multi_naive", "multi_da")}

    for k, (tr, te) in enumerate(gkf.split(Xt, yt, gt)):
        Xt_tr, yt_tr = Xt[tr], yt[tr]
        Xt_te, yt_te = Xt[te], yt[te]

        # 1) SINGLE
        m = riemann_clf(cw).fit(Xt_tr, yt_tr); p = m.predict(Xt_te)
        R["single"]["acc"].append(accuracy_score(yt_te, p)); R["single"]["f1"].append(f1_score(yt_te, p, average="macro"))

        # 2) MULTI-naif
        Xm = np.concatenate([Xs, Xt_tr]); ym = np.concatenate([ys, yt_tr])
        m = riemann_clf(cw).fit(Xm, ym); p = m.predict(Xt_te)
        R["multi_naive"]["acc"].append(accuracy_score(yt_te, p)); R["multi_naive"]["f1"].append(f1_score(yt_te, p, average="macro"))

        # 3) MULTI-DA (TLCenter recenter)
        Xs_r, Xt_tr_r, Xt_te_r = recenter(Xs, Xt_tr, Xt_te, ys, yt_tr)
        Xm_r = np.concatenate([Xs_r, Xt_tr_r]); ym = np.concatenate([ys, yt_tr])
        m = riemann_clf(cw).fit(Xm_r, ym); p = m.predict(Xt_te_r)
        R["multi_da"]["acc"].append(accuracy_score(yt_te, p)); R["multi_da"]["f1"].append(f1_score(yt_te, p, average="macro"))
        log(f"    fold {k+1}/{n_splits}: single={R['single']['acc'][-1]:.3f} naive={R['multi_naive']['acc'][-1]:.3f} DA={R['multi_da']['acc'][-1]:.3f}")

    def mean(c, m): return round(float(np.mean(R[c][m])), 3)
    out = {"target": target, "source": source, "n_splits": n_splits}
    for c in ("single", "multi_naive", "multi_da"):
        out[c] = {"acc": mean(c, "acc"), "f1": mean(c, "f1")}
    out["da_vs_naive_acc"] = round(out["multi_da"]["acc"] - out["multi_naive"]["acc"], 3)
    out["da_vs_single_acc"] = round(out["multi_da"]["acc"] - out["single"]["acc"], 3)
    out["da_vs_single_rel_pct"] = round((out["multi_da"]["acc"] - out["single"]["acc"]) / out["single"]["acc"] * 100, 1) if out["single"]["acc"] else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target", default="STEW")
    ap.add_argument("--source", default="COG-BCI")
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    feat_root = (base / cfg["features_root"]).resolve()
    out_root = (base / cfg["output_root"]).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    pairs = [(args.target, args.source)]
    if args.both:
        pairs.append((args.source, args.target))
    results = []
    for tgt, src in pairs:
        log(f"\n=== Transfer+DA: target={tgt} + sumber={src} ===")
        r = run(tgt, src, cfg, feat_root, log)
        results.append(r)
        log(f"  SINGLE={r['single']['acc']} | NAIF={r['multi_naive']['acc']} | DA={r['multi_da']['acc']}")
        log(f"  DA vs naif: {r['da_vs_naive_acc']:+} | DA vs single: {r['da_vs_single_acc']:+} ({r['da_vs_single_rel_pct']:+}%)")

    (out_root / "transfer_da_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nSelesai. Hasil -> {out_root / 'transfer_da_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

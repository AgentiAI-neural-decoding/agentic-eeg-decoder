#!/usr/bin/env python3
"""Tahap 06 — frozen-encoder probing + class-conditional DA untuk transfer.

Protokol AMAN per benchmark 2026 (arXiv:2604.23091): bekukan encoder, latih hanya
probe linear di atas embedding (hindari SFT naif yang negative-transfer 24.8%).

Tiga kondisi pada subjek TARGET hold-out (GroupKFold cross-subject):
  - SINGLE      : probe pada embedding target-train.
  - MULTI       : probe pada embedding target-train + SUMBER (pooling embedding).
  - MULTI+ccDA  : sumber diselaraskan PER-KELAS ke target-train (class-conditional
                  mean alignment di ruang embedding — wawasan cdaDA), lalu probe.

Encoder default = placeholder (validasi harness). Swap ke CBraMod via config.

Penggunaan:
    python probe_transfer.py --config ../config/fm.json --target STEW --source COG-BCI --both
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm_encoder import build_encoder, load_harmonized_epochs

warnings.filterwarnings("ignore")


def class_conditional_align(Xs, ys, Xt_tr, yt_tr):
    """Geser embedding sumber per-kelas agar mean kelas-c sumber = mean kelas-c target-train."""
    Xs2 = Xs.copy()
    for c in np.unique(yt_tr):
        if (ys == c).sum() == 0:
            continue
        shift = Xt_tr[yt_tr == c].mean(0) - Xs[ys == c].mean(0)
        Xs2[ys == c] += shift
    return Xs2


def _sym_pow(C, p):
    """Pangkat matriks simetris PSD via eigendekomposisi."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    return (V * (w ** p)) @ V.T


def coral_align(Xs, ys, Xt_tr, yt_tr, reg=1e-2):
    """CORAL per-kelas: selaraskan mean + KOVARIANS tiap kelas sumber ke target-train
    (second-order, lebih kuat dari align-mean). Fallback ke mean bila sampel < dim."""
    Xa = Xs.astype(float).copy()
    d = Xs.shape[1]
    for c in np.unique(yt_tr):
        si, ti = ys == c, yt_tr == c
        if si.sum() == 0 or ti.sum() == 0:
            continue
        ms, mt = Xs[si].mean(0), Xt_tr[ti].mean(0)
        if si.sum() <= d or ti.sum() <= d:
            Xa[si] += mt - ms
            continue
        cs = np.cov(Xs[si], rowvar=False) + reg * np.eye(d)
        ct = np.cov(Xt_tr[ti], rowvar=False) + reg * np.eye(d)
        Xa[si] = (Xs[si] - ms) @ (_sym_pow(cs, -0.5) @ _sym_pow(ct, 0.5)) + mt
    return Xa


def probe(Xtr, ytr, Xte, cw):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight=cw).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def run(target, source, emb, cfg, log):
    Et, yt, gt = emb[target]
    Es, ys, gs = emb[source]
    cw = cfg["probe"].get("class_weight")
    n_splits = min(cfg["cv"]["n_splits"], len(np.unique(gt)))
    gkf = GroupKFold(n_splits=n_splits)
    R = {c: {"acc": [], "f1": []} for c in ("single", "multi", "multi_ccda", "multi_coral")}

    for tr, te in gkf.split(Et, yt, gt):
        Et_tr, Et_te = Et[tr], Et[te]
        def ev(Xtr, ytr, cond):
            p = probe(Xtr, ytr, Et_te, cw)
            R[cond]["acc"].append(accuracy_score(yt[te], p))
            R[cond]["f1"].append(f1_score(yt[te], p, average="macro"))
        ym = np.concatenate([yt[tr], ys])
        ev(Et_tr, yt[tr], "single")
        ev(np.vstack([Et_tr, Es]), ym, "multi")
        ev(np.vstack([Et_tr, class_conditional_align(Es, ys, Et_tr, yt[tr])]), ym, "multi_ccda")
        ev(np.vstack([Et_tr, coral_align(Es, ys, Et_tr, yt[tr])]), ym, "multi_coral")

    def m(c, k): return round(float(np.mean(R[c][k])), 3)
    out = {"target": target, "source": source, "n_splits": n_splits}
    for c in ("single", "multi", "multi_ccda", "multi_coral"):
        out[c] = {"acc": m(c, "acc"), "f1": m(c, "f1")}
    b = out["single"]["f1"]
    out["gain_f1_multi"] = round(out["multi"]["f1"] - b, 3)
    out["gain_f1_ccda"] = round(out["multi_ccda"]["f1"] - b, 3)
    out["gain_f1_coral"] = round(out["multi_coral"]["f1"] - b, 3)
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
    out_root = (base / cfg["output_root"]).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    common = cfg["common_channels"]

    enc_opts = cfg["encoder_options"].get(cfg["encoder"], {})
    enc = build_encoder(cfg["encoder"], enc_opts, len(common), cfg.get("device", "cpu"))
    # Encoder boleh menuntut sampling-rate tertentu (CBraMod=200 Hz); jika tidak, pakai default config.
    sfreq = enc_opts.get("sfreq", cfg["target_sfreq_hz"])
    scheme = cfg.get("label_scheme", "binary")
    log(f"Encoder: {cfg['encoder']} (device={cfg.get('device')}, sfreq={sfreq} Hz, label={scheme})")

    # Ekstrak embedding sekali per dataset
    emb = {}
    for ds in {args.target, args.source}:
        deriv = (base / cfg["derivatives"][ds]).resolve()
        X, y, g = load_harmonized_epochs(deriv, ds, common, sfreq,
                                         cfg["cv"]["max_epochs_per_dataset"], cfg["cv"]["random_state"],
                                         scheme=scheme)
        log(f"  {ds}: {X.shape[0]} epoch ter-harmonisasi {X.shape[1:]} -> embedding...")
        emb[ds] = (enc.embed(X), y, g)
        log(f"  {ds}: embedding {emb[ds][0].shape}")

    pairs = [(args.target, args.source)]
    if args.both:
        pairs.append((args.source, args.target))
    results = []
    for tgt, src in pairs:
        log(f"\n=== Probing transfer: target={tgt} + sumber={src} ===")
        r = run(tgt, src, emb, cfg, log)
        results.append(r)
        log(f"  SINGLE F1={r['single']['f1']} | MULTI {r['multi']['f1']} ({r['gain_f1_multi']:+}) "
            f"| ccDA-mean {r['multi_ccda']['f1']} ({r['gain_f1_ccda']:+}) "
            f"| ccDA-CORAL {r['multi_coral']['f1']} ({r['gain_f1_coral']:+})")

    (out_root / "fm_probe_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nSelesai. Hasil -> {out_root / 'fm_probe_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

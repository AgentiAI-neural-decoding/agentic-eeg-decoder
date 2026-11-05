"""
08_ds003838_transfer.py — masukkan ds003838 (digit-span WM, 65 subj, research-grade)
ke analisis transfer Pool B, menguji ulang hipotesis "perangkat, bukan volume data".

ds003838 = high-density research-grade (drop-in 14-ch), konstruk beban = working-memory
digit-span (rest vs memorize-13). Bersama COG-BCI (MATB) & Hinss (MATB) membentuk
Pool B research-grade; STEW = Emotiv consumer (cross-device).

Menghitung, per skema (workload_extreme primer, workload_binary pembanding):
  1. Langit-langit within-dataset (cross-subject GroupKFold-5) tiap dataset.
  2. Transfer terarah PSD: device-matched (ds003838<->COG-BCI, ds003838<->Hinss)
     vs cross-device (ds003838<->STEW), + baseline COG<->STEW.
  3. Transfer tangent-space Riemannian utk pasangan kunci (pembanding fitur).
  4. Pool B research-grade: latih (COG-BCI+Hinss) -> ds003838, dan -> STEW.
  5. span_graded within-dataset (ds003838): decoding beban WM 3-tingkat (5/9/13).

Output: new_analysis/data/ds003838_transfer_results.json + ringkasan cetak.
"""
import json
import numpy as np
from transfer_lib import (load, within_dataset_f1, transfer_f1, clf_pipeline,
                          DATA_OUT, SEED)
from sklearn.metrics import f1_score

SCHEMES = ["workload_extreme", "workload_binary"]

# Pasangan terarah (src -> tgt, relasi perangkat)
PAIRS = [
    ("ds003838", "COG-BCI", "device-matched"),
    ("COG-BCI", "ds003838", "device-matched"),
    ("ds003838", "Hinss", "device-matched"),
    ("Hinss", "ds003838", "device-matched"),
    ("ds003838", "STEW", "cross-device"),
    ("STEW", "ds003838", "cross-device"),
    ("COG-BCI", "STEW", "cross-device"),   # baseline referensi
]
CEIL_DS = ["COG-BCI", "Hinss", "STEW", "ds003838"]


def tangent_transfer(src, tgt, scheme):
    """Transfer terarah pakai fitur tangent-space Riemannian (X_tangent 105-dim)."""
    _, _, Xs_t, ys, _ = load(src, scheme)
    _, _, Xt_t, yt, _ = load(tgt, scheme)
    clf = clf_pipeline().fit(Xs_t, ys)
    return float(f1_score(yt, clf.predict(Xt_t), average="macro"))


def pool_transfer(sources, tgt, scheme):
    """Latih PSD pada gabungan beberapa sumber -> uji target. Return macro-F1."""
    Xs_list, ys_list = [], []
    for s in sources:
        Xp, _, _, y, _ = load(s, scheme)
        Xs_list.append(Xp)
        ys_list.append(y)
    Xs = np.concatenate(Xs_list)
    ys = np.concatenate(ys_list)
    Xt, _, _, yt, _ = load(tgt, scheme)
    clf = clf_pipeline().fit(Xs, ys)
    return float(f1_score(yt, clf.predict(Xt), average="macro"))


def main():
    out = {}
    for scheme in SCHEMES:
        print(f"\n===== {scheme} =====", flush=True)
        # 1) langit-langit within-dataset
        ceil = {}
        for ds in CEIL_DS:
            Xp, _, _, y, g = load(ds, scheme)
            f1, sd = within_dataset_f1(Xp, y, g, n_splits=5)
            ceil[ds] = {"within_f1": round(f1, 4), "within_f1_std": round(sd, 4),
                        "n_epoch": int(len(y)), "n_subj": int(len(np.unique(g)))}
            print(f"  ceiling {ds:10s}: F1 {f1:.3f}±{sd:.3f} "
                  f"(n={len(y)}, subj={len(np.unique(g))})", flush=True)

        # 2) transfer terarah PSD + 3) tangent utk pasangan ds003838
        pairs = {}
        for src, tgt, kind in PAIRS:
            Xs, _, _, ys, _ = load(src, scheme)
            Xt, _, _, yt, _ = load(tgt, scheme)
            f1_psd, acc = transfer_f1(Xs, ys, Xt, yt)
            rec = {"kind": kind, "f1_psd": round(f1_psd, 4), "acc_psd": round(acc, 4)}
            if "ds003838" in (src, tgt):
                rec["f1_tangent"] = round(tangent_transfer(src, tgt, scheme), 4)
            pairs[f"{src}->{tgt}"] = rec
            tg = f" tangent {rec.get('f1_tangent')}" if "f1_tangent" in rec else ""
            print(f"  transfer {src:9s}->{tgt:9s} [{kind:13s}] PSD F1 {f1_psd:.3f}{tg}",
                  flush=True)

        # 4) Pool B research-grade
        poolB = {
            "COG-BCI+Hinss->ds003838": round(pool_transfer(["COG-BCI", "Hinss"],
                                                            "ds003838", scheme), 4),
            "COG-BCI+Hinss+ds003838->STEW": round(pool_transfer(
                ["COG-BCI", "Hinss", "ds003838"], "STEW", scheme), 4),
            "COG-BCI+Hinss->STEW": round(pool_transfer(["COG-BCI", "Hinss"],
                                                        "STEW", scheme), 4),
        }
        for k, v in poolB.items():
            print(f"  poolB {k:34s} F1 {v:.3f}", flush=True)

        out[scheme] = {"within_ceiling": ceil, "transfer": pairs, "poolB": poolB}

    # 5) span_graded within-dataset (ds003838)
    Xp, _, _, y, g = load("ds003838", "span_graded")
    f1, sd = within_dataset_f1(Xp, y, g, n_splits=5)
    out["span_graded"] = {"ds003838_within_f1": round(f1, 4),
                          "std": round(sd, 4), "n_epoch": int(len(y)),
                          "n_classes": int(len(np.unique(y)))}
    print(f"\n  span_graded ds003838 (3-tingkat WM): within F1 {f1:.3f}±{sd:.3f}",
          flush=True)

    DATA_OUT.mkdir(exist_ok=True)
    (DATA_OUT / "ds003838_transfer_results.json").write_text(json.dumps(out, indent=2))
    print("\n[08] wrote ds003838_transfer_results.json", flush=True)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()

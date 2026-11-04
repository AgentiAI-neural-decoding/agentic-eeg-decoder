#!/usr/bin/env python3
"""Tahap 05 — Multi-task shared-backbone untuk transfer lintas-dataset (Family D).

Motivasi (riset literatur Tahap 05): pooling naif & re-centering Riemannian gagal
(negative transfer; re-centering menghapus sinyal power). Family D — backbone neural
BERSAMA + head klasifikasi PER-DATASET — adalah ruang kebaruan (nol preseden di
literatur untuk EEG workload) yang:
  - berbagi representasi lintas-dataset (transfer lewat regularisasi representasi),
  - TIDAK memaksa semantik label identik (head terpisah) → menjawab mismatch konstruk,
  - memakai fitur PSD (70-dim, power-preserving) → tak menghapus sinyal magnitude.

Uji: untuk dataset TARGET, bandingkan pada subjek target hold-out (GroupKFold):
  - SINGLE : trunk + head target, dilatih HANYA pada target-train.
  - MULTI  : trunk BERSAMA + dua head, dilatih target-train (head target) + SELURUH
             dataset SUMBER (head sumber). Trunk diregularisasi tugas auxiliary.
Bila MULTI > SINGLE pada target-test → transfer positif via representasi bersama,
yang gagal dicapai pooling/re-centering.

Penggunaan:
    python multitask.py --config ../config/experiments.json --target STEW --source COG-BCI --both
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(42)


def pick_device() -> str:
    """Auto: CUDA (Colab/GPU) > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


class SharedBackbone(nn.Module):
    """Trunk MLP bersama + satu head linear per dataset."""

    def __init__(self, in_dim: int, heads: dict[str, int], hidden=(128, 64), p=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({name: nn.Linear(d, n_cls) for name, n_cls in heads.items()})

    def forward(self, x, head: str):
        return self.heads[head](self.trunk(x))


def _class_weights(y, n_cls):
    cnt = np.bincount(y, minlength=n_cls).astype(float)
    w = len(y) / (n_cls * np.maximum(cnt, 1))
    return torch.tensor(w, dtype=torch.float32)


def train_model(data: dict, in_dim, epochs=40, lr=1e-3, bs=256, head_weights=None, log=None):
    """data: {head_name: (X, y, n_cls)}. Latih multi-task; kembalikan model.

    head_weights: skala loss per head (mis. {"tgt":1.0,"src":0.3}) agar tugas auxiliary
    MEREGULARISASI trunk tanpa mendominasinya.
    """
    heads = {name: d[2] for name, d in data.items()}
    hw = head_weights or {name: 1.0 for name in data}
    model = SharedBackbone(in_dim, heads).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = {name: nn.CrossEntropyLoss(weight=_class_weights(d[1], d[2]).to(DEVICE))
            for name, d in data.items()}
    tensors = {name: (torch.tensor(d[0], dtype=torch.float32),
                      torch.tensor(d[1], dtype=torch.long)) for name, d in data.items()}
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for name, (X, y) in tensors.items():
            perm = torch.randperm(len(X))
            for i in range(0, len(X), bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                out = model(X[idx].to(DEVICE), name)
                loss = crit[name](out, y[idx].to(DEVICE)) * hw[name]
                loss.backward(); opt.step()
                tot += loss.item()
        if log and (ep + 1) % 20 == 0:
            log(f"      epoch {ep+1}/{epochs} loss={tot:.1f}")
    return model


@torch.no_grad()
def predict(model, X, head):
    model.eval()
    out = model(torch.tensor(X, dtype=torch.float32).to(DEVICE), head)
    return out.argmax(1).cpu().numpy()


def load(npz, cap, seed):
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"]; n = len(y)
    idx = np.random.default_rng(seed).choice(n, cap, replace=False) if cap and n > cap else np.arange(n)
    return d["X_psd"][idx], y[idx], g[idx]


def run(target, source, cfg, feat_root, log):
    cvc = cfg["cv"]; seed = cvc.get("random_state", 42); cap = cvc.get("subsample_per_run")
    Xt, yt, gt = load(feat_root / f"{target}_workload_binary.npz", cap, seed)
    Xs, ys, gs = load(feat_root / f"{source}_workload_binary.npz", cap, seed)
    n_cls_t, n_cls_s = len(np.unique(yt)), len(np.unique(ys))
    n_splits = min(cvc["n_splits"], len(np.unique(gt)))
    aux_w = cfg.get("aux_weight", 0.3)
    gkf = GroupKFold(n_splits=n_splits)
    R = {c: {"acc": [], "f1": []} for c in ("single", "multi", "multi_w")}

    for k, (tr, te) in enumerate(gkf.split(Xt, yt, gt)):
        sc = StandardScaler().fit(Xt[tr])
        Xt_tr, Xt_te = sc.transform(Xt[tr]), sc.transform(Xt[te])
        Xs_s = sc.transform(Xs)  # sumber dipetakan ke skala target (ruang fitur sama)
        in_dim = Xt.shape[1]
        tgt_data = {"tgt": (Xt_tr, yt[tr], n_cls_t)}
        multi_data = {"tgt": (Xt_tr, yt[tr], n_cls_t), "src": (Xs_s, ys, n_cls_s)}

        def evalm(model, cond):
            p = predict(model, Xt_te, "tgt")
            R[cond]["acc"].append(accuracy_score(yt[te], p))
            R[cond]["f1"].append(f1_score(yt[te], p, average="macro"))

        evalm(train_model(tgt_data, in_dim, log=log), "single")
        evalm(train_model(multi_data, in_dim, log=log), "multi")
        evalm(train_model(multi_data, in_dim, head_weights={"tgt": 1.0, "src": aux_w}, log=log), "multi_w")
        log(f"    fold {k+1}/{n_splits}: single={R['single']['f1'][-1]:.3f} "
            f"multi={R['multi']['f1'][-1]:.3f} multi_w={R['multi_w']['f1'][-1]:.3f} (F1)")

    def mean(c, m): return round(float(np.mean(R[c][m])), 3)
    out = {"target": target, "source": source, "n_splits": n_splits, "aux_weight": aux_w,
           "single": {"acc": mean("single", "acc"), "f1": mean("single", "f1")},
           "multi": {"acc": mean("multi", "acc"), "f1": mean("multi", "f1")},
           "multi_w": {"acc": mean("multi_w", "acc"), "f1": mean("multi_w", "f1")}}
    base_f1 = out["single"]["f1"]
    out["gain_f1_multi"] = round(out["multi"]["f1"] - base_f1, 3)
    out["gain_f1_multi_w"] = round(out["multi_w"]["f1"] - base_f1, 3)
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
        log(f"\n=== Multi-task shared-backbone: target={tgt} + sumber={src} ===")
        r = run(tgt, src, cfg, feat_root, log)
        results.append(r)
        log(f"  SINGLE F1={r['single']['f1']} | MULTI F1={r['multi']['f1']} (gain {r['gain_f1_multi']:+}) | "
            f"MULTI_W(aux={r['aux_weight']}) F1={r['multi_w']['f1']} (gain {r['gain_f1_multi_w']:+})")

    (out_root / "multitask_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nSelesai. Hasil -> {out_root / 'multitask_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

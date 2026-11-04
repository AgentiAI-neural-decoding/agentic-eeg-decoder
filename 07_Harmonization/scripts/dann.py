#!/usr/bin/env python3
"""Tahap 07 — Harmonisasi domain-adversarial (DANN) yang DIPELAJARI, sebelum decoding.

Motivasi: residual tembok transfer = device/signal DOMAIN SHIFT (Tahap 05-06). Di sini kita
PELAJARI representasi device-invariant: encoder (CNN atau Mamba) memetakan epoch 14-kanal ->
latent; dilatih dengan (a) task-loss workload + (b) domain-adversarial loss via Gradient
Reversal Layer (GRL) yang memaksa latent tak-terbedakan antar-dataset. Harmonisasi datang dari
OBJEKTIF (adversarial), bukan dari backbone; backbone (CNN vs Mamba) di-ablation utk efisiensi.

Bandingkan, pada subjek TARGET hold-out (cross-subject GroupKFold):
  SINGLE : encoder+label head dilatih hanya target-train.
  POOL   : dilatih target-train + SUMBER, tanpa adversary (lambda=0)  -> "pooling yg dipelajari".
  DANN   : idem + domain-adversarial (lambda>0) -> harmonisasi device-invariant.
Hipotesis: DANN > POOL (>= SINGLE) bila invariansi yg dipelajari menutup domain shift.

Encoder default = CNN (jalan di mana saja). Mamba (butuh mamba-ssm + CUDA, mis. Colab) via
--backbone mamba. Disarankan GPU.

Penggunaan:
    python dann.py --config ../config/dann.json --backbone cnn --both
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "06_FoundationModel" / "scripts"))
from fm_encoder import load_harmonized_epochs, resolve_device  # noqa: E402
warnings.filterwarnings("ignore")
torch.manual_seed(42)


# ---------------- Gradient Reversal ----------------
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd):
    return _GRL.apply(x, lambd)


# ---------------- Encoders (backbone ablation) ----------------
class CNNEncoder(nn.Module):
    """EEGNet-style encoder (temporal conv -> depthwise spatial -> separable) -> (B, d).

    Arsitektur baku & kuat untuk raw EEG (Lawhern dkk. 2018); ringan (cocok edge).
    Input (B, ch, T) -> (B, d=F1*D).
    """
    def __init__(self, n_ch, F1=8, D=2, kern=64, p=0.3):
        super().__init__()
        F2 = F1 * D
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False), nn.BatchNorm2d(F1))
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F2, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.separable = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = F2

    def forward(self, x):                 # (B, ch, T)
        x = x.unsqueeze(1)                # (B, 1, ch, T)
        x = self.separable(self.depthwise(self.temporal(x)))
        return self.pool(x).flatten(1)    # (B, F2)


class _MinimalMamba(nn.Module):
    """Blok Mamba pure-PyTorch (selective SSM, referensi; TANPA kernel CUDA fused).

    Algoritma sama dengan mamba-ssm (selective scan) tapi via loop sekuensial -> lebih lambat;
    untuk performa/edge pakai mamba-ssm di CUDA. Dipakai sebagai fallback bila mamba-ssm gagal
    di-install (umum di Colab). Jujur dilabeli sebagai 'mamba(min)'.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.di = expand * d_model
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, self.di * 2, bias=False)
        self.conv1d = nn.Conv1d(self.di, self.di, d_conv, groups=self.di, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.di, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                             .repeat(self.di, 1)))
        self.D = nn.Parameter(torch.ones(self.di))
        self.out_proj = nn.Linear(self.di, d_model, bias=False)
        self.d_state = d_state

    def forward(self, x):                       # (B, L, d_model)
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)             # (B,L,di) each
        xi = self.conv1d(xi.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xi = torch.nn.functional.silu(xi)
        A = -torch.exp(self.A_log)              # (di, d_state)
        dbl = self.x_proj(xi)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt))           # (B,L,di)
        dA = torch.exp(dt.unsqueeze(-1) * A)                          # (B,L,di,d_state)
        dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xi.unsqueeze(-1)   # (B,L,di,d_state)
        h = torch.zeros(B, self.di, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))           # (B,di)
        y = torch.stack(ys, dim=1) + xi * self.D                     # (B,L,di)
        return self.out_proj(y * torch.nn.functional.silu(z))


class MambaEncoder(nn.Module):
    """Mamba (SSM) temporal -> embedding (B, d). Pakai mamba-ssm (CUDA) bila ada, jika tidak
    fallback ke _MinimalMamba pure-PyTorch (lebih lambat, jalan di mana saja)."""
    def __init__(self, n_ch, d=128, n_layer=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.kind = "mamba(ssm)"
            blk = lambda: Mamba(d_model=d)  # noqa: E731
        except Exception:
            self.kind = "mamba(min)"
            blk = lambda: _MinimalMamba(d)  # noqa: E731
        self.proj = nn.Conv1d(n_ch, d, 1)
        self.blocks = nn.ModuleList([blk() for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):                 # x: (B, ch, T)
        h = self.proj(x).transpose(1, 2)  # (B, T, d)
        for b in self.blocks:
            h = b(h) + h
        return self.norm(h).mean(dim=1)   # (B, d)


class DANNModel(nn.Module):
    def __init__(self, encoder, n_cls, n_dom=2):
        super().__init__()
        self.enc = encoder
        d = encoder.out_dim
        self.label = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, n_cls))
        self.domain = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, n_dom))

    def forward(self, x, lambd=0.0):
        z = self.enc(x)
        return self.label(z), self.domain(grad_reverse(z, lambd)), z


# ---------------- Util ----------------
def zscore(X):  # per epoch, per kanal
    mu = X.mean(-1, keepdims=True); sd = X.std(-1, keepdims=True) + 1e-7
    return ((X - mu) / sd).astype("float32")


def build_encoder(name, n_ch):
    return MambaEncoder(n_ch) if name == "mamba" else CNNEncoder(n_ch)


def train(model, Xt, yt, dev, epochs, bs, Xs=None, ys=None, lambd_max=0.0, cw=None, log=None):
    """Xt/yt target-train (selalu dipakai). Xs/ys sumber (opsional). lambd_max>0 -> DANN."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_lab = nn.CrossEntropyLoss(weight=cw.to(dev) if cw is not None else None)
    ce_dom = nn.CrossEntropyLoss()
    # gabung data berlabel utk task; domain: target=1, source=0
    Xall = Xt if Xs is None else np.concatenate([Xt, Xs])
    yall = yt if ys is None else np.concatenate([yt, ys])
    dall = np.concatenate([np.ones(len(Xt)), np.zeros(0 if Xs is None else len(Xs))]).astype(int)
    Xall = torch.tensor(Xall); yall = torch.tensor(yall); dall = torch.tensor(dall)
    n = len(Xall)
    model.train()
    for ep in range(epochs):
        lambd = lambd_max * (2.0 / (1.0 + np.exp(-10 * ep / epochs)) - 1.0)  # ramp DANN
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 4:
                continue
            xb = Xall[idx].to(dev); yb = yall[idx].to(dev); db = dall[idx].to(dev)
            opt.zero_grad()
            ylog, dlog, _ = model(xb, lambd)
            loss = ce_lab(ylog, yb)
            if Xs is not None and lambd_max > 0:
                loss = loss + ce_dom(dlog, db)
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, X, dev, classes):
    model.eval()
    out = []
    Xt = torch.tensor(X)
    for i in range(0, len(Xt), 512):
        ylog, _, _ = model(Xt[i:i + 512].to(dev))
        out.append(ylog.argmax(1).cpu().numpy())
    return classes[np.concatenate(out)]


def run(target, source, backbone, cfg, base, log):
    dev = resolve_device(cfg.get("device", "auto"))
    common = cfg["common_channels"]; sfreq = cfg["sfreq"]; scheme = cfg["label_scheme"]
    cap = cfg["cv"]["max_epochs"]; seed = cfg["cv"]["random_state"]; epochs = cfg["epochs"]; bs = cfg["batch_size"]
    Xt, yt, gt = load_harmonized_epochs((base / cfg["derivatives"][target]).resolve(), target,
                                        common, sfreq, cap, seed, scheme=scheme)
    Xs, ys, _ = load_harmonized_epochs((base / cfg["derivatives"][source]).resolve(), source,
                                       common, sfreq, cap, seed, scheme=scheme)
    Xt, Xs = zscore(Xt), zscore(Xs)
    classes = np.unique(yt); n_cls = len(classes); n_ch = len(common)
    cnt = np.bincount(yt, minlength=n_cls); cw = torch.tensor(len(yt) / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32)
    n_splits = min(cfg["cv"]["n_splits"], len(np.unique(gt)))
    lambdas = cfg.get("dann_lambdas", [cfg.get("lambda_max", 1.0)])  # λ-sweep
    R = {"single": [], "pool": []}
    for lam in lambdas:
        R[f"dann@{lam}"] = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits).split(Xt, yt, gt)):
        Xtr, ytr, Xte = Xt[tr], yt[tr], Xt[te]
        def fit_eval(Xs_, ys_, lam):
            m = DANNModel(build_encoder(backbone, n_ch), n_cls).to(dev)
            train(m, Xtr, ytr, dev, epochs, bs, Xs_, ys_, lam, cw, log)
            return f1_score(yt[te], predict(m, Xte, dev, classes), average="macro")
        R["single"].append(fit_eval(None, None, 0.0))
        R["pool"].append(fit_eval(Xs, ys, 0.0))
        for lam in lambdas:
            R[f"dann@{lam}"].append(fit_eval(Xs, ys, lam))
        dstr = " ".join(f"d@{lam}={R[f'dann@{lam}'][-1]:.3f}" for lam in lambdas)
        log(f"    fold {k+1}/{n_splits}: single={R['single'][-1]:.3f} pool={R['pool'][-1]:.3f} {dstr}")
    # efisiensi backbone
    m = DANNModel(build_encoder(backbone, n_ch), n_cls).to(dev)
    nparam = sum(p.numel() for p in m.parameters())
    xb = torch.tensor(zscore(Xt[:256])).to(dev)
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time(); _ = predict(m, Xt[:256], dev, classes); lat = (time.time() - t0) / 256 * 1000
    out = {"target": target, "source": source, "backbone": backbone,
           "encoder_kind": getattr(m.enc, "kind", backbone), "n_splits": n_splits,
           "device": dev, "params": int(nparam), "latency_ms_per_epoch": round(lat, 3)}
    for c in R:
        out[c] = round(float(np.mean(R[c])), 3)
    dann_keys = [c for c in R if c.startswith("dann@")]
    best = max(dann_keys, key=lambda c: out[c])
    out["best_dann_lambda"] = best.split("@")[1]
    out["best_dann"] = out[best]
    out["gain_pool"] = round(out["pool"] - out["single"], 3)
    out["gain_best_dann"] = round(out["best_dann"] - out["single"], 3)
    out["best_dann_vs_pool"] = round(out["best_dann"] - out["pool"], 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--backbone", default="cnn", choices=["cnn", "mamba"])
    ap.add_argument("--both", action="store_true")
    # Override config via CLI (notebook tak perlu edit file config -> tak ada konflik git pull):
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--cap", type=int, help="max_epochs per dataset")
    ap.add_argument("--k", type=int, help="n_splits")
    ap.add_argument("--lambdas", help="λ-sweep dipisah koma, mis. 0.1,0.3,1.0")
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.epochs:  cfg["epochs"] = args.epochs
    if args.cap:     cfg["cv"]["max_epochs"] = args.cap
    if args.k:       cfg["cv"]["n_splits"] = args.k
    if args.lambdas: cfg["dann_lambdas"] = [float(x) for x in args.lambdas.split(",")]
    if args.device:  cfg["device"] = args.device
    base = args.config.resolve().parent.parent
    out_root = (base / cfg["output_root"]).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    pairs = [("STEW", "COG-BCI")]
    if args.both:
        pairs.append(("COG-BCI", "STEW"))
    results = []
    for tgt, src in pairs:
        log(f"\n=== DANN harmonization: target={tgt} +{src} | backbone={args.backbone} ===")
        r = run(tgt, src, args.backbone, cfg, base, log)
        results.append(r)
        log(f"  SINGLE={r['single']} POOL={r['pool']} ({r['gain_pool']:+}) "
            f"DANN(best λ={r['best_dann_lambda']})={r['best_dann']} ({r['gain_best_dann']:+}) "
            f"| best-DANN vs POOL {r['best_dann_vs_pool']:+} | {r['encoder_kind']} params={r['params']:,} "
            f"lat={r['latency_ms_per_epoch']}ms ({r['device']})")
    (out_root / f"dann_{args.backbone}_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nHasil -> {out_root / f'dann_{args.backbone}_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

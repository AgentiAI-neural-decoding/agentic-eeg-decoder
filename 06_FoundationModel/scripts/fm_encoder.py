"""Tahap 06 — pemuat epoch ter-harmonisasi + antarmuka foundation-model encoder.

Foundation model EEG mengonsumsi sinyal epoch MENTAH (bukan fitur PSD/kovarians),
maka modul ini memuat ulang `.fif` dari Tahap 02, menyamakan ke 14 kanal Emotiv +
128 Hz (sama dgn Tahap 03), lalu menyediakan antarmuka `.embed(X)->emb`.

- `PlaceholderEncoder`: CNN-1D frozen acak (deterministik) untuk MEMVALIDASI harness
  probing/DA end-to-end di MPS sekarang, tanpa bobot eksternal.
- `CBraModEncoder`: stub untuk bobot CBraMod (HuggingFace) — lihat PLAN.md untuk wiring.
"""
from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn

mne.set_log_level("ERROR")


def resolve_device(pref: str = "auto") -> str:
    """Resolusi device: 'auto' -> CUDA (Colab) > MPS (Apple) > CPU; atau hormati pilihan bila tersedia."""
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref in ("cpu",):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# --- Skema label (selaras 03_Features/config/features.json) ---
_SCHEMES = {
    "binary": {  # longgar: rest+easy vs sulit
        "low": {"RS_Beg_EC", "RS_Beg_EO", "RS_End_Ec", "RS_End_EO", "MATBeasy", "zeroBACK"},
        "high": {"MATBmed", "MATBdiff", "oneBACK", "twoBACK", "PVT", "Flanker"},
    },
    "extreme": {  # harmonisasi konstruk: rest vs paling-menuntut (sebanding STEW rest/SIMKAP)
        "low": {"RS_Beg_EC", "RS_Beg_EO", "RS_End_Ec", "RS_End_EO"},
        "high": {"MATBdiff", "twoBACK"},
    },
}


def workload_label(dataset: str, task: str, scheme: str = "binary"):
    if dataset == "STEW":
        return {"lo": 0, "hi": 1}.get(task)
    s = _SCHEMES.get(scheme, _SCHEMES["binary"])
    if task in s["low"]:
        return 0
    if task in s["high"]:
        return 1
    return None  # task ambigu di skema ini → dilewati


def _parse(path: Path, dataset: str) -> dict:
    name = path.stem
    sub = re.search(r"sub-(\w+?)_", name)
    sub = sub.group(1) if sub else "??"
    if dataset == "COG-BCI":
        t = re.search(r"task-(.+?)_run", name)
        return {"subject": sub, "task": t.group(1) if t else "?"}
    t = re.search(r"task-(lo|hi)", name)
    return {"subject": sub, "task": t.group(1) if t else "?"}


def load_harmonized_epochs(deriv_root: Path, dataset: str, common: list[str],
                           sfreq: float, max_epochs: int, seed: int, scheme: str = "binary"):
    """Muat epoch mentah ter-harmonisasi (n, 14, T) + label workload + grup subjek."""
    files = sorted(Path(deriv_root).rglob("*_proc-clean_epo.fif"))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)  # acak agar cap mencakup beragam subjek (bukan hanya yg pertama)
    Xs, ys, gs, total = [], [], [], 0
    for f in files:
        meta = _parse(f, dataset)
        lab = workload_label(dataset, meta["task"], scheme)
        if lab is None:
            continue
        ep = mne.read_epochs(f, verbose="ERROR").pick(common)
        ep.reorder_channels(common)
        if abs(ep.info["sfreq"] - sfreq) > 1e-3:
            ep.resample(sfreq)
        d = ep.get_data(copy=False).astype("float32")
        Xs.append(d); ys += [lab] * len(d); gs += [f"{dataset}-{meta['subject']}"] * len(d)
        total += len(d)
        if total >= max_epochs:
            break
    X = np.concatenate(Xs); y = np.array(ys); g = np.array(gs)
    if len(y) > max_epochs:
        idx = rng.choice(len(y), max_epochs, replace=False)
        X, y, g = X[idx], y[idx], g[idx]
    return X, y, g


class FoundationEncoder:
    """Antarmuka: embed(X[n,ch,T]) -> emb[n,d]. Encoder DIBEKUKAN (frozen)."""

    def embed(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PlaceholderEncoder(FoundationEncoder):
    """CNN-1D frozen acak (deterministik) — validasi harness, BUKAN model terlatih."""

    def __init__(self, n_ch: int, emb_dim: int = 128, device: str = "cpu"):
        torch.manual_seed(0)
        self.device = device  # sudah di-resolve oleh build_encoder/resolve_device
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 64, 7, stride=2), nn.ReLU(),
            nn.Conv1d(64, 64, 7, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, emb_dim),
        ).to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        t = torch.tensor(X, dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, len(t), 512):
            out.append(self.net(t[i:i + 512]).cpu().numpy())
        return np.concatenate(out)


class CBraModEncoder(FoundationEncoder):
    """CBraMod (Wang dkk., ICLR 2025; arXiv:2412.07236) via **braindecode** — frozen probing.

    Bobot pretrained dari HuggingFace `braindecode/cbramod-pretrained`. Arsitektur:
    criss-cross transformer, patch_size=200 (=1 dtk @200 Hz), emb_dim=200.

    PRASYARAT input: epoch **200 Hz**, `n_times` kelipatan 200 (mis. 2 dtk = 400).
    Set `target_sfreq_hz: 200` di config saat memakai encoder ini.

    Catatan normalisasi: input diskalakan Volt→µV lalu /`scale` (default 100, konvensi umum
    CBraMod). Verifikasi terhadap preprocessing dataset di repo bila menuntut presisi penuh.
    """

    HF_DEFAULT = "braindecode/cbramod-pretrained"

    def __init__(self, hf_repo: str, device: str = "cuda", n_chans: int = 14,
                 n_times: int = 400, scale: float = 100.0, norm: str = "scale",
                 pooling: str = "mean"):
        from braindecode.models import CBraMod  # lazy: braindecode hanya perlu utk encoder ini
        self.device = resolve_device(device)
        repo = hf_repo if hf_repo and not hf_repo.startswith("<") else self.HF_DEFAULT
        self.model = CBraMod.from_pretrained(
            repo, n_chans=n_chans, n_times=n_times, sfreq=200, n_outputs=2,
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scale = scale
        self.n_times = n_times  # waktu input harus = n_times (kelipatan patch 200)
        self.norm = norm        # "scale" (µV/scale) atau "zscore" (per-epoch per-kanal)
        self.pooling = pooling  # "mean" (200-dim) atau "meanstd" (400-dim)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.norm == "zscore":  # per epoch, per kanal, sepanjang waktu
            mu = X.mean(axis=-1, keepdims=True)
            sd = X.std(axis=-1, keepdims=True) + 1e-8
            return (X - mu) / sd
        return (X * 1e6) / self.scale  # default: Volt -> µV -> /scale

    def _pool(self, f):
        """Pool fitur (b, ch, patch, emb) -> (b, d) — montase-invariant."""
        dims = tuple(range(1, f.ndim - 1))  # rerata atas kanal & patch, sisakan emb
        if self.pooling == "meanstd":
            return torch.cat([f.mean(dim=dims), f.std(dim=dims)], dim=-1)
        return f.mean(dim=dims)

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        # CBraMod butuh waktu == n_times (kelipatan patch 200). Resample 2 dtk kadang
        # menghasilkan 401 (pembulatan) → crop ke n_times; pad bila terlalu pendek.
        t = X.shape[-1]
        if t >= self.n_times:
            X = X[:, :, :self.n_times]
        elif t < self.n_times:
            X = np.pad(X, ((0, 0), (0, 0), (0, self.n_times - t)))
        Xs = self._normalize(X)
        out = []
        for i in range(0, len(Xs), 128):
            xb = torch.tensor(Xs[i:i + 128], dtype=torch.float32, device=self.device)
            res = self.model(xb, return_features=True)
            f = res["features"] if isinstance(res, dict) else res
            out.append(self._pool(f).float().cpu().numpy())
        return np.concatenate(out)


def build_encoder(name: str, opts: dict, n_ch: int, device: str) -> FoundationEncoder:
    dev = resolve_device(device)
    if name == "placeholder":
        return PlaceholderEncoder(n_ch, emb_dim=opts.get("emb_dim", 128), device=dev)
    if name == "cbramod":
        return CBraModEncoder(opts.get("hf_repo", ""), device=dev, n_chans=n_ch,
                              n_times=opts.get("n_times", 400), scale=opts.get("scale", 100.0),
                              norm=opts.get("norm", "scale"), pooling=opts.get("pooling", "mean"))
    raise ValueError(f"encoder tak dikenal: {name}")

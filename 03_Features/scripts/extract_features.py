#!/usr/bin/env python3
"""Ekstraksi fitur Tahap 03 — transfer-compatible cross-dataset (COG-BCI <-> STEW).

Menghasilkan representasi fitur yang SEBANDING antar-dataset untuk decoding beban
kognitif & transfer learning (hipotesis inti penelitian):

  1. Harmonisasi: pilih 14 kanal Emotiv bersama + resample ke 128 Hz (kedua dataset
     -> ruang fitur berdimensi sama = "kompatibilitas data 100%").
  2. PSD band-power (delta..gamma), relative + log -> 5 x 14 = 70 fitur/epoch (siap-pakai;
     per-epoch sehingga bebas leakage).
  3. Kovarians spasial 14x14/epoch (estimator OAS) -> disimpan untuk Riemannian/CSP yang
     HARUS di-fit di dalam CV (anti-leakage). Tangent-space global juga disimpan
     (referensi geometrik global) untuk eksplorasi cepat — lihat caveat di README.

Output per (dataset, skema): features/{dataset}_{scheme}.npz, plus gabungan lintas-
dataset untuk skema workload_binary, plus feature_manifest.json.

Penggunaan:
    python extract_features.py --config ../config/features.json
    python extract_features.py --config ../config/features.json --datasets COG-BCI
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mne
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import label_for

mne.set_log_level("ERROR")


def find_epoch_files(deriv_root: Path, dataset: str) -> list[Path]:
    if dataset == "COG-BCI":
        return sorted(deriv_root.rglob("*_proc-clean_epo.fif"))
    return sorted(deriv_root.rglob("*_proc-clean_epo.fif"))


def parse_meta(path: Path, dataset: str) -> dict:
    """Ambil subjek, sesi, task dari nama file derivatives."""
    name = path.stem
    sub = re.search(r"sub-(\w+?)_", name)
    sub = sub.group(1) if sub else "??"
    if dataset in ("COG-BCI", "ds003838"):
        ses = re.search(r"ses-(\w+?)_", name)
        task = re.search(r"task-(.+?)_run", name)
        return {"subject": sub, "session": ses.group(1) if ses else "-",
                "task": task.group(1) if task else "?"}
    # STEW: sub-XX_task-{lo,hi}
    task = re.search(r"task-(lo|hi)", name)
    return {"subject": sub, "session": "-", "task": task.group(1) if task else "?"}


def harmonize(epochs: mne.Epochs, common: list[str], target_sfreq: float) -> mne.Epochs:
    """Pilih kanal bersama (urut konsisten) + resample ke target sampling rate."""
    epochs = epochs.copy().pick(common)
    epochs.reorder_channels(common)
    if abs(epochs.info["sfreq"] - target_sfreq) > 1e-3:
        epochs.resample(target_sfreq)
    return epochs


def band_power(epochs: mne.Epochs, cfg: dict) -> np.ndarray:
    """PSD band-power relative+log -> (n_epochs, n_bands * n_channels)."""
    bands = cfg["bands"]
    psd_cfg = cfg["psd"]
    sfreq = epochs.info["sfreq"]
    n_fft = int(psd_cfg["n_fft_s"] * sfreq)
    fmin = min(b[0] for b in bands.values())
    fmax = max(b[1] for b in bands.values())
    spectrum = epochs.compute_psd(method=psd_cfg["method"], fmin=fmin, fmax=fmax,
                                  n_fft=n_fft, verbose="ERROR")
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_ep, n_ch, n_freq)
    total = psds.sum(axis=-1, keepdims=True) + 1e-20
    feats = []
    for lo, hi in bands.values():
        idx = (freqs >= lo) & (freqs < hi)
        bp = psds[:, :, idx].sum(axis=-1)  # (n_ep, n_ch)
        if psd_cfg.get("relative_power"):
            bp = bp / total[:, :, 0]
        feats.append(bp)
    X = np.concatenate(feats, axis=1)  # (n_ep, n_bands*n_ch)
    if psd_cfg.get("log_transform"):
        X = np.log(X + 1e-12)
    return X.astype(np.float32)


def process_dataset(dataset: str, deriv_root: Path, cfg: dict, logger) -> dict:
    """Ekstrak fitur mentah per-epoch untuk satu dataset (semua rekaman)."""
    common = cfg["common_channels"]
    tsf = cfg["target_sfreq_hz"]
    files = find_epoch_files(deriv_root, dataset)
    logger(f"[{dataset}] {len(files)} rekaman .fif")

    X_psd, X_cov, subs, tasks, sess = [], [], [], [], []
    cov_est = Covariances(estimator=cfg["covariance"]["estimator"])
    for i, f in enumerate(files):
        meta = parse_meta(f, dataset)
        try:
            ep = mne.read_epochs(f, verbose="ERROR")
            ep = harmonize(ep, common, tsf)
        except Exception as e:  # noqa: BLE001
            logger(f"  [skip] {f.name}: {e}")
            continue
        if len(ep) == 0:
            continue
        data = ep.get_data(copy=False)  # (n_ep, n_ch, n_times), Volt
        psd = band_power(ep, cfg)
        cov = cov_est.fit_transform(data * 1e6)  # skala µV agar kondisi numerik baik
        n = len(ep)
        X_psd.append(psd)
        X_cov.append(cov.astype(np.float32))
        subs += [meta["subject"]] * n
        tasks += [meta["task"]] * n
        sess += [meta["session"]] * n
        if (i + 1) % 50 == 0:
            logger(f"  ...{i+1}/{len(files)} rekaman")

    return {
        "X_psd": np.concatenate(X_psd) if X_psd else np.empty((0, 0), np.float32),
        "X_cov": np.concatenate(X_cov) if X_cov else np.empty((0, 0, 0), np.float32),
        "subjects": np.array(subs),
        "tasks": np.array(tasks),
        "sessions": np.array(sess),
    }


def apply_scheme(raw: dict, dataset: str, scheme_cfg: dict) -> dict | None:
    """Filter epoch yang punya label valid utk skema -> X_psd/X_cov/y/groups."""
    if dataset not in scheme_cfg.get("datasets", []):
        return None
    y = np.array([label_for(scheme_cfg, dataset, t) for t in raw["tasks"]], dtype=object)
    keep = np.array([v is not None for v in y])
    if keep.sum() == 0:
        return None
    return {
        "X_psd": raw["X_psd"][keep],
        "X_cov": raw["X_cov"][keep],
        "y": y[keep].astype(int),
        "groups": np.array([f"{dataset}-{s}" for s in raw["subjects"][keep]]),
        "tasks": raw["tasks"][keep],
        "subjects": raw["subjects"][keep],
    }


def add_tangent(bundle: dict, cfg: dict) -> np.ndarray:
    """Tangent-space global (referensi geometrik di SELURUH epoch) — eksploratif."""
    if not cfg["covariance"].get("save_tangent_global") or len(bundle["X_cov"]) == 0:
        return np.empty((len(bundle["X_psd"]), 0), np.float32)
    ts = TangentSpace(metric="riemann")
    return ts.fit_transform(bundle["X_cov"]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", help="Subset dataset. Kosong = semua.")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    out_root = (base / cfg["output_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    datasets = args.datasets or list(cfg["inputs"].keys())
    raw_by_ds = {}
    for ds in datasets:
        deriv = (base / cfg["inputs"][ds]).resolve()
        if not deriv.exists():
            log(f"[warn] {ds}: derivatives tidak ada ({deriv}) — dilewati.")
            continue
        raw_by_ds[ds] = process_dataset(ds, deriv, cfg, log)

    manifest = {"common_channels": cfg["common_channels"],
                "target_sfreq_hz": cfg["target_sfreq_hz"],
                "bands": list(cfg["bands"].keys()),
                "n_psd_features": len(cfg["bands"]) * len(cfg["common_channels"]),
                "n_tangent_features": len(cfg["common_channels"]) * (len(cfg["common_channels"]) + 1) // 2,
                "outputs": {}}

    for scheme_name, scheme_cfg in cfg["label_schemes"].items():
        combined = {k: [] for k in ("X_psd", "X_cov", "y", "groups", "tasks")}
        for ds in datasets:
            if ds not in raw_by_ds:
                continue
            b = apply_scheme(raw_by_ds[ds], ds, scheme_cfg)
            if b is None:
                continue
            tangent = add_tangent(b, cfg)
            fn = out_root / f"{ds}_{scheme_name}.npz"
            np.savez_compressed(fn, X_psd=b["X_psd"], X_cov=b["X_cov"],
                                X_tangent=tangent, y=b["y"], groups=b["groups"],
                                tasks=b["tasks"], subjects=b["subjects"])
            classes, counts = np.unique(b["y"], return_counts=True)
            manifest["outputs"][fn.name] = {
                "dataset": ds, "scheme": scheme_name, "n_epochs": int(len(b["y"])),
                "n_subjects": int(len(np.unique(b["subjects"]))),
                "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
            }
            log(f"[save] {fn.name}: {len(b['y'])} epoch, kelas {dict(zip(classes.tolist(), counts.tolist()))}")
            for k in ("X_psd", "X_cov", "y", "groups", "tasks"):
                combined[k].append(b[k])

        # Gabungan lintas-dataset (hanya bila >1 dataset berkontribusi) untuk transfer.
        if len(combined["y"]) > 1:
            cb = {k: np.concatenate(v) for k, v in combined.items()}
            tangent = add_tangent(cb, cfg)
            fn = out_root / f"combined_{scheme_name}.npz"
            np.savez_compressed(fn, X_psd=cb["X_psd"], X_cov=cb["X_cov"],
                                X_tangent=tangent, y=cb["y"], groups=cb["groups"],
                                tasks=cb["tasks"])
            classes, counts = np.unique(cb["y"], return_counts=True)
            manifest["outputs"][fn.name] = {
                "dataset": "COMBINED", "scheme": scheme_name, "n_epochs": int(len(cb["y"])),
                "n_subjects": int(len(np.unique(cb["groups"]))),
                "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
            }
            log(f"[save] {fn.name}: {len(cb['y'])} epoch (gabungan), kelas "
                f"{dict(zip(classes.tolist(), counts.tolist()))}")

    (out_root / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"\nSelesai. Manifest -> {out_root / 'feature_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

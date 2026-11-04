#!/usr/bin/env python3
"""Pipeline preprocessing STEW (14-ch Emotiv EPOC, 128 Hz).

Pasangan file per subjek: subNN_lo.txt (resting) & subNN_hi.txt (SIMKAP multitasking).
Setiap file berisi N_samples x 14 kanal, dipisahkan spasi/komma.

Tahapan:
  1. Muat .txt → mne.io.RawArray dengan montase 10-20
  2. Notch 50 Hz, bandpass 0.5-45 Hz
  3. Re-reference rata-rata
  4. ICA (extended infomax, n=13) → tolak komponen ber-korelasi tinggi dengan
     rerata kanal frontal (proksi EOG, karena Emotiv tak punya kanal EOG dedicated)
  5. Epoch sliding window 2 s overlap 1 s + reject amplitudo
  6. Simpan derivatives/STEW/sub-XX/sub-XX_task-{lo,hi}_proc-clean_epo.fif
  7. Tulis JSON log ICA + plot PSD QC

Penggunaan:
    python preprocess_stew.py --config ../config/stew.json
    python preprocess_stew.py --config ../config/stew.json --subjects 01 02 03
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ICReport, load_config, save_qc_psd, setup_logger, write_report_json


def discover_subjects(raw_root: Path) -> list[tuple[str, Path, Path]]:
    """Pasangkan file lo & hi per subjek. Return list (sub_id, lo_path, hi_path)."""
    files = list(raw_root.rglob("sub*.txt"))
    pairs: dict[str, dict[str, Path]] = {}
    for f in files:
        m = re.search(r"sub[_-]?(\d+).*?(lo|hi)", f.name, re.I)
        if not m:
            continue
        sub = m.group(1).zfill(2)
        cond = m.group(2).lower()
        pairs.setdefault(sub, {})[cond] = f
    return [(s, d["lo"], d["hi"]) for s, d in sorted(pairs.items()) if "lo" in d and "hi" in d]


def load_raw_txt(path: Path, cfg: dict) -> mne.io.RawArray:
    """Muat .txt 14-kolom → mne.io.RawArray."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != len(cfg["channels"]):
        # coba transpose
        if data.ndim == 2 and data.shape[0] == len(cfg["channels"]):
            data = data.T
        else:
            raise ValueError(f"Bentuk tak terduga {data.shape} untuk {path.name}")
    # Emotiv EPOC menyimpan dalam µV → konversi ke V untuk MNE
    data = data.T * 1e-6
    info = mne.create_info(cfg["channels"], cfg["sfreq_hz"], ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage(cfg["montage"], on_missing="warn")
    raw.info["line_freq"] = cfg["line_freq_hz"]
    return raw


def detect_bad_channels_robust(raw: mne.io.BaseRaw, cfg: dict, logger) -> list[str]:
    """Deteksi kanal rusak pada data Emotiv EPOC (14 ch) tanpa LOF.

    Dengan kanal sesedikit ini, LOF tidak handal. Kita pakai dua kriteria robust:
      1. **Flat**: peak-to-peak < ``flat_threshold_uv`` (kanal mati/lepas).
      2. **Noisy**: robust z-score (median + MAD) pada log peak-to-peak; kanal dengan
         z di atas ``robust_z_threshold`` ditandai (mis. T7 saturasi 9000+ µV).

    Deteksi dilakukan setelah bandpass tapi SEBELUM re-referencing, agar satu kanal
    rusak tidak mencemari kanal lain lewat average reference.
    """
    bc = cfg.get("bad_channel_detection", {})
    if not bc.get("enabled", False):
        return []
    data = raw.get_data()  # n_ch x n_times (Volt)
    ptp_uv = np.ptp(data, axis=1) * 1e6
    bads: set[str] = set()

    flat_thr = bc.get("flat_threshold_uv", 1.0)
    for i, p in enumerate(ptp_uv):
        if p < flat_thr:
            bads.add(raw.ch_names[i])

    logp = np.log(ptp_uv + 1e-12)
    med = np.median(logp)
    mad = np.median(np.abs(logp - med)) + 1e-12
    z = 0.6745 * (logp - med) / mad  # robust z (MAD-based)
    zthr = bc.get("robust_z_threshold", 3.5)
    for i in range(len(ptp_uv)):
        if z[i] > zthr:  # hanya outlier amplitudo TINGGI (noisy); flat sudah ditangani
            bads.add(raw.ch_names[i])

    bad_list = sorted(bads, key=raw.ch_names.index)
    if bad_list:
        detail = {raw.ch_names[i]: round(float(ptp_uv[i])) for i in range(len(ptp_uv))
                  if raw.ch_names[i] in bads}
        logger.info("bad channels: %s (ptp µV: %s)", bad_list, detail)
    return bad_list


def reject_frontal_correlated(raw_for_ica: mne.io.BaseRaw, ica: ICA, cfg: dict) -> list[int]:
    """Heuristik tanpa kanal EOG dedicated: korelasikan setiap sumber IC dengan
    rerata kanal frontal. Komponen dengan |r| ≥ threshold ditolak (proxy blink/saccade)."""
    frontal = cfg["artifact_heuristic"]["frontal_channels_for_eog"]
    thr = cfg["artifact_heuristic"]["eog_corr_threshold"]

    picks = [raw_for_ica.ch_names.index(c) for c in frontal if c in raw_for_ica.ch_names]
    if not picks:
        return []
    frontal_signal = raw_for_ica.get_data(picks=picks).mean(axis=0)
    sources = ica.get_sources(raw_for_ica).get_data()  # shape: n_comp x n_samples
    rejected = []
    for i, src in enumerate(sources):
        r = np.corrcoef(src, frontal_signal)[0, 1]
        if abs(r) >= thr:
            rejected.append(i)
    return rejected


def preprocess_condition(raw_path: Path, sub: str, cond: str, cfg: dict,
                         deriv_root: Path, logger) -> ICReport | None:
    tag = f"sub-{sub}_task-{cond}"
    logger.info("=== %s ===", tag)
    out_dir = deriv_root / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)
    epo_path = out_dir / f"{tag}_proc-clean_epo.fif"
    if epo_path.exists():
        logger.info("sudah ada → skip: %s", epo_path.name)
        return None

    raw = load_raw_txt(raw_path, cfg)

    if cfg.get("qc_plots"):
        save_qc_psd(raw, deriv_root.parent.parent / "reports" / "STEW" / "psd_pre" / f"{tag}.png",
                    title=f"{tag} | PSD pre-filter")

    raw.notch_filter(freqs=cfg["notch_freqs_hz"], verbose="ERROR")
    raw.filter(l_freq=cfg["bandpass_hz"][0], h_freq=cfg["bandpass_hz"][1], verbose="ERROR")

    # Deteksi + interpolasi bad-channel SEBELUM average-ref (kanal rusak seperti T7
    # saturasi mencemari semua kanal lewat average reference bila tak ditangani).
    bads = detect_bad_channels_robust(raw, cfg, logger)
    bc = cfg.get("bad_channel_detection", {})
    max_frac = bc.get("max_bad_fraction", 0.30)
    low_quality = len(bads) > max_frac * len(raw.ch_names)
    note = ""
    if low_quality:
        logger.warning("sub-%s/%s: %d/%d kanal rusak (> %.0f%%) — rekaman LOW-QUALITY.",
                       sub, cond, len(bads), len(raw.ch_names), max_frac * 100)
        note = f"low-quality: {len(bads)}/{len(raw.ch_names)} bad channels"
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")

    raw.set_eeg_reference(cfg["reference"], projection=False, verbose="ERROR")

    # Rank menyusut setelah interpolasi (kanal terinterpolasi = kombinasi linear) dan
    # average-ref (−1). Batasi n_components ICA agar tidak melebihi rank efektif.
    rank = len(raw.ch_names) - len(bads) - 1
    n_comp = min(cfg["ica"]["n_components"], max(1, rank))
    raw_for_ica = raw.copy().filter(l_freq=cfg["ica_highpass_hz"], h_freq=None, verbose="ERROR")
    ica = ICA(
        n_components=n_comp,
        method=cfg["ica"]["method"],
        fit_params=cfg["ica"].get("fit_params"),
        random_state=cfg["ica"]["random_state"],
        max_iter=cfg["ica"]["max_iter"],
    )
    ica.fit(raw_for_ica, verbose="ERROR")
    logger.info("ICA fit: %d komponen (rank≈%d, bad=%d)", ica.n_components_, rank, len(bads))

    rejected = reject_frontal_correlated(raw_for_ica, ica, cfg)
    logger.info("frontal-corr menolak %d komponen: %s", len(rejected), rejected)

    ica.exclude = rejected
    ica.apply(raw, verbose="ERROR")

    # Sliding window
    epo_cfg = cfg["epoching"]
    step = epo_cfg["epoch_length_s"] - epo_cfg["overlap_s"]
    events = mne.make_fixed_length_events(raw, duration=step)
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=epo_cfg["epoch_length_s"],
        baseline=None, reject=epo_cfg["reject_uv"],
        preload=True, verbose="ERROR",
    )
    # Inject label via metadata
    epochs.metadata = None
    import pandas as pd
    epochs.metadata = pd.DataFrame({"label": [cfg["label_mapping"][cond]] * len(epochs)})
    retention = len(epochs) / len(events) if len(events) else 0.0
    logger.info("epoch tersisa: %d / %d (%.0f%%)", len(epochs), len(events), retention * 100)

    notes = "rejection: frontal-corr heuristic"
    if note:
        notes += f" | {note}"

    # Gate kualitas: recording yang retensi epoch-nya terlalu rendah (mis. sub-01_lo
    # yang rusak global) dieksklusi — .fif tidak disimpan agar tak mencemari decoding.
    min_ret = cfg.get("epoching", {}).get("min_epoch_retention", 0.3)
    if len(epochs) == 0 or retention < min_ret:
        notes += f" | EXCLUDED: retensi {retention*100:.0f}% < {min_ret*100:.0f}%"
        logger.warning("sub-%s/%s DIEKSKLUSI: retensi %.0f%% < %.0f%% — .fif tidak disimpan.",
                       sub, cond, retention * 100, min_ret * 100)
    else:
        epochs.save(epo_path, overwrite=True, verbose="ERROR")

    report = ICReport(
        subject=sub, session="-", task=cond,
        n_components=int(ica.n_components_),
        rejected_indices=rejected,
        bad_channels=bads,
        n_epochs_kept=len(epochs),
        n_epochs_total=len(events),
        notes=notes,
    )
    write_report_json(report, out_dir / f"{tag}_ica-report.json")

    if cfg.get("qc_plots"):
        save_qc_psd(raw, deriv_root.parent.parent / "reports" / "STEW" / "psd_post" / f"{tag}.png",
                    title=f"{tag} | PSD post-clean")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--subjects", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = args.config.resolve().parent.parent
    raw_root = (base / cfg["raw_root"]).resolve()
    deriv_root = (base / cfg["output_root"]).resolve()
    deriv_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("stew", log_file=base / "reports" / "preprocess_stew.log")
    logger.info("Raw root: %s", raw_root)
    logger.info("Derivatives: %s", deriv_root)

    if not raw_root.exists():
        logger.error("Raw root tidak ditemukan: %s", raw_root)
        return 1

    pairs = discover_subjects(raw_root)
    if args.subjects:
        wanted = {s.zfill(2) for s in args.subjects}
        pairs = [p for p in pairs if p[0] in wanted]
    logger.info("Memproses %d subjek", len(pairs))

    for sub, lo, hi in pairs:
        for cond, p in (("lo", lo), ("hi", hi)):
            try:
                preprocess_condition(p, sub, cond, cfg, deriv_root, logger)
            except Exception as e:
                logger.exception("gagal sub-%s/%s: %s", sub, cond, e)
    logger.info("selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

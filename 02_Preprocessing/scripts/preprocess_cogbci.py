#!/usr/bin/env python3
"""Pipeline preprocessing COG-BCI (BIDS, 32-ch EEG).

Tahapan per (subject, session, task/run):
  1. Muat raw EEGLAB (.set/.fdt) lewat MNE
  2. Set montage standard_1020, channel types
  3. Notch 50/100/150 Hz, bandpass 0.5-45 Hz
  4. Resample 250 Hz
  5. Deteksi bad channel via LOF + interpolasi
  6. Re-reference rata-rata
  7. ICA (extended infomax, n_comp=99% varian) pada salinan high-pass 1 Hz
  8. ICLabel → tolak komponen non-otak (eye, muscle, heart, line, channel)
  9. Terapkan ICA bersih ke raw original
 10. Buat epoch tetap 2 detik + reject amplitudo
 11. Simpan ke derivatives/COG-BCI/sub-XX/ses-SY/eeg/*-epo.fif
 12. Tulis JSON log ICA + plot PSD QC

Penggunaan:
    python preprocess_cogbci.py --config ../config/cogbci.json --subjects 01 02
    python preprocess_cogbci.py --config ../config/cogbci.json  # semua subjek
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA, find_bad_channels_lof

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ICReport, load_config, save_qc_psd, setup_logger, write_report_json


def find_eeg_files(bids_root: Path, sub: str) -> list[Path]:
    """Cari semua rekaman .set untuk satu subjek.

    Dataset COG-BCI (Zenodo 6874129) menamai berkas menurut task EEGLAB
    (mis. ``Flanker.set``, ``MATBdiff.set``, ``oneBACK.set``) di dalam
    ``sub-XX/ses-SY/eeg/`` — bukan nama BIDS penuh ``*_eeg.set``. Jadi kita
    glob semua ``*.set`` di bawah folder subjek.
    """
    return sorted((bids_root / f"sub-{sub}").rglob("*.set"))


def parse_bids_entities(path: Path) -> dict[str, str]:
    """Derivasi entitas dari struktur path COG-BCI.

    Layout: ``<root>/sub-XX/ses-SY/eeg/<TASK>.set`` → subjek & sesi diambil dari
    folder, task dari nama berkas (mis. ``MATBdiff``, ``zeroBACK``, ``RS_Beg_EC``).
    COG-BCI tidak memakai entitas ``run`` → default "1".
    """
    out: dict[str, str] = {"run": "1"}
    for part in path.parts:
        if part.startswith("sub-"):
            out["sub"] = part.split("-", 1)[1]
        elif part.startswith("ses-"):
            out["ses"] = part.split("-", 1)[1]
    # task = nama berkas tanpa ekstensi (sudah deskriptif & aman untuk nama file)
    out["task"] = path.stem
    return out


def preprocess_one(raw_path: Path, cfg: dict, derivatives_root: Path, logger) -> ICReport | None:
    ents = parse_bids_entities(raw_path)
    sub, ses = ents.get("sub", "??"), ents.get("ses", "??")
    task = ents.get("task", "rest")
    run = ents.get("run", "1")
    tag = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"
    logger.info("=== %s ===", tag)

    out_dir = derivatives_root / f"sub-{sub}" / f"ses-{ses}" / "eeg"
    out_dir.mkdir(parents=True, exist_ok=True)
    epo_path = out_dir / f"{tag}_proc-clean_epo.fif"
    if epo_path.exists():
        logger.info("sudah ada → skip: %s", epo_path.name)
        return None

    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose="ERROR")
    raw.set_montage(cfg["montage"], on_missing="warn")

    # Buang kanal non-EEG (mis. ECG1 = sinyal jantung) & kanal tanpa posisi montase.
    # Keduanya merusak interpolate_bads (posisi NaN → NaN menjalar) dan, untuk ECG,
    # mencemari average-reference dengan sinyal jantung. COG-BCI memuat 2 varian
    # montase (63/64 ch) lintas-subjek; pembersihan ini menyamakan ke kanal EEG sah.
    non_eeg = [ch for ch in raw.ch_names
               if ch.upper().startswith(("ECG", "EOG", "EMG", "EXG", "STATUS", "TRIG", "GSR"))]
    if non_eeg:
        raw.drop_channels(non_eeg)
        logger.info("drop non-EEG: %s", non_eeg)
    no_pos = [raw.ch_names[i] for i, ch in enumerate(raw.info["chs"])
              if np.isnan(ch["loc"][:3]).any() or np.allclose(ch["loc"][:3], 0.0)]
    if no_pos:
        raw.drop_channels(no_pos)
        logger.info("drop kanal tanpa posisi montase: %s", no_pos)

    if cfg.get("line_freq_hz"):
        raw.info["line_freq"] = cfg["line_freq_hz"]

    # PSD sebelum filter
    if cfg.get("qc_plots"):
        save_qc_psd(raw, derivatives_root.parent / "reports" / "psd_pre" / f"{tag}.png",
                    title=f"{tag} | PSD pre-filter")

    raw.notch_filter(freqs=cfg["notch_freqs_hz"], verbose="ERROR")
    raw.filter(l_freq=cfg["bandpass_hz"][0], h_freq=cfg["bandpass_hz"][1], verbose="ERROR")
    if cfg.get("resample_hz"):
        raw.resample(cfg["resample_hz"], verbose="ERROR")

    # Bad channel detection
    bads: list[str] = []
    bad_cfg = cfg["bad_channel_detection"]
    if bad_cfg["method"] == "lof":
        try:
            bads = find_bad_channels_lof(raw, threshold=bad_cfg["threshold"])
        except Exception as e:
            logger.warning("LOF gagal (%s) — lewati deteksi bad channel.", e)
            bads = []
        raw.info["bads"] = bads
        logger.info("bad channels: %s", bads)
        if bads:
            raw.interpolate_bads(verbose="ERROR")

    raw.set_eeg_reference(cfg["reference"], projection=False, verbose="ERROR")

    # ICA pada salinan high-pass agar stabil
    raw_for_ica = raw.copy().filter(l_freq=cfg["ica_highpass_hz"], h_freq=None, verbose="ERROR")
    ica = ICA(
        n_components=cfg["ica"]["n_components"],
        method=cfg["ica"]["method"],
        fit_params=cfg["ica"].get("fit_params"),
        random_state=cfg["ica"]["random_state"],
        max_iter=cfg["ica"]["max_iter"],
    )
    ica.fit(raw_for_ica, verbose="ERROR")
    logger.info("ICA fit: %d komponen", ica.n_components_)

    report = ICReport(
        subject=sub, session=ses, task=f"{task}_run-{run}",
        n_components=int(ica.n_components_),
        bad_channels=list(bads),
    )

    rejected: list[int] = []
    if cfg["iclabel"]["enabled"]:
        try:
            from mne_icalabel import label_components

            labels_out = label_components(raw_for_ica, ica, method="iclabel")
            ic_labels = labels_out["labels"]
            ic_probs = labels_out["y_pred_proba"]
            min_prob = cfg["iclabel"]["min_prob"]
            reject_set = set(cfg["iclabel"]["reject_labels"])
            for i, (lbl, prob) in enumerate(zip(ic_labels, ic_probs)):
                report.labels[i] = lbl
                report.probabilities[i] = float(prob)
                if lbl in reject_set and prob >= min_prob:
                    rejected.append(i)
            logger.info("ICLabel tolak %d komponen: %s", len(rejected), rejected)
        except ImportError:
            logger.warning("mne-icalabel tidak tersedia — lewati klasifikasi otomatis.")
            report.notes = "iclabel-unavailable"
        except Exception as e:
            logger.warning("ICLabel error: %s — lewati.", e)
            report.notes = f"iclabel-error: {e}"

    report.rejected_indices = rejected
    ica.exclude = rejected
    ica.apply(raw, verbose="ERROR")

    # Epoching tetap 2 detik
    epo_cfg = cfg["epoching"]
    events = mne.make_fixed_length_events(raw, duration=epo_cfg["tmax_s"])
    epochs = mne.Epochs(
        raw, events, tmin=epo_cfg["tmin_s"], tmax=epo_cfg["tmax_s"],
        baseline=epo_cfg["baseline"], reject=epo_cfg["reject_uv"],
        preload=True, verbose="ERROR",
    )
    logger.info("epoch tersisa: %d / %d", len(epochs), len(events))

    epochs.save(epo_path, overwrite=True, verbose="ERROR")
    write_report_json(report, out_dir / f"{tag}_ica-report.json")

    if cfg.get("qc_plots"):
        save_qc_psd(raw, derivatives_root.parent / "reports" / "psd_post" / f"{tag}.png",
                    title=f"{tag} | PSD post-clean")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--subjects", nargs="*", help="Daftar XX (01..29). Kosong = semua subjek terdeteksi.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = args.config.resolve().parent.parent
    bids_root = (base / cfg["bids_root"]).resolve()
    deriv_root = (base / cfg["output_root"]).resolve()
    deriv_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("cogbci", log_file=base / "reports" / "preprocess_cogbci.log")
    logger.info("BIDS root: %s", bids_root)
    logger.info("Derivatives: %s", deriv_root)

    if not bids_root.exists():
        logger.error("BIDS root tidak ditemukan: %s", bids_root)
        return 1

    if args.subjects:
        subs = [s.zfill(2) for s in args.subjects]
    else:
        subs = sorted(p.name.split("-")[1] for p in bids_root.glob("sub-*") if p.is_dir())

    if not subs:
        logger.error("Tidak ada subjek ditemukan. Apakah ZIP sudah diekstrak?")
        return 2

    logger.info("Memproses %d subjek: %s", len(subs), subs)
    for sub in subs:
        files = find_eeg_files(bids_root, sub)
        logger.info("sub-%s: %d rekaman", sub, len(files))
        for f in files:
            try:
                preprocess_one(f, cfg, deriv_root, logger)
            except Exception as e:
                logger.exception("gagal %s: %s", f.name, e)
    logger.info("selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

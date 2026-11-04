# 02 — Preprocessing & ICA

Tahap kedua **Pelaksanaan Penelitian Neural Decoding Berbasis Agentic AI**.
Input: data mentah dari `01_Akuisisi_Data/raw/`. Output: epoch bersih `.fif` siap-fitur
di `derivatives/`, plus log ICA per rekaman + plot PSD QC.

## 1. Filosofi pipeline

Pipeline mengikuti rekomendasi PREP + MNE community (Bigdely-Shamlo 2015, Pedroni 2019)
dan dirancang **reproducible** (config-driven, JSON), **deterministik**
(`random_state` tetap), dan **non-destruktif** (raw asli tidak ditimpa). Untuk klasifikasi
artefak ICA digunakan **ICLabel** (Pion-Tonachini 2019) pada COG-BCI yang punya 32-kanal;
pada STEW yang hanya 14-kanal Emotiv EPOC digunakan heuristik korelasi-frontal sebagai
proksi EOG karena ICLabel tidak handal di bawah ~20 kanal.

| Langkah | COG-BCI (32 ch) | STEW (14 ch) |
|---------|-----------------|--------------|
| Loading | EEGLAB `.set`/`.fdt` (BIDS) | Plain `.txt` 14-kol |
| Montase | standard_1020 | standard_1020 |
| Notch | 50, 100, 150 Hz | 50 Hz |
| Bandpass | 0.5–45 Hz | 0.5–45 Hz |
| Resample | 250 Hz | tidak (sudah 128) |
| Bad ch detection | LOF (thr=1.5) + interpolasi | tidak — terlalu sedikit kanal |
| Re-reference | average | average |
| ICA | extended infomax, 99% varian | extended infomax, n=13 |
| Rejeksi IC | **ICLabel** ≥ 0.7 prob | korelasi kanal frontal ≥ 0.6 |
| Epoching | fixed 2 s | sliding 2 s, overlap 1 s |
| Reject ampl. | 150 µV | 200 µV |

## 2. Instalasi environment

Karena disk utama hampir penuh, **gunakan venv di disk eksternal/cloud**:

```bash
# Misal di /Volumes/EEG_Data
python3 -m venv /Volumes/EEG_Data/venv-eeg
source /Volumes/EEG_Data/venv-eeg/bin/activate
pip install -r requirements.txt
```

Alternatif tanpa ICLabel (lebih ringan, tanpa PyTorch/ONNX):

```bash
pip install mne>=1.7 mne-bids>=0.15 numpy scipy pandas matplotlib
# lalu set "iclabel.enabled": false di config/cogbci.json
```

## 3. Cara jalan

### COG-BCI

```bash
cd 02_Preprocessing
# Pilot — 3 subjek dulu (cocok dengan unduhan subset di tahap 01)
python scripts/preprocess_cogbci.py --config config/cogbci.json --subjects 01 02 03

# Setelah unduh penuh & disk cukup
python scripts/preprocess_cogbci.py --config config/cogbci.json
```

### STEW

```bash
python scripts/preprocess_stew.py --config config/stew.json
# atau subset
python scripts/preprocess_stew.py --config config/stew.json --subjects 01 02 03
```

## 4. Output

```
02_Preprocessing/
├── derivatives/
│   ├── COG-BCI/
│   │   └── sub-XX/ses-SY/eeg/
│   │       ├── sub-XX_ses-SY_task-MATB_run-1_proc-clean_epo.fif
│   │       └── sub-XX_ses-SY_task-MATB_run-1_ica-report.json
│   └── STEW/
│       └── sub-XX/
│           ├── sub-XX_task-lo_proc-clean_epo.fif   # resting
│           ├── sub-XX_task-hi_proc-clean_epo.fif   # SIMKAP
│           └── sub-XX_task-{lo,hi}_ica-report.json
└── reports/
    ├── preprocess_cogbci.log
    ├── preprocess_stew.log
    ├── psd_pre/  psd_post/    # PSD sebelum/sesudah pembersihan
    └── STEW/psd_pre/ psd_post/
```

Setiap `.fif` siap dimuat dengan `mne.read_epochs(...)` untuk tahap berikutnya
(ekstraksi fitur / decoding).

## 5. Quality control manual

Setelah batch selesai, periksa minimal:

1. **PSD post-clean**: pita 8–13 Hz (alpha) seharusnya terlihat jelas saat resting, line-noise 50 Hz harus rata.
2. **`ica-report.json`** per rekaman: jumlah komponen tertolak wajar di rentang ~10–40% (terlalu sedikit = artefak lolos; terlalu banyak = ICLabel terlalu agresif).
3. **Jumlah epoch tersisa**: jika < 50% dari total → kandidat outlier, catat di logbook.
4. **Bad channels**: kalau ada subjek dengan > 4 bad channels, pertimbangkan eksklusi.

Hasil QC dituangkan ke logbook eksperimen.

## 6. Reproducibility

- Semua parameter di `config/*.json` (versi-kontrol di git/Zenodo bila publikasi).
- `random_state=42` tetap untuk ICA → komponen identik bila dijalankan ulang.
- Format output `.fif` adalah standar MNE — kompatibel lintas-bahasa lewat FieldTrip/EEGLAB-io.

## 7. Referensi metode utama

- Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., & Robbins, K. A. (2015). *The PREP pipeline: standardized preprocessing for large-scale EEG analysis.* Front. Neuroinform.
- Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). *ICLabel: An automated electroencephalographic independent component classifier, dataset, and website.* NeuroImage 198.
- Gramfort, A., et al. (2014). *MNE software for processing MEG and EEG data.* NeuroImage 86.

---

Lanjut ke **03_Features** (ekstraksi PSD/CSP/Riemannian + persiapan tensor untuk Agentic AI decoder)
setelah QC OK pada setidaknya pilot 3 subjek.

# Logbook Preprocessing — Tahap 02

Catatan pelaksanaan preprocessing + ICA untuk penelitian *Neural Decoding Berbasis
Agentic AI* (DIKTI 2026). Input: `01_Akuisisi_Data/raw/`. Output: epoch bersih `.fif`
di `derivatives/` + log ICA JSON + plot PSD QC.

## Lingkungan

- **venv**: Python 3.12.10 (`../.venv`) via `uv`. *Catatan: Python 3.14 global tidak
  dipakai karena `torch`/`onnxruntime` (dependensi ICLabel) belum punya wheel stabil.*
- Paket kunci: mne 1.12.1, mne-icalabel 0.9.0, torch 2.12.0, onnxruntime 1.26.0,
  numpy 2.4.6, scipy 1.17.1, scikit-learn 1.8.0.

---

## Perbaikan skrip yang diperlukan sebelum jalan

### COG-BCI — penamaan file (BLOCKER, 2026-05-31)

Dataset Zenodo COG-BCI menamai berkas per task EEGLAB (`Flanker.set`, `MATBdiff.set`,
`oneBACK.set`, …) di `sub-XX/ses-SY/eeg/`, **bukan** BIDS penuh `*_eeg.set`.
`preprocess_cogbci.py` semula meng-glob `*_eeg.set` → menemukan **0** dari 1044 file.

Perbaikan ([`scripts/preprocess_cogbci.py`](scripts/preprocess_cogbci.py)):
- `find_eeg_files`: glob `*.set` (bukan `*_eeg.set`).
- `parse_bids_entities`: derivasi subjek/sesi dari **path folder**, task dari nama berkas.

### COG-BCI — kanal ECG & varian montase (2026-05-31)

Full run pertama 29 subjek menghasilkan **168 kegagalan** "non-finite (NaN/Inf)",
terkonsentrasi di 5 subjek (sub-17, 25, 26, 28, 29). Akar penyebab:
- COG-BCI memuat **2 varian montase** lintas-subjek (63 vs 64 kanal; varian
  {Cz,F9,F10,P9,P10} vs {FT9,FT10,CP5,TP10}).
- Kanal **`ECG1`** (sinyal jantung) berlabel tipe `eeg` namun **tanpa posisi montase**
  → `interpolate_bads()` menjalarkan NaN ke seluruh kanal.

Perbaikan ([`scripts/preprocess_cogbci.py`](scripts/preprocess_cogbci.py)) — sekaligus
benar metodologis (ECG tak boleh ikut average-reference EEG):
- Drop kanal non-EEG (`ECG/EOG/EMG/EXG/STATUS/TRIG/GSR`) + kanal tanpa posisi montase.
- Perbaikan pelaporan: `bad_channels` kini mencatat kanal yang diinterpolasi (sebelumnya
  `[]` karena `interpolate_bads` mereset daftar bad).
- **Seluruh 29 subjek diproses ulang** demi konsistensi (reref tanpa ECG seragam).
- Hasil ulang: **0 kegagalan, 1044/1044 `.fif`**, retensi epoch median 97% (3 rekaman < 50%).

### STEW — deteksi bad-channel (peningkatan metodologis, 2026-05-31)

Pilot awal STEW mengungkap kehilangan epoch massal (sub-01_lo: **0/150**). Diagnosis:
data Emotiv EPOC memuat kanal patologis (mis. T7 = 9432 µV ptp) yang, lewat **average
reference**, mencemari seluruh kanal → hampir semua epoch melewati ambang reject 200 µV.
Pipeline STEW semula sengaja tak punya deteksi bad-channel ("terlalu sedikit kanal").

Keputusan (atas arahan: *"lakukan yang paling benar secara metodologis"*) — tambahkan
penanganan bad-channel principled, bukan menaikkan ambang reject (yang meloloskan artefak):
- `detect_bad_channels_robust`: flag kanal **flat** (ptp < 1 µV) & **noisy** (robust
  z-score MAD pada log-ptp > 3.5).
- Interpolasi via montase standard_1020 **sebelum** average-ref.
- `n_components` ICA dibatasi ke rank efektif (turun akibat interpolasi + average-ref).
- **Gate retensi**: recording dengan retensi epoch < 30% (rusak global, tak terselamatkan
  interpolasi) dieksklusi — `.fif` tidak disimpan, ditandai `EXCLUDED` di laporan JSON.
- Config baru di [`config/stew.json`](config/stew.json): blok `bad_channel_detection` +
  `epoching.min_epoch_retention`.
- Field reproducibility baru di `ICReport` (`bad_channels`, `n_epochs_kept/total`).

---

## Hasil & QC

### COG-BCI — penuh 29 subjek · 2026-05-31

| Metrik | Nilai |
|--------|-------|
| Rekaman diproses | **1044/1044** (29 subj × 3 sesi × 12 task) → `.fif`, **0 kegagalan** |
| Kanal EEG | 63 (varian montase; ECG1 di-drop) — *catatan: data riil 63 ch, bukan 32 spt manifest* |
| Retensi epoch | median **97%**, min 14%, **3 rekaman < 50%** |
| Komponen ICA | median 21, varian 99% |
| IC ditolak (ICLabel ≥ 0.7) | rasio median **20%** (sehat 10–40%) |
| Rejeksi per task (pilot) | tertinggi PVT & twoBACK (beban tinggi), terendah resting |

Kualitas riset-grade — bersih. **QC: LULUS.** (Pilot awal 3 subjek juga lulus; full run
menggantikannya setelah perbaikan ECG.)

### STEW — penuh 48 subjek · 2026-05-31

| Metrik | Nilai |
|--------|-------|
| Rekaman dipakai | **86/96 (90%)** → `.fif` |
| Dieksklusi (retensi < 30%) | **10** — terdokumentasi di bawah |
| Retensi epoch (dipakai) | median **77%**, mean 67%, maks 99% |
| Bad-channel diinterpolasi | median 1/rekaman, maks 4, total 87 |
| Label balance | 44 `lo` (resting) vs 42 `hi` (SIMKAP) — seimbang untuk klasifikasi biner |

**Rekaman dieksklusi (10)** — Emotiv consumer-grade, rusak global tak terselamatkan:
sub-01 (lo+hi), sub-02 (hi), sub-06 (lo), sub-18 (hi), sub-22 (lo), sub-27 (hi),
sub-30 (lo), sub-34 (hi), sub-45 (hi). Untuk dicatat di BAB Metode sebagai data-quality
exclusion (kriteria pra-tentukan: retensi epoch < 30%).

---

## Output

- `derivatives/COG-BCI/sub-XX/ses-SY/eeg/*_proc-clean_epo.fif` + `*_ica-report.json`
- `derivatives/STEW/sub-XX/*_task-{lo,hi}_proc-clean_epo.fif` + `*_ica-report.json`
- `reports/psd_pre|psd_post/*.png` (COG-BCI), `reports/STEW/psd_pre|psd_post/*.png`
- Ukuran: derivatives ≈ 1,9 GB (pilot COG-BCI + STEW penuh).

## Sisa pekerjaan Tahap 02

- [ ] Preprocessing COG-BCI **penuh 29 subjek** (pilot sudah lolos QC) — `derivatives` akan jauh lebih besar.
- [ ] QC visual manual plot PSD post-clean (alpha 8–13 Hz saat resting; line-noise 50 Hz rata).

Lanjut ke **03_Features** setelah COG-BCI penuh diproses.

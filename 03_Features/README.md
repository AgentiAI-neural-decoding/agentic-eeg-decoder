# 03 — Ekstraksi Fitur (Transfer-Compatible Cross-Dataset)

Tahap ketiga **Pelaksanaan Penelitian Neural Decoding Berbasis Agentic AI**. Input:
epoch bersih `.fif` dari `02_Preprocessing/derivatives/`. Output: tensor fitur +
label + grup subjek di `features/`, siap untuk decoding (Tahap 04–05).

## 1. Filosofi — selaras tujuan penelitian

Hipotesis inti penelitian adalah **transfer learning lintas-dataset**: pola beban
kognitif dari dataset klinis (COG-BCI, STEW) dapat ditransfer. Konsekuensi metodologis
(eksplisit di proposal: *"harmonisasi sampling rate"*, *"kompatibilitas data 100%"*):
fitur **wajib sebanding antar-dataset** — dimensi & makna identik. Karena itu:

| Langkah harmonisasi | Keputusan | Alasan |
|---------------------|-----------|--------|
| Ruang kanal | 14 kanal Emotiv bersama (ada di COG-BCI 63-ch & STEW 14-ch) | dimensi fitur identik → transfer |
| Sampling | resample → 128 Hz (native STEW) | "harmonisasi sampling rate" |
| Skala | µV untuk kovarians (kondisi numerik) | stabilitas estimator |

## 2. Fitur yang diekstrak (per epoch, ruang 14 kanal)

| Fitur | Dimensi | Sifat | Catatan leakage |
|-------|--------:|-------|-----------------|
| **PSD band-power** | 70 (5 pita × 14 ch) | relative + log (delta/theta/alpha/beta/gamma) | per-epoch → **bebas leakage**, siap-pakai |
| **Kovarians spasial** | 14×14 | estimator OAS | disimpan mentah → CSP/tangent **di-fit dalam CV** |
| **Tangent-space (global)** | 105 | referensi geometrik global | ⚠️ referensi di-fit di SELURUH epoch → **hanya untuk eksplorasi**; untuk evaluasi sahih, fit `TangentSpace` di dalam fold dari `X_cov` |

> **Penting (anti-leakage):** untuk hasil decoding yang valid, CSP & Riemannian tangent
> harus di-*fit* hanya pada data latih tiap fold. Gunakan `X_cov` + pipeline sklearn
> (`pyriemann.TangentSpace` / `mne.decoding.CSP`) di dalam cross-validation. `X_tangent`
> yang tersimpan memakai referensi global dan disediakan hanya untuk inspeksi cepat.

## 3. Skema label (config-driven)

| Skema | Dataset | Kelas | Peran |
|-------|---------|-------|-------|
| `workload_binary` | COG-BCI + STEW | rendah(0) / tinggi(1) | **target PRIMER transfer** — selaras STEW lo/hi |
| `nback_graded` | COG-BCI | 0/1/2-back | sekunder, beban memori kerja |
| `matb_graded` | COG-BCI | easy/med/diff | sekunder, beban tugas MATB |

Pemetaan `workload_binary` COG-BCI: **rendah** = resting (RS_*) + MATBeasy + zeroBACK;
**tinggi** = MATBmed/diff, one/twoBACK, PVT, Flanker. Konfigurasi penuh di
[`config/features.json`](config/features.json).

## 4. Output

```
features/
├── COG-BCI_workload_binary.npz   STEW_workload_binary.npz   combined_workload_binary.npz
├── COG-BCI_nback_graded.npz      COG-BCI_matb_graded.npz
└── feature_manifest.json
```

Tiap `.npz` berisi: `X_psd` (n×70), `X_cov` (n×14×14), `X_tangent` (n×105),
`y` (label), `groups` (`<dataset>-<subjek>` untuk GroupKFold cross-subject),
`tasks`, `subjects`. File `combined_*` menumpuk kedua dataset untuk pelatihan transfer.

## 5. Cara jalan

```bash
cd 03_Features
../.venv/bin/python scripts/extract_features.py --config config/features.json
# subset: --datasets COG-BCI
```

## 6. Validasi (COG-BCI 29 subjek + STEW 48 subjek)

**Transfer-compatibility** — dimensi fitur identik lintas-dataset: PSD 70 = 70,
tangent 105 = 105, cov (14,14) = (14,14). Tanpa NaN/inf. ✅

Jumlah epoch: COG-BCI 138.900 (29 subj), STEW 9.491 (47 subj), combined 148.391 (76 grup).

**Kualitas fitur** (cross-subject `GroupKFold` 5-fold, anti-leakage, classifier linear,
subsample ≤12k/uji, seed 42):

| Uji | PSD+LogReg | Riemann+LogReg | Chance |
|-----|-----------|----------------|--------|
| STEW workload biner (47 subj) | **0,734** (F1 0,73) | 0,670 | 0,55 |
| COG-BCI N-back 3-kelas (29 subj) | 0,482 | **0,515** | 0,34 |
| COG-BCI MATB 3-kelas (29 subj) | 0,517 | **0,604** | 0,35 |
| COG-BCI workload biner (29 subj) | 0,597 (F1 0,56) | 0,621 | 0,73* |

Decoding within-COG-BCI **menguat jelas** dari pilot 3-subjek (N-back Riemann 0,38 → 0,52;
MATB 0,60) — membuktikan fitur diskriminatif lintas-subjek. **Riemannian > PSD** untuk
task graded (sesuai literatur). Angka ini = baseline linear cross-subject; target proposal
85% ditargetkan via arsitektur agentic/deep (Tahap 04).

> *Workload biner COG-BCI sangat imbalans (36k rendah vs 103k tinggi) → akurasi mentah
> menyesatkan (baseline mayoritas 0,73). Dengan `class_weight='balanced'` model mengorbankan
> akurasi demi recall minoritas; **F1-macro 0,56** menunjukkan diskriminasi nyata. Untuk
> within-COG-BCI, target **graded (N-back/MATB) lebih bersih** daripada biner.

## 7. Catatan untuk Tahap 04–05

- Imbalans `COG-BCI_workload_binary` (≈3.9k rendah vs 10.9k tinggi) → pakai
  `class_weight='balanced'` / stratifikasi.
- Decoding wajib `GroupKFold`/`LeaveOneGroupOut` per subjek (cegah bocor identitas).
- Untuk klaim transfer (+40% vs single-dataset): latih di `combined_*`, uji
  per-dataset (atau leave-one-dataset-out), bandingkan dengan single-dataset.
- Fitur PSD interpretable (theta frontal ↑, alpha parietal ↓ = penanda beban kognitif)
  → cocok untuk agen spesialis "spektral" di arsitektur agentic (Tahap 04).

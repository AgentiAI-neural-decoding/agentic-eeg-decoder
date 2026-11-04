# Dataset & Provenance — Agentic EEG Decoder

Dokumen ini melacak **asal-usul lengkap data** yang dipakai penelitian ini: dari sumber
mentah publik, melalui preprocessing, hingga tensor fitur siap-decoding. Setiap parameter
di bawah dirujuk langsung ke config/skrip pipeline (bukan ringkasan naratif), sehingga
dapat direproduksi dan diaudit (syarat reproducibility DIKTI 2026).

> **Ringkas alur**: `raw (publik)` → **Tahap 01** unduh+verifikasi → **Tahap 02** preprocessing+ICA
> → `derivatives/*.fif` → **Tahap 03** harmonisasi+ekstraksi fitur → `features/*.npz` →
> Tahap 04–06 decoding. Data mentah & turunan **tidak** ada di git (`.gitignore`); lihat
> [§5 Distribusi data](#5-distribusi-data-git--google-drive).

---

## 1. Sumber data mentah (raw)

Dua dataset EEG publik. **Tidak diakuisisi sendiri** — keduanya *secondary public dataset*
yang sudah lulus IRB di institusi penyedia (lihat [`01_Akuisisi_Data/README.md`](01_Akuisisi_Data/README.md) §5).

| | **COG-BCI** | **STEW** |
|---|---|---|
| Sumber | [Zenodo records/6874129](https://zenodo.org/records/6874129) | [IEEE DataPort](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset) |
| DOI dataset | [10.5281/zenodo.6874128](https://doi.org/10.5281/zenodo.6874128) | [10.21227/44r8-ya50](https://doi.org/10.21227/44r8-ya50) |
| Publikasi | Hinss et al., *Sci Data* 2023 — [10.1038/s41597-022-01898-y](https://doi.org/10.1038/s41597-022-01898-y) | Lim et al., *IEEE TNSRE* 2018 |
| Lisensi | CC-BY-4.0 (redistribusi+turunan boleh, **wajib sitasi**) | Open Access IEEE (wajib sitasi, **butuh login** unduh) |
| Subjek | 29 (folder BIDS `sub-01..29`) | 48 (`sub01..48`) |
| Akuisisi | EEG riset, 32 kanal, montase 10-20 | Emotiv EPOC consumer, **14 kanal**, 128 Hz |
| Tugas | MATB, N-Back, PVT, Flanker + resting (RSEC/RSEO), **3 sesi** | resting (`lo`) + SIMKAP multitasking (`hi`) |
| Format | EEGLAB `.set`+`.fdt`, struktur BIDS | plain `.txt` 14-kolom + `ratings.csv` |
| Ukuran | ~31,5 GB (29 × ~1,1 GB) | ~52 MB |
| Total rekaman | **1044** (29 subj × 3 sesi × 12 task/kondisi) | **96** (48 `lo` + 48 `hi`) |

**Persetujuan etis sumber**: COG-BCI — Université de Toulouse (CER 2021-342). Keduanya
*de-identified* → tidak diperlukan ethical clearance lokal untuk analisis sekunder.

### Cara akuisisi (Tahap 01)
- **COG-BCI** — otomatis: [`download_cogbci.py`](01_Akuisisi_Data/scripts/download_cogbci.py)
  mengunduh tiap berkas (resume HTTP Range) + **verifikasi MD5** terhadap
  [`cogbci_manifest.json`](01_Akuisisi_Data/manifest/cogbci_manifest.json) (32/32 berkas cocok).
- **STEW** — manual (IEEE butuh login): unduh `STEW Dataset.zip` lalu
  [`verify_stew.py`](01_Akuisisi_Data/scripts/verify_stew.py) mencatat MD5 + ekstrak + validasi struktur.

### Endpoint & integritas (untuk reproduksibilitas byte-exact)

URL di tabel atas adalah **halaman manusia**. Yang sebenarnya dipakai skrip & dipakai untuk
verifikasi integritas:

| | COG-BCI | STEW |
|---|---|---|
| Endpoint unduh aktual | **REST API Zenodo**: `https://zenodo.org/api/records/6874129/files/{name}/content` ([`download_cogbci.py:26`](01_Akuisisi_Data/scripts/download_cogbci.py#L26)) | Manual via halaman IEEE (**login wajib** → tak bisa di-script) |
| Integritas | **MD5 per file** dicek otomatis vs manifest; `content-length` server = `size_bytes` manifest | **MD5 zip** dihitung [`verify_stew.py`](01_Akuisisi_Data/scripts/verify_stew.py) vs manifest |
| Nilai acuan | per-file di [`cogbci_manifest.json`](01_Akuisisi_Data/manifest/cogbci_manifest.json) (mis. `sub-01.zip` = 1.096.630.047 byte) | `STEW Dataset.zip`: **md5 `987330cd36004f1669a2e4dba4f865ba`**, **54.811.358 byte** |
| Status reproduksi | ✅ otomatis & byte-exact | ⚠️ unduh **manual** (login-gated); integritas dijamin via MD5 pasca-unduh |

> **Catatan DOI**: manifest COG-BCI memakai *concept DOI* `10.5281/zenodo.6874128` (selalu
> menunjuk versi terbaru), sedangkan *version record* spesifik = `6874129`. Keduanya konsisten.
>
> **Verifikasi cepat** (tanpa unduh ulang) — cek file STEW yang sudah ada:
> ```bash
> md5sum "STEW Dataset.zip"   # harus 987330cd36004f1669a2e4dba4f865ba
> ```

---

## 2. Preprocessing & ICA (Tahap 02)

Input: `01_Akuisisi_Data/raw/`. Output: epoch bersih `*_proc-clean_epo.fif` +
`*_ica-report.json` + plot PSD QC di [`02_Preprocessing/derivatives/`](02_Preprocessing/).
Pipeline **config-driven**, **deterministik** (`random_state=42`), **non-destruktif**
(mengikuti PREP + MNE community; lihat [`02_Preprocessing/README.md`](02_Preprocessing/README.md)).

Parameter aktual (dari [`config/cogbci.json`](02_Preprocessing/config/cogbci.json) &
[`config/stew.json`](02_Preprocessing/config/stew.json)):

| Langkah | COG-BCI (32 ch) | STEW (14 ch) |
|---|---|---|
| Loading | EEGLAB `.set`/`.fdt` (BIDS) | `.txt` 14-kolom |
| Montase | `standard_1020` | `standard_1020` |
| Notch | 50, 100, 150 Hz | 50 Hz |
| Bandpass | 0,5–45 Hz | 0,5–45 Hz |
| Resample | **250 Hz** | tidak (sudah 128 Hz) |
| Bad channel | LOF (thr 1,5) + interpolasi | robust-z 3,5 / flat 1 µV / maks 30% |
| Re-reference | average | average |
| ICA | extended infomax, 99% varian, highpass-fit 1 Hz | extended infomax, n=13, highpass-fit 1 Hz |
| Rejeksi IC | **ICLabel** ≥ 0,7 (muscle, eye-blink, heart, line-noise, channel-noise) | heuristik korelasi frontal (AF3/AF4/F7/F8) ≥ 0,6 sebagai proksi EOG |
| Epoching | fixed 2 s (baseline none) | sliding 2 s, overlap 1 s |
| Reject amplitudo | 150 µV | 200 µV (retensi min 30%) |

> **Kenapa beda?** STEW hanya 14 kanal → ICLabel tak handal di bawah ~20 kanal, jadi dipakai
> heuristik korelasi-frontal; bad-channel detection LOF juga dilewati (terlalu sedikit kanal).
> COG-BCI di-resample 250 Hz di sini, lalu **diturunkan ke 128 Hz** saat ekstraksi fitur (Tahap 03)
> demi harmonisasi dengan native STEW.

Output `.fif` per rekaman bernama mis.
`sub-01_ses-S1_task-MATB_run-1_proc-clean_epo.fif` (COG-BCI) /
`sub-01_task-hi_proc-clean_epo.fif` (STEW). QC: PSD pre/post + `ica-report.json`
(rasio komponen tertolak wajar ~10–40%).

---

## 3. Ekstraksi fitur & harmonisasi (Tahap 03)

Input: `derivatives/*.fif`. Output: [`03_Features/features/*.npz`](03_Features/) +
`feature_manifest.json`. Sumber: [`extract_features.py`](03_Features/scripts/extract_features.py),
[`config/features.json`](03_Features/config/features.json).

**Tidak ada langkah *feature selection* statistik** (mis. mutual-info / RFE). Yang ada adalah
**harmonisasi** — reduksi ke ruang fitur bersama agar fitur **sebanding lintas-dataset**
(prasyarat transfer learning). "Pemilihan" terjadi pada level kanal & pita, bukan seleksi data-driven:

### 3.1 Harmonisasi (`harmonize()`)
1. **Pilih 14 kanal Emotiv bersama** (irisan COG-BCI 32-ch ∩ STEW 14-ch), urutan konsisten:
   `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`.
2. **Resample → 128 Hz** (native STEW; "harmonisasi sampling rate" proposal).
   → kedua dataset masuk ruang fitur berdimensi identik = "kompatibilitas data 100%".

### 3.2 Fitur yang diekstrak (per epoch)
| Fitur | Dimensi | Metode | Catatan leakage |
|---|---|---|---|
| **PSD band-power** | **70** (5 pita × 14 ch) | Welch (`n_fft`=2 s), **relative + log** | per-epoch → bebas leakage, siap-pakai |
| **Kovarians spasial** | **14×14** | estimator **OAS** (skala µV) | disimpan mentah → CSP/tangent **di-fit dalam CV** |
| **Tangent-space global** | **105** (14·15/2) | referensi Riemannian di SELURUH epoch | ⚠️ **eksploratif saja** (referensi global = leakage) |

Pita frekuensi: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz.

> ⚠️ **Anti-leakage (kritis)**: hanya `X_psd` dan `X_cov` yang sah untuk evaluasi. CSP &
> Riemannian tangent **harus** di-fit di dalam tiap fold CV dari `X_cov`. `X_tangent` global
> hanya untuk inspeksi cepat — jangan dipakai untuk klaim akurasi.

### 3.3 Skema label (`labels.py` + config)
| Skema | Dataset | Kelas | Peran |
|---|---|---|---|
| `workload_binary` | COG-BCI + STEW | rendah(0)/tinggi(1) | **target PRIMER transfer** |
| `nback_graded` | COG-BCI | 0/1/2-back | sekunder (memori kerja) |
| `matb_graded` | COG-BCI | easy/med/diff | sekunder (beban MATB) |

Pemetaan `workload_binary` COG-BCI — **rendah**: resting (RS_*) + `MATBeasy` + `zeroBACK`;
**tinggi**: `MATBmed/diff`, `one/twoBACK`, `PVT`, `Flanker`. STEW: `lo`=0, `hi`=1.

### 3.4 Output `.npz`
```
features/
├── COG-BCI_workload_binary.npz   STEW_workload_binary.npz   combined_workload_binary.npz
├── COG-BCI_nback_graded.npz      COG-BCI_matb_graded.npz
└── feature_manifest.json
```
Tiap `.npz`: `X_psd` (n×70), `X_cov` (n×14×14), `X_tangent` (n×105), `y`, `groups`
(`<dataset>-<subjek>` untuk GroupKFold cross-subject), `tasks`, `subjects`.
File `combined_*` menumpuk kedua dataset untuk pelatihan transfer.

**Jumlah epoch (hasil pipeline)**: COG-BCI ≈ 138.900 (29 subj), STEW ≈ 9.491 (47 subj),
combined ≈ 148.391 (76 grup subjek).

---

## 4. Ringkasan provenance (satu pandangan)

| Artefak | Dihasilkan oleh | Lokasi (lokal) | Dipakai di |
|---|---|---|---|
| `raw/COG-BCI/`, `raw/STEW/` | Tahap 01 (unduh) | `01_Akuisisi_Data/raw/` | input Tahap 02 |
| `derivatives/**/*.fif` | Tahap 02 (preprocess) | `02_Preprocessing/derivatives/` | input Tahap 03 & 06 |
| `features/*.npz` | Tahap 03 (extract) | `03_Features/features/` | Tahap 04 & 05 |

---

## 5. Distribusi data (git & Google Drive)

Data **tidak di-commit** ke git (lisensi CC-BY/IEEE melarang redistribusi publik; ukuran besar).
Untuk training di Colab (lihat [`README_COLAB.md`](README_COLAB.md)), data dipindah lewat
**Google Drive privat** `MyDrive/agentic-eeg/`:

| File Drive | = artefak | Cara membuat |
|---|---|---|
| `features/*.npz` (~393 MB) | output Tahap 03 | copy langsung `03_Features/features/*.npz` |
| `derivatives.zip` (~14 GB) | output Tahap 02 (di-zip) | `cd 02_Preprocessing && zip -r -q ~/derivatives.zip derivatives/` |
| `checkpoints/` | hasil eksperimen | dibuat otomatis oleh notebook |

> **Kenapa `derivatives` di-zip, `features` tidak**: derivatives berisi ~1.100 file `.fif`
> kecil → Google Drive sync per-file lambat & rawan gagal; satu `.zip` sync andal, lalu
> notebook meng-unzip ke disk lokal Colab. `features/` hanya 5 file → cukup di-copy/symlink.

---

## 6. Sitasi wajib

Bila memakai data ini, **wajib** mensitasi sumber asli:

> Hinss, M. F., Somon, B., Jahanpour, E. S., Lotte, F., & Roy, R. N. (2023). *Open
> multi-session and multi-task EEG cognitive Dataset for passive brain-computer Interface
> Applications.* Scientific Data. doi:10.1038/s41597-022-01898-y

> Lim, W. L., Sourina, O., & Wang, L. P. (2018). *STEW: Simultaneous Task EEG Workload
> Dataset.* IEEE Dataport. doi:10.21227/44r8-ya50

Untuk kode/pipeline repo ini, lihat [`CITATION.cff`](CITATION.cff).

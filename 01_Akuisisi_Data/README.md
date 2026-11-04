# 01 — Akuisisi Dataset (COG-BCI & STEW)

Langkah pertama pelaksanaan penelitian **Neural Decoding Berbasis Agentic AI** (DIKTI 2026,
Sugeng Rifqi Mubaroq, Akademi Digital Bandung). Dokumen ini berisi prosedur resmi untuk
mengakuisisi dua dataset publik yang menjadi sumber sinyal EEG:

| Dataset | Peran dalam penelitian | Ukuran | Lisensi | Login |
|---------|------------------------|--------|---------|-------|
| **COG-BCI** (Hinss et al., 2023) | Beban kognitif multi-sesi, multi-task (MATB, N-Back, PVT, Flanker) — input utama untuk arsitektur agentic neural decoder | ~31,5 GB (29 subjek × ~1,1 GB) | CC-BY-4.0 | Tidak |
| **STEW** (Lim et al., 2018) | Validasi silang beban kerja mental (SIMKAP multitasking, Emotiv EPOC 14 ch) | ~52 MB | Open Access IEEE DataPort | Ya (akun gratis IEEE) |

---

## 0. Pra-syarat

1. **Ruang disk**: minimal **35 GB** kosong di lokasi target (`--out`). Drive utama Anda
   saat ini hanya menyisakan ~768 MiB — **gunakan disk eksternal atau server lab**.
2. **Python ≥ 3.10** (modul `urllib`, `hashlib`, `zipfile` cukup; tidak ada dependensi pip).
3. **Koneksi stabil**: COG-BCI berukuran 31 GB → estimasi 1–5 jam tergantung bandwidth.
   Skrip mendukung *resume* otomatis (HTTP Range) — aman bila terputus.
4. **Akun IEEE** (gratis) untuk STEW: daftar di <https://ieee-dataport.org/user/register>.

---

## 1. Struktur folder

```
01_Akuisisi_Data/
├── README.md                  ← dokumen ini
├── manifest/
│   ├── cogbci_manifest.json   ← daftar 32 berkas Zenodo + MD5 + sitasi
│   └── stew_manifest.json     ← metadata STEW (MD5 diisi setelah unduh)
├── scripts/
│   ├── download_cogbci.py     ← pengunduh otomatis dengan verifikasi MD5
│   └── verify_stew.py         ← ekstraksi + validasi struktur STEW
└── raw/
    ├── COG-BCI/               ← TARGET unduh COG-BCI (atau symlink ke disk eksternal)
    └── STEW/                  ← TARGET ekstraksi STEW
```

> **Tips disk eksternal**: bila Anda ingin menyimpan data mentah di drive lain (misalnya
> `/Volumes/EEG_Data`), buat symlink:
> ```bash
> rm -rf raw/COG-BCI raw/STEW
> ln -s "/Volumes/EEG_Data/COG-BCI" raw/COG-BCI
> ln -s "/Volumes/EEG_Data/STEW"    raw/STEW
> ```

---

## 2. Akuisisi COG-BCI (otomatis)

**Sumber:** <https://zenodo.org/records/6874129> · DOI [10.5281/zenodo.6874128](https://doi.org/10.5281/zenodo.6874128)

### 2.1. Uji coba — hanya metadata (≈ 235 KB)

```bash
cd "01_Akuisisi_Data"
python scripts/download_cogbci.py --out raw/COG-BCI --metadata-only
```

Akan mengunduh `COG-BCI_info.pdf`, `notebook.mat`, `triggerlist.txt`. Pakai ini dulu
untuk memvalidasi koneksi & menulis ringkasan protokol akuisisi sebelum unduh penuh.

### 2.2. Subset awal untuk pilot (3 subjek, ~3,3 GB)

```bash
python scripts/download_cogbci.py --out raw/COG-BCI --subjects 01 02 03
```

Disarankan dilakukan terlebih dahulu untuk uji pipeline preprocessing (Tahun-1, bulan 1–3
di peta jalan penelitian).

### 2.3. Unduh penuh (~31,5 GB)

```bash
python scripts/download_cogbci.py --out raw/COG-BCI
```

Skrip akan:
- mengunduh tiap berkas dengan resume otomatis bila terputus,
- memverifikasi MD5 terhadap manifest,
- melewati berkas yang sudah lengkap & valid.

### 2.4. Ekstraksi

```bash
cd raw/COG-BCI
for z in sub-*.zip; do unzip -n "$z"; done
```

Struktur akhir sesuai standar **BIDS**: `sub-XX/ses-S{1,2,3}/eeg/*.set + *.fdt`.

---

## 3. Akuisisi STEW (manual + skrip verifikasi)

**Sumber:** <https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset> · DOI [10.21227/44r8-ya50](https://doi.org/10.21227/44r8-ya50)

> IEEE DataPort tidak menyediakan API anonim → langkah unduh harus manual.

### 3.1. Unduh manual

1. Login ke <https://ieee-dataport.org> (daftar gratis bila belum punya akun).
2. Buka halaman dataset → klik **"STEW Dataset.zip"** (52,27 MB).
3. Simpan ke `~/Downloads/STEW Dataset.zip` (atau lokasi lain).

### 3.2. Verifikasi & ekstraksi

```bash
python scripts/verify_stew.py \
    --zip "$HOME/Downloads/STEW Dataset.zip" \
    --out raw/STEW
```

Skrip akan:
- menghitung & mencatat MD5 (otomatis diisikan ke `manifest/stew_manifest.json`),
- mengekstrak ZIP,
- memvalidasi keberadaan ratings.csv + ≥ 40 pasang `subNN_lo.txt` / `subNN_hi.txt`.

---

## 4. Validasi pasca-akuisisi

Setelah kedua dataset terunduh, jalankan validasi cepat:

```bash
# Hitung jumlah subjek COG-BCI yang valid (folder BIDS sub-XX)
ls raw/COG-BCI | grep -E '^sub-[0-9]{2}$' | wc -l   # harus 29

# Hitung file STEW
ls raw/STEW/*.txt | wc -l                            # ~96 (48 lo + 48 hi)
```

Dokumentasikan timestamp + checksum di catatan logbook penelitian (Notion/Obsidian/Excel
Anda — sesuai kebijakan reproducibility DIKTI).

---

## 5. Kepatuhan etika & lisensi

- **COG-BCI** — CC-BY-4.0: redistribusi & turunan diperbolehkan, **wajib mensitasi**:
  > Hinss, M. F., Somon, B., Jahanpour, E. S., Lotte, F., & Roy, R. N. (2023).
  > *Open multi-session and multi-task EEG cognitive Dataset for passive brain-computer Interface Applications.* Scientific Data. doi:10.1038/s41597-022-01898-y
  >
  > Persetujuan etis: Université de Toulouse (CER 2021-342).

- **STEW** — Open Access IEEE DataPort, **wajib mensitasi**:
  > Lim, W. L., Sourina, O., & Wang, L. P. (2018). *STEW: Simultaneous Task EEG Workload Dataset.* IEEE Dataport. doi:10.21227/44r8-ya50

Kedua dataset bersifat publik & telah lulus IRB di institusi penyedia → tidak diperlukan
ethical clearance lokal untuk *secondary analysis*. Tetap catat di proposal/BAB Metode
bahwa data adalah *de-identified public dataset*.

---

## 6. Output langkah ini (deliverable Tahun-1, M1)

- [x] `raw/COG-BCI/sub-01..29/` (struktur BIDS) — verified via MD5 _(29 subjek × 3 sesi × 12 task = 1044 rekaman, 2026-05-31)_
- [x] `raw/STEW/sub*_lo.txt`, `sub*_hi.txt`, `ratings.txt` — extracted & validated _(48 lo + 48 hi + ratings)_
- [x] `manifest/cogbci_manifest.json` & `manifest/stew_manifest.json` final _(cogbci dikoreksi vs Zenodo: 32/32 cocok; stew MD5 tercatat)_
- [x] Catatan logbook akuisisi (tanggal, durasi, sumber, checksum, ukuran) — lihat [`LOGBOOK.md`](LOGBOOK.md)

Lanjutkan ke **02_Preprocessing** (ICA artifact removal, bandpass 0.5–45 Hz, epoching)
setelah deliverable di atas tercentang.

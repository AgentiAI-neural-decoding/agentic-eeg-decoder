# Menjalankan Training di Google Colab (GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AgentiAI-neural-decoding/agentic-eeg-decoder/blob/main/colab_training.ipynb)

Panduan offload training ke Colab (GPU CUDA) — menghindari beban termal MPS lokal
(suhu Mac sempat 98°C). CUDA juga lebih cepat untuk fine-tuning foundation model &
membuka kernel Mamba (`mamba-ssm`) bila diperlukan.

> Repo ini publik: badge di atas langsung membuka notebook di Colab. Langkah PAT di bawah
> hanya diperlukan bila Anda ingin *push* hasil kembali ke fork Anda sendiri.

## Dua notebook (urutan kerja)

| # | Notebook | Peran | Output → input berikutnya |
|---|----------|-------|---------------------------|
| 1 | [`colab_harmonisasi.ipynb`](colab_harmonisasi.ipynb) | **Hulu** — Tahap 01→03: unduh COG-BCI (Zenodo) + ambil STEW dari Drive → preprocessing+ICA → ekstraksi fitur | menghasilkan `features/*.npz` + `derivatives.zip` di Drive |
| 2 | [`colab_training.ipynb`](colab_training.ipynb) | **Hilir** — Tahap 05–06: eksperimen multi-task & foundation-model probing | memakai artefak dari notebook #1 |

> **Jalankan #1 dulu** untuk menyiapkan data, baru #2. Notebook #1 **tahan-banting** (checkpoint
> ke Drive tiap batch + resume otomatis) sehingga aman bila sesi Colab putus — buka ulang & Run all.
> Detail asal data: [`README_DATASET.md`](README_DATASET.md).

## Mengapa Colab
- **Termal & kecepatan**: GPU T4/A100 Colab >> MPS untuk training berat; Mac tidak panas.
- **Reproducible**: notebook + repo = lingkungan tertata.

## Langkah sekali-saja

### 1. (Opsional) Buat Personal Access Token (PAT) GitHub
GitHub → **Settings → Developer settings → Personal access tokens**:
- **Fine-grained** (disarankan): pilih repo `AgentiAI-neural-decoding/agentic-eeg-decoder`,
  permission **Contents: Read** (Read-Write bila ingin push hasil dari Colab).
- Atau **Classic**: centang **hanya scope `repo`** ("Full control of private repositories").
  Pada token classic akses repo privat bersifat all-or-nothing, jadi `repo` = scope minimum
  (sudah cukup untuk clone + push). Centang `repo` otomatis mencentang anak-anaknya — normal.
  **Semua scope lain biarkan kosong** (`workflow`, `admin:*`, `delete_repo`, `user`, dll.).
- Beri **Expiration** (mis. 30–90 hari, jangan "No expiration") + **Note** deskriptif
  (mis. `colab-agentic-eeg`) agar mudah dicabut.
- Salin token; dimasukkan di notebook lewat `getpass` (TIDAK tersimpan di notebook).

### 2. Siapkan struktur folder di Google Drive
Buat `MyDrive/agentic-eeg/` (100 GB Anda lebih dari cukup):

```
MyDrive/agentic-eeg/
├── features/                 # untuk eksperimen Tahap 04/05 (~393 MB, 5 file .npz)
│   └── *.npz
├── derivatives.zip           # untuk foundation model Tahap 06 (~14 GB, arsip epoch .fif)
└── checkpoints/              # output (dibuat otomatis)
```

**Penting — zip dulu, jangan upload loose file**: derivatives berisi ~1.100 file `.fif`
kecil. Mengunggahnya sebagai folder membuat Google Drive sync per-file (lambat & rawan
gagal). Pakai **satu `derivatives.zip`** → sync sebagai satu file (cepat & andal); notebook
meng-unzip ke disk lokal Colab. Notebook ini memakai struktur tsb.

**Upload data dari Mac**: folder ter-sync Google Drive desktop (`~/Library/CloudStorage/
GoogleDrive-<akun>/My Drive/agentic-eeg/`). `features/` (5 .npz) + `derivatives.zip` cukup
di-copy ke folder itu; app Drive mengunggah ke cloud otomatis.

> Data mentah & turunan TIDAK ada di git (di-`.gitignore`) — hanya via Drive. Lisensi
> dataset (CC-BY/IEEE) melarang redistribusi publik, jadi Drive privat = tepat.

## Menjalankan
1. Buka [`colab_training.ipynb`](colab_training.ipynb) di Colab
   (GitHub → buka file → "Open in Colab", atau upload manual).
2. **Runtime → Change runtime type → GPU**.
3. Jalankan sel berurutan: cek GPU → mount Drive → clone (masukkan PAT) → install →
   link data → eksperimen.
4. Hasil JSON otomatis disalin ke `Drive/agentic-eeg/checkpoints/`.

## Saklar device
Skrip auto-deteksi: **CUDA** (Colab) > **MPS** (Mac) > **CPU**. Tak perlu ubah kode
berpindah mesin (`05_Experiments/scripts/multitask.py:pick_device`,
`06_FoundationModel/scripts/fm_encoder.py:resolve_device`).

## Wiring CBraMod (foundation model)
Lihat [`06_FoundationModel/README.md`](06_FoundationModel/README.md) §3. `CBraModEncoder`
**sudah terimplementasi** via [braindecode](https://braindecode.org)
(`braindecode/cbramod-pretrained`) — tak perlu menulis kode lagi. Di Colab cukup:
`!pip install braindecode`, lalu set `encoder="cbramod"` di
`06_FoundationModel/config/fm.json` (opsi cbramod sudah diisi: `hf_repo`, `sfreq=200`,
`n_times=400`, `scale=100`). Bobot ter-unduh otomatis dari HuggingFace saat pertama jalan.

## Tips sesi Colab
- Sesi gratis bisa putus (~12 jam / idle) → notebook idempotent, jalankan ulang dari atas.
- Untuk training panjang: simpan checkpoint model ke `Drive/.../checkpoints/` berkala.
- Colab Pro memberi GPU lebih kuat (A100/L4) & sesi lebih panjang bila diperlukan.

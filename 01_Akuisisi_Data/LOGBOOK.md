# Logbook Akuisisi Data — Tahap 01

Catatan kronologis akuisisi dataset publik untuk penelitian *Neural Decoding Berbasis
Agentic AI* (DIKTI Penelitian Fundamental 2026). Memenuhi kebijakan reproducibility:
sumber, timestamp, ukuran, dan checksum tiap berkas dicatat.

---

## COG-BCI (Zenodo 6874129 · Hinss et al., 2023 · CC-BY-4.0)

| Tanggal | Aksi | Cakupan | Hasil |
|---------|------|---------|-------|
| 2026-05-12 | Unduh metadata (`--metadata-only`) | `COG-BCI_info.pdf`, `notebook.mat`, `triggerlist.txt` | MD5 cocok |
| 2026-05-31 | Unduh pilot (`--subjects 01 02 03`) | sub-01, sub-02, sub-03 (~3,27 GB) | **MD5 cocok (3/3)** |
| 2026-05-31 | Ekstraksi BIDS (`unzip -n`) | 3 subjek × 3 sesi × 12 rekaman = 108 | Struktur BIDS valid |
| 2026-05-31 | Unduh penuh 29 subjek (`download_cogbci.py`) | sub-01..29 (~31,5 GB) | **29/29 MD5 cocok** (setelah koreksi manifest + retry timeout) |
| 2026-05-31 | Ekstraksi penuh + validasi struktur | 29 subjek × 3 sesi × 12 task | **1044 rekaman, .set/.fdt lengkap ✅** |

> **SELESAI** — Tahap 01 akuisisi COG-BCI tuntas & tervalidasi. Catatan: sub-16 & sub-27
> sempat timeout (diselesaikan via resume), sub-11 & sub-26 false-alarm akibat typo manifest
> (lihat "Koreksi manifest" di bawah).

### Checksum pilot (terverifikasi terhadap manifest)

| Berkas | Ukuran (byte) | MD5 |
|--------|--------------:|-----|
| sub-01.zip | 1.096.630.047 | becd2fb862fff84973a1d136fcb03262 |
| sub-02.zip | 1.077.564.602 | 55aa94b478d2227dd1edd924929b60b… |
| sub-03.zip | 1.091.919.452 | (cocok manifest) |

> Verifikasi MD5 dilakukan otomatis oleh `scripts/download_cogbci.py` terhadap
> `manifest/cogbci_manifest.json`. Log lengkap: `/tmp/cogbci_pilot.log`, `/tmp/cogbci_full.log`.

### Koreksi manifest (2026-05-31)

Unduhan penuh awal melaporkan 4 "kegagalan": sub-11, sub-16, sub-26, sub-27.
Investigasi terhadap **checksum otoritatif Zenodo API** (`/api/records/6874129`):

- **sub-16, sub-27** — kegagalan asli (read timeout); diunduh ulang dengan resume → MD5 cocok.
- **sub-11, sub-26** — BUKAN korup. MD5 di `cogbci_manifest.json` salah ketik 1 karakter:
  - sub-11: `…3bf19a`**c**`5` → seharusnya `…3bf19a`**e**`5`
  - sub-26: `…017`**40**`2dd1` → seharusnya `…017`**49**`2dd1`

  Manifest dikoreksi terhadap Zenodo (kini **32/32 entri cocok**). File yang terunduh
  sejak awal sebenarnya valid.

### Struktur per sesi (terkonfirmasi pada sub-01)

12 rekaman EEGLAB (`.set` + `.fdt`) per sesi:
- **MATB** — MATBeasy, MATBmed, MATBdiff
- **N-Back** — zeroBACK, oneBACK, twoBACK
- **PVT**, **Flanker**
- **Resting state** — RS_Beg_EC, RS_Beg_EO, RS_End_Ec, RS_End_EO

Tiap subjek juga berisi folder `behavioral/` dan `chanlocs/` per sesi.

---

## STEW (IEEE DataPort 44r8-ya50 · Lim et al., 2018 · Open Access)

| Tanggal | Aksi | Hasil |
|---------|------|-------|
| 2026-05-31 | Unduh manual (login IEEE) oleh user → `raw/STEW/STEW Dataset.zip` | 54.811.358 byte |
| 2026-05-31 | Verifikasi (`verify_stew.py --keep-zip`) | MD5 tercatat di manifest |
| 2026-05-31 | Bereskan duplikasi (hapus subfolder `STEW Dataset/`, pertahankan flat + zip) | Struktur flat kanonik |

### Checksum & isi STEW

| Berkas | Ukuran (byte) | MD5 |
|--------|--------------:|-----|
| STEW Dataset.zip | 54.811.358 | 987330cd36004f1669a2e4dba4f865ba |

Isi tervalidasi: **48 `subNN_lo.txt`** (resting) + **48 `subNN_hi.txt`** (SIMKAP) +
`ratings.txt`, subjek 1–48. Layout flat di `raw/STEW/` (sesuai `raw_root` config
preprocessing & validasi README §4). Format: teks numerik, 14 kanal Emotiv EPOC, 128 Hz.

> Catatan: zip berisi semua file di dalam folder `STEW Dataset/`. User sudah mengekstrak
> flat sebelumnya; `verify_stew.py` mengekstrak ulang ke subfolder → duplikat dihapus
> setelah konfirmasi MD5 identik. Zip dipertahankan (`--keep-zip`) untuk provenance.

---

## Lingkungan akuisisi

- **Mesin**: macOS (Darwin 25.6.0), Python 3.14.5
- **Disk**: ~117 GiB tersedia pada saat unduh penuh (cukup untuk 31,5 GB)
- **Metode**: HTTP Range resume + verifikasi MD5 per berkas (lihat skrip)

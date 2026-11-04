# Agentic EEG Decoder

<img width="1336" height="535" alt="image" src="https://github.com/user-attachments/assets/4fe48c82-989a-4757-a6b8-085c9929c454" />


**Neural Decoding Berbasis Agentic AI untuk Analisis Dinamika Kolaborasi Tim Lintas Disiplin**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AgentiAI-neural-decoding/agentic-eeg-decoder/blob/main/colab_training.ipynb)

Implementasi pipeline penelitian DIKTI Penelitian Fundamental 2026 — orkestrasi multi-agent
untuk klasifikasi beban kognitif dari sinyal EEG, divalidasi pada dua dataset publik
(**COG-BCI** dan **STEW**) sebelum disesuaikan ke konteks kolaborasi tim lintas disiplin.

## Paper

This repository contains the analysis code for:

> Mubaroq, S. R., Awangga, R. M., Pragama, T. D., Fathummubin, S., & Abdulhaq, A. Y. (2026).
> A Multi-Specialist Stacking Decoder of Cognitive Workload and a Decomposition of the Limits
> of Cross-Dataset Transfer in Electroencephalography. *Intelligent Automation & Soft Computing*,
> 41(1), 27–46. https://doi.org/10.32604/iasc.2026.088039

Pipeline stages map to the paper as follows: data acquisition (`01_Akuisisi_Data/`),
preprocessing and ICA/ICLabel (`02_Preprocessing/`, config `config/cogbci.json`,
`config/stew.json`), feature extraction and label schemes (`03_Features/`, config
`config/features.json`), specialists and stacking meta-learner (`04_Agentic/`), experiments
and transfer analyses (`05_Experiments/`), foundation-model probing (`06_FoundationModel/`),
and domain-adversarial harmonization (`07_Harmonization/`). Raw EEG data are not
redistributed; download them from the original sources listed below.

Documentation inside the stage folders is written in Indonesian.

---

> Status: ✅ Tahap 01–06 tuntas (akuisisi → preprocessing → fitur → agentic → eksperimen →
> foundation model). Within-dataset decoding kuat; transfer lintas-dataset = tantangan
> terbuka terkarakterisasi (lihat [`05_Experiments/`](05_Experiments/)). Training berat
> via Colab GPU — lihat [`README_COLAB.md`](README_COLAB.md).

---

## Roadmap penelitian (2 tahun)

| Tahap | Fokus | Deliverable | Repo path |
|------:|-------|-------------|-----------|
| 01 | Akuisisi dataset publik (COG-BCI, STEW) | raw + manifest + checksum | [`01_Akuisisi_Data/`](01_Akuisisi_Data/) |
| 02 | Preprocessing + ICA (MNE-Python, ICLabel) | epoch `.fif` bersih + log ICA + QC | [`02_Preprocessing/`](02_Preprocessing/) |
| 03 | Ekstraksi fitur (PSD/CSP/Riemannian) — transfer-compatible | tensor siap-decoding | [`03_Features/`](03_Features/) |
| 04 | Arsitektur Agentic AI (orchestrator + spesialis) | model + bobot | [`04_Agentic/`](04_Agentic/) |
| 05 | Eksperimen offline, ablation & transfer lintas-dataset | hasil + tinjauan SOTA bersitasi | [`05_Experiments/`](05_Experiments/) |
| 06 | Adaptasi ke konteks kolaborasi tim | dataset internal + protokol IRB | _Tahun-2 H1_ |
| 07 | Validasi cross-dataset + publikasi | naskah jurnal Q1 | _Tahun-2 H2_ |

---

## Quick start

### 1. Akuisisi data

```bash
cd 01_Akuisisi_Data
# Metadata uji (~235 KB)
python scripts/download_cogbci.py --out raw/COG-BCI --metadata-only
# Pilot 3 subjek COG-BCI (~3,3 GB)
python scripts/download_cogbci.py --out raw/COG-BCI --subjects 01 02 03
# STEW — unduh manual via IEEE DataPort, lalu:
python scripts/verify_stew.py --zip "$HOME/Downloads/STEW Dataset.zip" --out raw/STEW
```

Detail lengkap (kebutuhan disk, lisensi, sitasi wajib): [`01_Akuisisi_Data/README.md`](01_Akuisisi_Data/README.md).

### 2. Setup environment & preprocessing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r 02_Preprocessing/requirements.txt
cd 02_Preprocessing
python scripts/preprocess_cogbci.py --config config/cogbci.json --subjects 01 02 03
python scripts/preprocess_stew.py   --config config/stew.json
```

Output → `02_Preprocessing/derivatives/{COG-BCI,STEW}/sub-XX/.../*-epo.fif`.
Detail pipeline & QC: [`02_Preprocessing/README.md`](02_Preprocessing/README.md).

---

## Dataset

| Dataset | Subjek | Tasks | Channels | Lisensi | Sitasi wajib |
|--------|-------:|-------|---------:|---------|--------------|
| [COG-BCI](https://zenodo.org/records/6874129) | 29 × 3 sesi | MATB, N-Back, PVT, Flanker | 32 (BIDS) | CC-BY-4.0 | Hinss et al., *Sci Data* 2023, [10.1038/s41597-022-01898-y](https://doi.org/10.1038/s41597-022-01898-y) |
| [STEW](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset) | 48 | resting + SIMKAP | 14 (Emotiv EPOC) | Open Access | Lim et al., *IEEE TNSRE* 2018, [10.21227/44r8-ya50](https://doi.org/10.21227/44r8-ya50) |

Dataset **tidak** disimpan di repo (lihat `.gitignore`); skrip akan mengunduh on-demand
dengan verifikasi MD5.

> Asal-usul data lengkap (sumber mentah → preprocessing → ekstraksi fitur, dengan parameter
> aktual tiap tahap): [`README_DATASET.md`](README_DATASET.md).

---

## Tim

- **Sugeng Rifqi Mubaroq** — *Ketua Peneliti*, Akademi Digital Bandung — [mubaroq@digitalbdg.ac.id](mailto:mubaroq@digitalbdg.ac.id)
- Rolly Maulana Awangga
- Tegar Ditya Pragama
- Sidiq Fathummubin
- Ali Yusuf Abdulhaq

Kontribusi: silakan buka *issue* atau *pull request*. Pastikan setiap PR:
1. Lulus syntax check (`python -m py_compile`).
2. Memperbarui dokumentasi terkait (README tahap, config, manifest).
3. Tidak mengikutsertakan data mentah, dokumen substansi (.docx), atau anggaran (.xlsx).

---

## Sitasi

Bila menggunakan kode atau pipeline di repositori ini, mohon sitasi paper di atas
(lihat juga [`CITATION.cff`](CITATION.cff)) atau gunakan tombol *"Cite this repository"*
di sidebar GitHub.

## Lisensi

Kode dirilis di bawah [MIT License](LICENSE). Dataset mengikuti lisensi
masing-masing sumber (lihat di atas).

## Pendanaan

Penelitian Fundamental — Direktorat Jenderal Pendidikan Tinggi, Riset, dan
Teknologi (DIKTI), Kementerian Pendidikan Tinggi, Sains, dan Teknologi
Republik Indonesia, Tahun Anggaran 2026.

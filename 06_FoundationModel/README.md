# 06 — Foundation Model: Frozen-Encoder Probing + Class-Conditional DA

Implementasi jalur rekomendasi #1 dari tinjauan SOTA (Tahap 05): **EEG foundation model
flexible-montage + frozen-encoder probing**, untuk transfer beban kognitif COG-BCI↔STEW
yang tidak tertembus metode klasik (pooling/​re-centering/multi-task).

> **Status**: harness tervalidasi di MPS (placeholder) **dan** `CBraModEncoder` sudah
> di-*wire* via **braindecode** (`braindecode/cbramod-pretrained`). Tinggal jalankan dgn
> `encoder=cbramod` (unduh bobot otomatis; disarankan di Colab/GPU). Forward + pooling sudah
> diverifikasi shape-nya (features (b,14,2,200) → pooled (b,200)). Skala normalisasi (µV/100)
> = verifikasi terhadap preprocessing dataset CBraMod bila perlu presisi penuh.

## 1. Mengapa pendekatan ini (basis bukti)

Dari tinjauan literatur SOTA transfer EEG (lihat daftar pustaka paper):
- **Hanya foundation model** yang menangani heterogenitas 63-vs-14 kanal secara native
  (patch tokenization / conditional positional encoding) **dan mempertahankan power**
  (tak menghapusnya seperti re-centering Riemannian yang gagal di Tahap 05).
- Benchmark 2026 (arXiv:2604.23091, penulis EEGLAB): **SFT naif → negative transfer 24.8%**
  (CBraMod+Riemannian kolaps ke chance). Maka protokol AMAN = **bekukan encoder, probe linear**.
- Wawasan cdaDA (DOI 10.1007/s11571-024-10215-9): alignment harus **class-conditional**,
  bukan global → diimplementasikan sebagai `multi_ccda` (penyelarasan mean per-kelas di
  ruang embedding).

## 2. Arsitektur & yang sudah jalan

```
.fif (Tahap 02) → load_harmonized_epochs (14 kanal Emotiv + 128 Hz)
                → FoundationEncoder.embed (FROZEN, di MPS) → embedding (n, d)
                → probe linear (LogReg) :  SINGLE | MULTI | MULTI+ccDA
                → GroupKFold cross-subject (F1-macro)
```

- [`scripts/fm_encoder.py`](scripts/fm_encoder.py): pemuat epoch ter-harmonisasi +
  `PlaceholderEncoder` (CNN-1D frozen acak, **jalan di MPS sekarang**) + `CBraModEncoder` (stub).
- [`scripts/probe_transfer.py`](scripts/probe_transfer.py): probing + class-conditional DA,
  3 kondisi (single / multi / multi+ccDA), cross-subject.
- **Tervalidasi**: harness berjalan end-to-end di MPS (~2 dtk untuk 2×2000 epoch).

## 3. Menjalankan CBraMod (sudah terimplementasi)

`CBraModEncoder` ([`scripts/fm_encoder.py`](scripts/fm_encoder.py)) memuat CBraMod
pretrained via **braindecode** (`braindecode.models.CBraMod.from_pretrained`), membekukan
encoder, dan mengekstrak fitur lewat `forward(x, return_features=True)["features"]` →
rerata atas kanal & patch → embedding 200-dim.

**Langkah pakai:**
1. Install: `uv pip install braindecode` (di Colab: `!pip install braindecode`).
2. Set `"encoder": "cbramod"` di [`config/fm.json`](config/fm.json). Opsi cbramod sudah
   diisi: `hf_repo=braindecode/cbramod-pretrained`, `sfreq=200`, `n_times=400`, `scale=100`.
3. Jalankan: `python scripts/probe_transfer.py --config config/fm.json --both`.
   Bobot ter-unduh otomatis dari HuggingFace pada pemanggilan pertama.

**Detail teknis (sudah ditangani):**
- Input di-resample ke **200 Hz** (patch 200 sampel = 1 dtk); epoch 2 dtk → `n_times=400` = 2 patch.
- Skala: Volt → µV → /100 (konvensi CBraMod; ubah `scale` bila model card menuntut lain).
- 14 kanal Emotiv dipetakan langsung (CBraMod ACPE menangani jumlah kanal beragam).
- Disarankan **GPU** (Colab CUDA / MPS); CPU lambat untuk transformer 12-layer.

## 4. Protokol evaluasi (tetap)

- **Frozen encoder** (jangan SFT naif). Bila ingin fine-tuning, lakukan bertahap & bandingkan
  dengan probing (benchmark 2026 menunjukkan SFT bisa lebih buruk).
- **Cross-subject GroupKFold**, F1-macro (robust imbalans).
- Bandingkan SINGLE vs MULTI vs MULTI+ccDA → uji apakah foundation model + ccDA mencapai
  transfer positif yang gagal dicapai metode klasik.

## 5. Kebutuhan sumber daya

| Item | Status |
|------|--------|
| GPU | ✅ MPS lokal / CUDA di Colab (disarankan) |
| Jaringan HuggingFace | ✅ |
| `braindecode` | ⬜ `pip install braindecode` saat pakai cbramod |
| Bobot CBraMod | ⬜ unduh otomatis (`braindecode/cbramod-pretrained`) |

## 6. Catatan kebaruan

Kombinasi **foundation model + class-conditional DA + orkestrasi agentic** (Tahap 04) untuk
EEG workload lintas-dataset **tanpa preseden di literatur** (riset Tahap 05) — ruang kontribusi
orisinal proposal. Tahap 06 ini menyiapkan komponen foundation-model-nya.

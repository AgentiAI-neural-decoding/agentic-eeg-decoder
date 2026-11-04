# 07 — Harmonisasi Domain-Adversarial (Learned), sebelum Decoding

Prototipe yang **mempelajari** representasi EEG *device-invariant* untuk menyerang tembok
residual transfer (domain shift perangkat) yang dilokalisasi di Tahap 05–06.

## Ide inti (dan koreksi penting)
Harmonisasi datang dari **OBJEKTIF**, bukan arsitektur. Encoder (EEGNet-CNN **atau** Mamba)
memetakan epoch 14-kanal → latent; dilatih dengan:
- **task-loss** (workload), dan
- **domain-adversarial loss** lewat **Gradient Reversal Layer (GRL)** yang memaksa latent
  tak-terbedakan antar-dataset (COG-BCI vs STEW) → invariansi domain yang dipelajari.

Backbone (CNN vs Mamba) di-**ablation** untuk efisiensi/edge — Mamba *bukan* sumber
harmonisasi, melainkan encoder temporal alternatif (O(L), ringan).

## Yang dibandingkan (cross-subject GroupKFold, target hold-out)
| Kondisi | Latih pada | Adversary |
|---------|-----------|-----------|
| **SINGLE** | target-train saja | — |
| **POOL** (λ=0) | target-train + sumber | tidak (pooling yang dipelajari) |
| **DANN** (λ>0) | target-train + sumber | ya (GRL domain-invariant) |

Hipotesis: **DANN > POOL** (dan idealnya ≥ SINGLE) bila invariansi yang dipelajari menutup
domain shift yang gagal diatasi pooling naif (Tahap 05) & re-centering (Tahap 06).

## Status & temuan awal (jujur)
- Pipeline **tervalidasi** (GRL, training, eval, metrik efisiensi). Encoder EEGNet **3.764 param**,
  inferensi ~0,2 ms/epoch (sangat ringan → argumen edge kuat).
- **Raw-EEG cross-subject lebih sulit dari fitur tangan**: EEGNet single COG-BCI(extreme)
  ≈ 0,55 (40 epoch, data terbatas) vs PSD+LogReg 0,775. Konsisten literatur: fitur spektral
  adalah baseline cross-subject yang kuat; deep-net raw butuh lebih banyak data/epoch.
- → **Eksperimen penuh harus di Colab (GPU CUDA, epoch lebih banyak)**; Mamba **wajib** CUDA
  (`mamba-ssm`). Angka final = dari Colab.

## Cara jalan (Colab GPU)
```bash
# (bootstrap sesi spt biasa: mount Drive, clone, unzip derivatives)
!pip install -q mamba-ssm causal-conv1d   # hanya utk backbone mamba (butuh CUDA)
!cd /content/agentic-eeg-decoder && python 07_Harmonization/scripts/dann.py \
    --config 07_Harmonization/config/dann.json --backbone cnn  --both
!cd /content/agentic-eeg-decoder && python 07_Harmonization/scripts/dann.py \
    --config 07_Harmonization/config/dann.json --backbone mamba --both
!cp /content/agentic-eeg-decoder/07_Harmonization/results/*.json /content/drive/MyDrive/agentic-eeg/checkpoints/
```
Naikkan `epochs` di config (mis. 60–100) untuk konvergensi penuh di GPU.

## Berkas
- [`scripts/dann.py`](scripts/dann.py) — GRL, encoder EEGNet/Mamba, label+domain head,
  harness SINGLE/POOL/DANN + efisiensi.
- [`config/dann.json`](config/dann.json) — kanal bersama, sfreq, λ, CV, epoch.
- `results/dann_{cnn,mamba}_results.json` — output.

## Catatan ilmiah
Eksperimen ini menguji apakah **harmonisasi yang dipelajari (domain-adversarial)** mengungguli
harmonisasi-fixed + class-conditional DA post-hoc (Tahap 06). Adversarial DA terkenal sulit-
stabil; hasil negatif pun informatif (menegaskan domain shift sbg tembok keras). Backbone Mamba
menyumbang sisi **efisiensi/edge** (params/latency), terukur terpisah dari akurasi.

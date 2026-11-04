# 05 — Eksperimen Offline, Ablation & Transfer Lintas-Dataset

Tahap kelima: eksperimen offline sistematis di atas arsitektur agentic (Tahap 04) +
fitur (Tahap 03). Fokus: (1) ablation kontribusi komponen, (2) uji hipotesis inti
proposal — transfer lintas-dataset, (3) eksplorasi metode lanjutan berbasis literatur SOTA.

## 1. Ablation study ([`scripts/ablation.py`](scripts/ablation.py))

Cross-subject GroupKFold, F1-macro. Hasil (`results/ablation_results.json`):

| Varian | MATB (COG-BCI) | STEW |
|--------|---------------:|-----:|
| only_spectral | 0.546 | **0.741** |
| only_riemann | **0.594** | 0.677 |
| only_csp | 0.517 | 0.736 |
| full_softvote | **0.620** | 0.772 |
| full_stacking | 0.612 | **0.777** |

**Temuan**: spesialis dominan **berbeda per dataset** (Riemann→COG-BCI, Spektral→STEW) →
justifikasi kuat ensemble. Orkestrator selalu di/dekat puncak (robust); spesialis tunggal
gagal di dataset bukan kekuatannya. **CSP kontribusi terkecil** (ΔF1 buang-CSP ≈ −0.002).

## 2. Uji transfer lintas-dataset (hipotesis inti proposal: +40% vs single)

Latih target+sumber, uji subjek TARGET hold-out (GroupKFold). Tiga keluarga metode diuji:

| Metode | Skrip | STEW gain (F1) | COG-BCI gain (F1) |
|--------|-------|---------------:|------------------:|
| Pooling naif (Riemann) | [`transfer_da.py`](scripts/transfer_da.py) | negatif | negatif (akurasi semu) |
| **Re-centering Riemannian (TLCenter)** | [`transfer_da.py`](scripts/transfer_da.py) | **lebih buruk** | **jatuh ke ~chance** |
| Multi-task shared-backbone | [`multitask.py`](scripts/multitask.py) | −0.035 | **−0.008 (≈netral)** |
| + auxiliary down-weighted (0.3) | [`multitask.py`](scripts/multitask.py) | −0.028 | −0.012 |

**Temuan utama (jujur)**: TIDAK ada metode di sini yang memberi transfer positif
COG-BCI↔STEW. Re-centering Riemannian **kontraproduktif** — menghapus sinyal workload yang
hidup di **magnitude kovarians** (desinkronisasi power). Multi-task shared-backbone (Family D)
**paling tidak merugikan** (praktis netral untuk COG-BCI), karena head terpisah tak memaksa
semantik label identik + PSD mempertahankan power.

**Mengapa sulit (terkonfirmasi literatur)**:
1. **Domain shift** — COG-BCI (riset 63-ch) vs STEW (Emotiv 14-ch) beda perangkat/sinyal.
2. **Mismatch konstruk** — "workload" = MATB/N-back/PVT/Flanker (COG-BCI) vs SIMKAP (STEW).
3. **Sinyal di power** — re-centering yang lazim justru menghapusnya.

Benchmark 2026 (arXiv:2604.23091, penulis EEGLAB) **mengonfirmasi**: CBraMod + re-centering
Riemannian kolaps ke chance. Temuan proyek ini sejalan SOTA, bukan artefak.

## 3. Arah lanjutan berbasis literatur SOTA

Ringkas jalur paling menjanjikan & belum tertembus:

1. **Foundation model flexible-montage** (CBraMod/BIOT) + **frozen-encoder probing** —
   satu-satunya keluarga yang menangani heterogenitas 63-vs-14 native & mempertahankan power.
2. **Class-conditional domain adaptation** (cdaDA) — bukan alignment global.
3. **T3A** test-time adaptation — adaptasi STEW tanpa label.
4. **Hindari** re-centering Riemannian pada magnitude.

## 4. Cara jalan

```bash
cd 05_Experiments
../.venv/bin/python scripts/ablation.py    --config config/experiments.json
../.venv/bin/python scripts/transfer_da.py --config config/experiments.json --both
../.venv/bin/python scripts/multitask.py   --config config/experiments.json --both
```

## 5. Berkas

- `scripts/transfer_da.py` — transfer SINGLE vs naif vs DA (TLCenter re-centering).
- `scripts/ablation.py` — kontribusi spesialis & tipe orkestrator.
- `scripts/multitask.py` — shared-backbone multi-task (Family D, PyTorch) + aux-weighting.
- `config/experiments.json`, `results/*.json`.

## 6. Kesimpulan Tahun-1

- **Within-dataset decoding kuat** (Tahap 03–04): STEW 0.77, MATB 0.61 (F1, cross-subject).
- **Orkestrator agentic menambah nilai** (robust > spesialis tunggal).
- **Transfer lintas-dataset = tantangan terbuka** yang dikarakterisasi empiris + dipetakan
  ke metode SOTA — bukan kegagalan, melainkan temuan yang memotivasi kontribusi lanjut
  (foundation model + class-conditional DA + orkestrasi agentic, ruang yang belum dieksplorasi).

# 04 — Arsitektur Agentic AI (Orchestrator + Spesialis)

Tahap keempat penelitian. Mengimplementasikan **decoder beban kognitif multi-spesialis
yang dikoordinasi orkestrator** — pemetaan "orchestrator + spesialis" proposal ke
sistem yang dapat divalidasi offline. Input: fitur Tahap 03. Output: model + hasil CV.

## 1. Arsitektur

```
            views (Tahap 03)
   ┌──────────┬──────────────┬──────────┐
   │ PSD (70) │  cov(14×14)  │ cov(14×14)│
   ▼          ▼              ▼
 [Spektral] [Riemannian]   [CSP]          ← AGEN SPESIALIS (tiap satu view)
  SVM-RBF    TangentSpace   pyriemann      (selaras proposal: SVM, LDA, LogReg; RF/LSTM opsional)
             +LogReg        CSP+LDA
   │          │              │
   └────►  probabilitas  ◄───┘
              ▼
        [ ORKESTRATOR ]                    ← stacking meta-learner (atau soft-vote)
         meta-LogReg atas
         out-of-fold proba
              ▼
          prediksi beban kognitif
```

- **Agen spesialis** ([`scripts/agents.py`](scripts/agents.py)): tiap agen pipeline sklearn
  yang hanya melihat satu "view" fitur. Spektral menangkap penanda theta/alpha; Riemannian
  & CSP menangkap struktur spasial kovarians.
- **Orkestrator**: menggabungkan keputusan spesialis. Mode `stacking` melatih meta-classifier
  atas **out-of-fold probabilities** spesialis (anti-leakage — meta tak melihat data latih
  spesialis). Mode `soft_vote` = rerata proba (baseline).

## 2. Anti-leakage (kritis untuk validitas)

- TangentSpace & CSP di-*fit* hanya pada data latih tiap fold (di dalam pipeline).
- Meta-learner stacking dilatih atas OOF proba (GroupKFold internal per subjek) →
  tak ada kebocoran spesialis→meta.
- Evaluasi luar: **GroupKFold per subjek** (cross-subject) — tak ada bocor identitas subjek.

## 3. Cara jalan

```bash
cd 04_Agentic
../.venv/bin/python scripts/evaluate.py --config config/agentic.json          # spesialis vs orkestrator
../.venv/bin/python scripts/transfer.py --config config/agentic.json --both   # transfer lintas-dataset
```

## 4. Hasil — orkestrator vs spesialis (cross-subject, 29/47 subjek, subsample ≤12k)

| Target | spektral | riemann | csp | **orkestrator** | Δacc | ΔF1 |
|--------|---------:|--------:|----:|----------------:|-----:|----:|
| COG-BCI N-back (3-kelas, chance 0.34) | 0.514 | 0.515 | 0.456 | **0.533** | +0.018 | +0.018 |
| COG-BCI MATB (3-kelas, chance 0.34) | 0.570 | 0.582 | 0.521 | **0.611** | +0.029 | +0.028 |
| COG-BCI workload biner (chance 0.74) | 0.741 | 0.610 | 0.743 | 0.652 | −0.091 | **+0.026** |
| STEW workload biner (chance 0.55) | 0.742 | 0.670 | 0.741 | **0.770** | +0.028 | +0.030 |

**Orkestrator mengungguli spesialis terbaik pada keempat target bila diukur F1-macro**
(metrik yang benar untuk kelas tak seimbang). Pada COG-BCI workload biner, akurasi mentah
menyesatkan: CSP mencapai 0.743 hanya dengan menebak kelas mayoritas (F1 0.464), sedangkan
orkestrator (balanced) F1 0.612. **Riemannian > spektral** untuk task graded — konsisten literatur.

> Angka ini adalah baseline linear cross-subject. Target proposal (akurasi >85%) ditargetkan
> via penguatan: spesialis deep (LSTM/CNN TensorFlow), penalaan, dan agregasi within-subject.

## 5. Uji transfer lintas-dataset — temuan penting

Klaim inti proposal: integrasi multi-dataset meningkatkan akurasi (+40% vs single). Uji
(latih target+sumber, uji subjek target hold-out) memberi hasil **jujur & instruktif**:

| Arah | SINGLE (acc/F1) | MULTI +sumber (acc/F1) | Δakurasi |
|------|-----------------|------------------------|----------|
| STEW + COG-BCI | 0.770 / 0.766 | 0.715 / 0.690 | **−7.2%** |
| COG-BCI + STEW | 0.652 / 0.612 | 0.711 / 0.582 | +9.0% (tapi F1 ↓) |

**Pooling NAIF lintas-dataset menyebabkan negative/no transfer.** Penyebab: **domain shift**
— COG-BCI (EEG riset 63→14 ch, MATB/N-back) vs STEW (Emotiv consumer, SIMKAP) berbeda
karakteristik sinyal & operasionalisasi "workload". Menggabung mentah merusak distribusi.

**Implikasi (memvalidasi urgensi proposal):** mencapai +40% **menuntut transfer learning /
domain adaptation proper**, bukan penggabungan mentah. Arah Tahap 05:
- **Re-centering Riemannian per-domain** (`pyriemann.transfer`: TLCenter/TLStretch/TLRotate
  — Riemannian Procrustes Analysis) untuk menyelaraskan kovarians antar-domain.
- Penyamaan konstruk label antar-dataset / subject alignment (Euclidean/Riemannian alignment).
- Spesialis deep + agregasi within-subject untuk menembus target akurasi.

## 6. Berkas

- [`scripts/agents.py`](scripts/agents.py) — spesialis + `AgenticDecoder` (orkestrator).
- [`scripts/evaluate.py`](scripts/evaluate.py) — CV spesialis vs orkestrator.
- [`scripts/transfer.py`](scripts/transfer.py) — uji transfer single vs multi-dataset.
- [`config/agentic.json`](config/agentic.json) — spesialis, orkestrator, CV.
- `results/agentic_results.json`, `results/transfer_results.json` — hasil CV.

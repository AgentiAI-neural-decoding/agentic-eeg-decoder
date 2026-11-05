# 08 · Graded workload index

Analysis code for the graded cognitive-workload index study (manuscript under review).
A lightweight multi-specialist EEG decoder (spectral, Riemannian and CSP views fused by a
stacking meta-learner; stage `04_Agentic`) is turned into a continuous load index
E[level] = Σ k·P(k) over three ordinal difficulty levels, and evaluated on the 14-channel
consumer montage shared with the Emotiv EPOC.

## Data

| Dataset | Use | Participants |
|---|---|---|
| COG-BCI (Hinss et al., 2023) | MATB and n-back, three levels, three sessions | 20 (see below) |
| EEGMAT (Zyma et al., 2019) | rest vs mental arithmetic, performance groups | 36 |
| STEW (Lim et al., 2018) | consumer headset, subjective ratings 1–9 | 47 |
| ds003838 (Pavlov et al., 2022) | digit span, research-grade, different site | 65 |

**COG-BCI integrity.** The public release contains participant IDs with byte-identical
EEG: sub-17 to sub-24 and sub-27 share every recording, and sub-25 and sub-28 share
session S1. We exclude every ID that shares at least one recording with another ID,
keeping the lowest-numbered ID of each cluster (sub-17, sub-25), which leaves 20
participants. Check any copy of the release with `tools/verify_cogbci_duplicates.py`.

Feature sets: `03_Features/features_v2` (sub-18 to sub-24 and sub-27 removed) is built by
`03_Features/scripts/extract_features.py` with `config/features_v2.json` and by
`scripts/extract_sessions_v2.py`; `03_Features/scripts/make_features_v3.py` then removes
sub-28 and writes `features_v3`, the input of every `*_v3.py` script. The epoch files read by
the CBraMod baseline, `02_Preprocessing/derivatives_v3/COG-BCI`, are the 20 retained
participant folders of `derivatives_v2/COG-BCI` (made with `02_Preprocessing/config/cogbci_v2.json`
and `scripts/run_pre_v2.sh`), copied as real directories (hard links), not symbolic links:
`Path.rglob` does not descend into linked directories.

## Scripts and results

Each `*_v3.py` script is a copy of the matching `*_v2.py` in which only the input and
output paths differ (`features_v2` to `features_v3`, `results_v2` to `results_v3`,
`derivatives_v2` to `derivatives_v3`); `jalankan_v3.sh` generates these copies and then runs
the whole chain. Docstrings inside the copies therefore still carry the `_v2` file names.
`jalankan_v3b.sh` is the parallel continuation that was used to finish the run (same scripts
and arguments, independent stages run side by side). Results of the 21-ID v2 run are not
included because they are superseded.

| Script | Analysis | Output |
|---|---|---|
| `graded_v3.py` | graded index, monotonicity, Spearman, macro-F1 (GroupKFold-8, 9,000 epochs) | `graded.json` |
| `deepen_v3.py` | per-participant tracking, alpha-only baseline, cross-task, dose-response | `deepen.json` |
| `multiseed_fdr_v3.py` | five seeds, Benjamini–Hochberg FDR over a pre-declared family of 12 tests | `multiseed_summary.json`, `multiseed_tables.tex`, `multiseed_report.md` |
| `icc_montage_v3.py`, `icc_ci_v3.py` | test–retest ICC(2,1) across sessions with F and bootstrap intervals; 14- vs 58-channel montage | `icc_montage.json`, `icc_montage_ci.json` |
| `strengthen_light_v3.py` | channel ablation, window length, in-silico adaptation loop | `strengthen_light.json` |
| `cbramod_graded_v3.py` | frozen CBraMod foundation-model baseline and random-init control | `cbramod_graded.json` |
| `stew_rating_v3.py` | transfer of the MATB index to a consumer headset and to subjective ratings | `stew_rating.json` |
| `eegmat_full.py`, `prep_eegmat.py` | EEGMAT preparation and analysis (dataset unaffected by the integrity issue) | `results/eegmat_full.json` |

| `10_Revision_IASC_MOABB/scripts/e1_loso_orchestrator_v3.py` | leave-one-subject-out comparison of the stacking decoder with each specialist (MOABB-style protocol, `moabb_protocol.py`) | `10_Revision_IASC_MOABB/results_v3/e1_loso.json` |

Cross-device analyses on the same feature set are in `new_analysis/scripts`
(`08_ds003838_transfer.py`, `13_hinss_leakfree.py`, with `ND_FEAT_ORIG=03_Features/features_v3`
and `ND_DATA_OUT=new_analysis/data_v3`; results in `new_analysis/data_v3/`; ds003838 and
Hinss2021 are prepared by `preprocess_ds003838.py` and `preprocess_hinss.py`).
`12_overlap_hinss_cogbci.py` (result `new_analysis/data_v2/overlap_hinss_cogbci.json`,
computed on raw recordings) documents that the Neuroergonomics 2021 hackathon data
(Hinss2021) are recordings of COG-BCI participants sub-01 to sub-15; Hinss2021 is therefore
not an independent dataset. In `hinss_leakfree.json` the size-matched control keys are named
`*_COG_ov6`, but each draw has the size of the overlap-free group (5 participants).

## Environment

MNE 1.12.1, mne-icalabel 0.9.0, scikit-learn 1.9.0, NumPy 2.5.1, SciPy 1.18.0,
pyRiemann 0.12, PyTorch 2.11.0, braindecode 1.6.1 (weights `braindecode/cbramod-pretrained`).
CPU for all analyses except CBraMod embedding (CUDA GPU).

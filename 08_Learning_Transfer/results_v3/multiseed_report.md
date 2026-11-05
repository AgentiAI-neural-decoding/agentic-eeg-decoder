# Multi-seed + FDR report (5 seeds)

Values are mean +/- SD across seeds. `manuscript` is the current single-seed
number in MANUSCRIPT_ADAPTIVE.tex. A value is FLAGGED when the manuscript
number falls outside the multi-seed 95% interval -- it must then be updated.

## Drift vs manuscript

| quantity | manuscript | multi-seed | 95% interval | verdict |
|---|---|---|---|---|
| n-back macro-F1 | 0.512 | 0.340 +/- 0.015 | [0.324, 0.357] | **FLAGGED** |
| n-back rho (subj-mean) | 0.485 | 0.134 +/- 0.026 | [0.100, 0.166] | **FLAGGED** |
| n-back ordinal MAE | 0.579 | 0.666 +/- 0.005 | [0.661, 0.673] | **FLAGGED** |
| MATB macro-F1 | 0.615 | 0.489 +/- 0.009 | [0.479, 0.498] | **FLAGGED** |
| MATB rho (subj-mean) | 0.661 | 0.559 +/- 0.027 | [0.529, 0.591] | **FLAGGED** |
| MATB ordinal MAE | 0.498 | 0.631 +/- 0.009 | [0.620, 0.640] | **FLAGGED** |
| n-back 14ch macro_f1 | 0.508 | 0.340 +/- 0.015 | [0.324, 0.357] | **FLAGGED** |
| n-back 14ch spearman | 0.353 | 0.071 +/- 0.017 | [0.049, 0.088] | **FLAGGED** |
| n-back 58ch macro_f1 | 0.529 | 0.341 +/- 0.016 | [0.320, 0.362] | **FLAGGED** |
| n-back 58ch spearman | 0.423 | 0.095 +/- 0.026 | [0.066, 0.127] | **FLAGGED** |
| MATB 14ch macro_f1 | 0.613 | 0.489 +/- 0.009 | [0.479, 0.498] | **FLAGGED** |
| MATB 14ch spearman | 0.539 | 0.388 +/- 0.010 | [0.377, 0.399] | **FLAGGED** |
| MATB 58ch macro_f1 | 0.652 | 0.522 +/- 0.008 | [0.510, 0.530] | **FLAGGED** |
| MATB 58ch spearman | 0.608 | 0.436 +/- 0.018 | [0.409, 0.450] | **FLAGGED** |

## Hardware-retention claim (abstract says 94-96% and 83-89%)

| paradigm | macro-F1 retained | rho retained |
|---|---|---|
| n-back | 99.7 +/- 4.1 % | 77.7 +/- 25.5 % |
| MATB | 93.6 +/- 1.2 % | 89.3 +/- 3.6 % |

If these intervals no longer span 94-96 / 83-89, update the abstract, the
Results paragraph, AND the cover letter -- the claim appears in all three.

## Alpha-baseline comparison (new: with a significance test)

| paradigm | decoder rho | alpha rho | fold-change | Wilcoxon p (median) |
|---|---|---|---|---|
| n-back | 0.071 +/- 0.017 | 0.087 +/- 0.009 | 0.83 +/- 0.26x | 0.7 |
| MATB | 0.388 +/- 0.010 | 0.183 +/- 0.008 | 2.13 +/- 0.11x | 9.5e-06 |

The manuscript claims 2.5- to 3.4-fold. Update that range if it has moved.

## Benjamini-Hochberg FDR

Family declared before running (12 tests), corrected as one family.
Per-seed p-values summarised by their median; max shown so fragile claims are visible.

`kind` records what the manuscript claims. For **effect** rows, surviving FDR is
required. For **null** rows (the demand-not-skill argument), NON-significance is
the claim -- a surviving p there would contradict the manuscript instead.

| test | kind | p median | p max | p adjusted | significant | verdict |
|---|---|---|---|---|---|---|
| spearman_subjectmean_n-back | effect | 0.302 | 0.457 | 0.724 | no | **CLAIM BROKEN** |
| spearman_subjectmean_MATB | effect | 2.62e-06 | 1.45e-05 | 1.05e-05 | yes | holds |
| spearman_epoch_n-back | effect | 1.44e-13 | 5.27e-06 | 8.62e-13 | yes | holds |
| spearman_epoch_MATB | effect | 0 | 1.24e-300 | 0 | yes | holds |
| decoder_vs_alpha_n-back | effect | 0.701 | 1 | 0.841 | no | **CLAIM BROKEN** |
| decoder_vs_alpha_MATB | effect | 9.54e-06 | 8.2e-05 | 2.86e-05 | yes | holds |
| eegmat_load_good_vs_poor | null | 0.633 | 0.633 | 0.841 | no | null confirmed |
| eegmat_load_vs_subtractions | null | 0.686 | 0.686 | 0.841 | no | null confirmed |
| eegmat_theta_good_vs_poor | null | 0.685 | 0.685 | 0.841 | no | null confirmed |
| eegmat_theta_vs_subtractions | null | 0.67 | 0.67 | 0.841 | no | null confirmed |
| eegmat_alpha_good_vs_poor | null | 0.986 | 0.986 | 0.995 | no | null confirmed |
| eegmat_alpha_vs_subtractions | null | 0.995 | 0.995 | 0.995 | no | null confirmed |

### Effects that did NOT survive FDR

Every sentence resting on these must be reworded as a trend, or dropped:

- spearman_subjectmean_n-back
- decoder_vs_alpha_n-back


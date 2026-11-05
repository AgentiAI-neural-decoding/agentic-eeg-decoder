#!/usr/bin/env python3
"""Shared MOABB-style evaluation protocol for the IASC revision (round 2).

This module reimplements the MOABB benchmarking *protocol* on top of the existing
custom pipeline (no MOABB library dependency, no dataset re-wrapping):

  * Evaluation  = CrossSubject  = **Leave-One-Subject-Out (LOSO)**  (outer loop).
                  One score per held-out subject -> N per-subject observations.
  * Model sel.  = **inner k-fold** (subject-grouped) for stacking OOF + any
                  hyperparameter selection (nested CV).  -> answers R2.11.
  * Metric      = **ROC-AUC** (binary; chance = 0.5 by definition -> R2.4),
                  macro-AUC (OvR) for multiclass, plus macro-F1 kept as a
                  secondary metric for continuity with the original submission.
  * Statistics  = per-subject **Wilcoxon signed-rank** + paired effect size
                  (Cohen's d_z / SMD); across datasets combine with **Stouffer**;
                  family-wide **Benjamini-Hochberg FDR** for multiplicity (R1.5/R2.5).

Rationale vs the round-1 (k=8, seeded GroupKFold, Nadeau-Bengio) pipeline:
LOSO test sets are disjoint across subjects, so the anti-conservative
fold-overlap correction (Nadeau-Bengio) is no longer needed; a plain paired
Wilcoxon across subjects is valid and higher-powered (N=29..47 vs k=8).

Non-destructive: pure helpers, no I/O side effects except what callers do.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def per_subject_cap(y, groups, cap, seed):
    """Stratified per-subject subsample: keep at most `cap` epochs per subject,
    drawn to preserve that subject's class proportions. Returns kept indices.

    This is the MOABB-aligned answer to R2.2: instead of a single global
    5,000-epoch draw, every subject contributes its own (capped) data, so no
    subject dominates and LOSO per-subject scoring is well-posed. cap=0 or None
    keeps everything.
    """
    y = np.asarray(y); groups = np.asarray(groups)
    if not cap:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        if len(gi) <= cap:
            keep.append(gi)
            continue
        # stratified draw within subject
        yg = y[gi]
        classes, counts = np.unique(yg, return_counts=True)
        # proportional allocation, at least 1 per present class
        alloc = np.maximum(1, np.round(cap * counts / counts.sum()).astype(int))
        # fix rounding so sum == cap
        while alloc.sum() > cap:
            alloc[alloc.argmax()] -= 1
        while alloc.sum() < cap:
            alloc[alloc.argmin()] += 1
        sub = []
        for c, a in zip(classes, alloc):
            ci = gi[yg == c]
            sub.append(rng.choice(ci, size=min(a, len(ci)), replace=False))
        keep.append(np.concatenate(sub))
    out = np.concatenate(keep)
    out.sort()
    return out


# --------------------------------------------------------------------------- #
# Outer evaluation: Leave-One-Subject-Out (MOABB CrossSubject)
# --------------------------------------------------------------------------- #
def loso_splits(groups):
    """Yield (train_idx, test_idx, held_out_subject) for each subject."""
    groups = np.asarray(groups)
    for s in np.unique(groups):
        te = np.where(groups == s)[0]
        tr = np.where(groups != s)[0]
        if len(te) and len(tr):
            yield tr, te, s


# --------------------------------------------------------------------------- #
# Metrics (per held-out subject)
# --------------------------------------------------------------------------- #
def score_subject(y_true, proba, pred, classes):
    """Return dict of per-subject metrics. AUC is NaN if the held-out subject
    does not contain the classes needed to define it (handled by callers)."""
    y_true = np.asarray(y_true); classes = np.asarray(classes)
    out = {}
    # macro-F1 (secondary; always defined)
    out["macro_f1"] = float(f1_score(y_true, pred, average="macro", labels=classes,
                                     zero_division=0))
    out["balanced_acc"] = float(balanced_accuracy_score(y_true, pred))
    # ROC-AUC (primary)
    try:
        if len(classes) == 2:
            # column for the positive (max) class label
            pos = int(np.argmax(classes))
            out["auc"] = float(roc_auc_score((y_true == classes[pos]).astype(int),
                                             proba[:, pos]))
        else:
            out["auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                             average="macro", labels=classes))
    except (ValueError, IndexError):
        out["auc"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def paired_stats(a, b):
    """Paired comparison of two per-subject score vectors a (test) vs b (ref).
    NaNs (undefined AUC) are dropped pairwise. Returns Wilcoxon p, SMD (d_z),
    mean diff, and direction counts.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    diff = a - b
    n = len(diff)
    res = {"n_subjects": int(n),
           "mean_a": float(a.mean()) if n else float("nan"),
           "mean_b": float(b.mean()) if n else float("nan"),
           "mean_diff": float(diff.mean()) if n else float("nan"),
           "sd_diff": float(diff.std(ddof=1)) if n > 1 else float("nan"),
           "n_pos": int((diff > 0).sum()),
           "n_nonneg": int((diff >= 0).sum())}
    # paired effect size (Cohen's d_z)
    res["smd_dz"] = float(diff.mean() / diff.std(ddof=1)) if n > 1 and diff.std(ddof=1) > 0 else float("nan")
    # Wilcoxon signed-rank (one-sided: a > b), fall back gracefully
    try:
        res["wilcoxon_p_greater"] = float(stats.wilcoxon(a, b, alternative="greater").pvalue)
        res["wilcoxon_p_two"] = float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        res["wilcoxon_p_greater"] = float("nan")
        res["wilcoxon_p_two"] = float("nan")
    return res


def stouffer(pvalues, weights=None):
    """Combine p-values across datasets (Stouffer's Z). One-sided inputs."""
    p = np.asarray(pvalues, float)
    m = np.isfinite(p)
    p = np.clip(p[m], 1e-12, 1 - 1e-12)
    if len(p) == 0:
        return float("nan")
    w = np.ones_like(p) if weights is None else np.asarray(weights, float)[m]
    z = stats.norm.isf(p)  # one-sided
    zc = np.sum(w * z) / np.sqrt(np.sum(w ** 2))
    return float(stats.norm.sf(zc))


def bh_fdr(pvalues):
    """Benjamini-Hochberg adjusted q-values (same order as input)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m, float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = m - rank
        val = min(prev, p[idx] * m / i)
        q[idx] = val
        prev = val
    return q

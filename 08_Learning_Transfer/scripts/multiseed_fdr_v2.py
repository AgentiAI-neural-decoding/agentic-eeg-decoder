#!/usr/bin/env python3
"""Multi-seed re-run + Benjamini-Hochberg FDR for the adaptive-learning manuscript.

WHY THIS EXISTS
---------------
Every number in MANUSCRIPT_ADAPTIVE.tex is a point estimate from ONE run of the
pipeline. Reviewers of the companion IASC manuscript demanded multi-seed
robustness and multiple-comparison correction; this script supplies both for
Tables 1-3 (the empirical backbone) before the same demand arrives here.

WHAT VARIES PER SEED (and why all three are necessary)
-----------------------------------------------------
The original scripts are almost deterministic, so a naive "multi-seed" run that
only reshuffles the epoch subsample would report an over-optimistically small SD.
Three independent sources of variance are therefore varied together:

  1. epoch subsample     - `load(cap, seed)` draws `cap` epochs out of the corpus.
  2. subject->fold split  - sklearn's GroupKFold has NO shuffle parameter: it
                            assigns subjects to folds deterministically by group
                            size, so every rerun reuses the same partition. This
                            script permutes the subject order before assignment,
                            which is the dominant source of cross-subject variance
                            and the one a reviewer actually cares about.
  3. classifier RNG       - agents.make_classifier hardcodes random_state=42; with
                            probability=True the SVC runs internal Platt-scaling
                            CV, which is stochastic. Re-seeded per run here.

`agents.py` is NOT modified: it is shared with the accepted IASC manuscript and
must keep producing that paper's numbers. The classifier factory is re-implemented
locally and composed with the original SpecialistAgent/AgenticDecoder classes, so
the model architecture is identical - only the RNG is threaded through.

STATISTICS
----------
* Aggregation: mean +/- SD across seeds, plus a percentile 95% interval. Report
  as "mean +/- SD (n seeds)" in the manuscript.
* Baseline comparison: the manuscript currently asserts the decoded index beats a
  posterior-alpha baseline by a RATIO, with no significance test. This script adds
  the defensible version - a per-subject paired Wilcoxon signed-rank test on
  within-subject Spearman rho (decoder vs alpha), which respects the nesting of
  epochs within participants.
* FDR: Benjamini-Hochberg (independent/PRDS) over a PRE-DECLARED family of the
  manuscript's inferential claims - see FAMILY below. Declaring the family up
  front is what makes the correction honest; do not extend it after seeing results.
* Per-seed p-values are summarised by their MEDIAN before correction, with
  min/max also reported. Averaging p-values is not valid; the median across
  resamples of the same data is a stable representative value. If median and max
  fall on opposite sides of the threshold the script flags the claim as FRAGILE.

USAGE
-----
    # restore the pipeline first (it is in git HEAD, deleted from the worktree):
    #   git checkout HEAD -- 08_Learning_Transfer/ 04_Agentic/ 03_Features/
    # then place this file in 08_Learning_Transfer/scripts/ and run:

    python multiseed_fdr.py --seeds 5 --n-jobs 5
    python multiseed_fdr.py --seeds 10 --n-jobs 10 --analyses graded,montage,baseline
    python multiseed_fdr.py --resume          # continue an interrupted run

*** THIS IS CPU-BOUND, NOT GPU WORK. ***
The pipeline is scikit-learn (SVC-RBF, LDA, LogReg) plus pyriemann. None of it
touches CUDA; a GPU will sit idle. What helps is CORE COUNT: set --n-jobs to the
number of seeds and give each worker a core. Set --n-jobs 1 on a memory-tight box.

Outputs (all under --out-dir, default ../results):
    multiseed_seed<NN>.json   per-seed raw metrics (checkpoint; enables --resume)
    multiseed_summary.json    aggregated mean/SD/CI + FDR verdicts
    multiseed_tables.tex      drop-in LaTeX replacements for Tables 1-3
    multiseed_report.md       human-readable diff vs the current manuscript values
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
try:
    from agents import AgenticDecoder, SpecialistAgent  # noqa: E402
    from pyriemann.spatialfilters import CSP  # noqa: E402
    from pyriemann.tangentspace import TangentSpace  # noqa: E402
except ModuleNotFoundError as exc:  # keep --help usable before the pipeline is restored
    missing = exc.name
    if missing == "agents":
        _HINT = (f"cannot import the decoder from {BASE / '04_Agentic' / 'scripts'}.\n"
                 "        The pipeline is in git HEAD but deleted from the worktree. Restore it:\n"
                 "          git checkout HEAD -- 03_Features/ 04_Agentic/ 08_Learning_Transfer/\n"
                 "        Then place this script in 08_Learning_Transfer/scripts/ and rerun.")
    else:
        _HINT = (f"missing package '{missing}'.\n"
                 "        pip install numpy scipy scikit-learn pyriemann joblib")
    AgenticDecoder = SpecialistAgent = CSP = TangentSpace = None

    def _require_deps():
        sys.exit(f"[fatal] {_HINT}")
else:
    def _require_deps():
        return None

FEAT = BASE / "03_Features" / "features_v2"

CFG = {
    "specialists": {
        "spectral": {"view": "psd", "classifier": "svm_rbf"},
        "riemann": {"view": "cov", "classifier": "logreg"},
        "csp": {"view": "cov", "classifier": "lda", "n_filters": 6},
    },
    "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3},
}
GRADED = {"COG-BCI_nback_graded": "n-back", "COG-BCI_matb_graded": "MATB"}

COMMON = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
POST = [COMMON.index(c) for c in ["P7", "P8", "O1", "O2"]]
ALPHA_BAND = 2  # delta, theta, alpha, beta, gamma -> index 2

# --- PRE-DECLARED FDR FAMILY -------------------------------------------------
# Every inferential p-value the manuscript reports as evidence. Fixed before
# running. Keys must match those emitted by collect_pvalues().
# "effect" = the manuscript asserts an effect; losing significance BREAKS the claim.
# "null"   = the manuscript asserts NO association (the demand-not-skill argument);
#            here non-significance CONFIRMS the claim, and a surviving p would break it.
# The direction matters: a blanket "did not survive FDR = problem" would misread the
# deliberate null results as failures. FDR correction is applied to the whole family
# regardless -- only the INTERPRETATION differs.
FAMILY_KIND = {
    "spearman_subjectmean_n-back": "effect",
    "spearman_subjectmean_MATB": "effect",
    "spearman_epoch_n-back": "effect",
    "spearman_epoch_MATB": "effect",
    "decoder_vs_alpha_n-back": "effect",
    "decoder_vs_alpha_MATB": "effect",
    "eegmat_load_good_vs_poor": "null",
    "eegmat_load_vs_subtractions": "null",
    "eegmat_theta_good_vs_poor": "null",
    "eegmat_theta_vs_subtractions": "null",
    "eegmat_alpha_good_vs_poor": "null",
    "eegmat_alpha_vs_subtractions": "null",
}
FAMILY = list(FAMILY_KIND)


# ---------------------------------------------------------------- seeded model
def make_classifier(name: str, seed: int, class_weight=None):
    """Mirror of agents.make_classifier with the RNG threaded through."""
    if name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=seed)
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=class_weight)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    raise ValueError(f"unknown classifier: {name}")


def build_specialist(name: str, cfg: dict, seed: int, class_weight=None):
    clf = make_classifier(cfg["classifier"], seed, class_weight)
    if name == "spectral":
        return make_pipeline(StandardScaler(), clf)
    if name == "riemann":
        return make_pipeline(TangentSpace(metric="riemann"), StandardScaler(), clf)
    if name == "csp":
        return make_pipeline(CSP(nfilter=cfg.get("n_filters", 6), log=True), clf)
    raise ValueError(f"unknown specialist: {name}")


def build_decoder(seed: int, class_weight="balanced") -> AgenticDecoder:
    """Same architecture as agents.build_decoder, with a per-seed RNG."""
    specs = [
        SpecialistAgent(n, sc["view"], build_specialist(n, sc, seed, class_weight))
        for n, sc in CFG["specialists"].items()
    ]
    return AgenticDecoder(specs, CFG["orchestrator"])


# ------------------------------------------------------------------ resampling
def grouped_folds(groups: np.ndarray, k: int, rng: np.random.Generator):
    """GroupKFold-equivalent split with a SHUFFLED subject->fold assignment.

    sklearn's GroupKFold is deterministic (it sorts groups by size), so reruns
    reuse one partition and understate cross-subject variance. Subjects are
    shuffled here, then greedily assigned to the currently smallest fold to keep
    folds balanced - the same balancing objective, a different random partition.
    """
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    counts = {g: int((groups == g).sum()) for g in uniq}
    fold_sizes = np.zeros(k, dtype=int)
    assign = {}
    for g in sorted(uniq, key=lambda x: -counts[x]):
        f = int(np.argmin(fold_sizes))
        assign[g] = f
        fold_sizes[f] += counts[g]
    fold_of = np.array([assign[g] for g in groups])
    for f in range(k):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        if len(te) and len(tr):
            yield tr, te


def load_graded(name: str, cap: int, seed: int):
    d = np.load(FEAT / f"{name}.npz", allow_pickle=True)
    y = d["y"].astype(int)
    g = d["groups"]
    if cap and len(y) > cap:
        idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
        idx.sort()
        return {"psd": d["X_psd"][idx], "cov": d["X_cov"][idx]}, y[idx], g[idx]
    return {"psd": d["X_psd"], "cov": d["X_cov"]}, y, g


def oof_decode(views, y, g, seed: int, k: int = 8):
    """Cross-subject OOF probabilities -> continuous index + per-fold macro-F1."""
    rng = np.random.default_rng(seed)
    k = min(k, len(np.unique(g)))
    phi = np.full(len(y), np.nan)
    pred = np.full(len(y), -1)
    f1_orch = []
    f1_spec = {n: [] for n in CFG["specialists"]}
    for tr, te in grouped_folds(g, k, rng):
        vtr = {kk: v[tr] for kk, v in views.items()}
        vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(seed)
        dec.fit(vtr, y[tr], groups=g[tr])
        proba = dec.predict_proba(vte)
        cls = np.array(dec.classes_)
        phi[te] = proba @ cls
        pred[te] = cls[np.argmax(proba, 1)]
        f1_orch.append(f1_score(y[te], pred[te], average="macro"))
        for s in dec.specialists:
            sp = s.classes_[np.argmax(s.predict_proba(vte), 1)]
            f1_spec[s.name].append(f1_score(y[te], sp, average="macro"))
    return phi, pred, f1_orch, f1_spec


# ------------------------------------------------------------------- analyses
def run_graded(name: str, seed: int, cap: int):
    """Table 1: graded decoding + continuous index."""
    views, y, g = load_graded(name, cap, seed)
    phi, _, f1_orch, f1_spec = oof_decode(views, y, g, seed)

    per_level = {int(l): float(phi[y == l].mean()) for l in np.unique(y)}
    sp_ep = stats.spearmanr(phi, y)

    rows = [(phi[(g == s) & (y == l)].mean(), l)
            for s in np.unique(g) for l in np.unique(y) if ((g == s) & (y == l)).any()]
    sp_subj = stats.spearmanr([r[0] for r in rows], [r[1] for r in rows])

    return {
        "n_epochs": int(len(y)),
        "n_subjects": int(len(np.unique(g))),
        "n_folds": len(f1_orch),
        "macro_f1_orchestrator": float(np.mean(f1_orch)),
        "macro_f1_specialists": {k2: float(np.mean(v)) for k2, v in f1_spec.items()},
        "load_index_per_true_level": per_level,
        "monotonic_increasing": bool(all(np.diff(list(per_level.values())) > 0)),
        "spearman_epoch_rho": float(sp_ep.correlation),
        "spearman_epoch_p": float(sp_ep.pvalue),
        "spearman_subjectmean_rho": float(sp_subj.correlation),
        "spearman_subjectmean_p": float(sp_subj.pvalue),
        "ordinal_mae": float(np.mean(np.abs(np.rint(phi) - y))),
    }


def run_baseline(name: str, seed: int, cap: int):
    """Table 2: within-subject reliability + posterior-alpha baseline WITH a test."""
    views, y, g = load_graded(name, cap, seed)
    phi, _, _, _ = oof_decode(views, y, g, seed)

    alpha_idx = -views["psd"].reshape(-1, 5, 14)[:, ALPHA_BAND, :][:, POST].mean(1)

    rho_dec, rho_alp, monos = [], [], []
    for s in np.unique(g):
        m = g == s
        if len(np.unique(y[m])) < 2:
            continue
        rho_dec.append(float(stats.spearmanr(phi[m], y[m]).correlation))
        rho_alp.append(float(stats.spearmanr(alpha_idx[m], y[m]).correlation))
        per = [phi[m & (y == l)].mean() for l in np.unique(y) if (m & (y == l)).any()]
        monos.append(bool(np.all(np.diff(per) > 0)))
    rho_dec, rho_alp = np.array(rho_dec), np.array(rho_alp)

    # paired per-subject test: does the decoder track load better than alpha alone?
    valid = ~(np.isnan(rho_dec) | np.isnan(rho_alp))
    try:
        w = stats.wilcoxon(rho_dec[valid], rho_alp[valid])
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), 1.0

    pooled_dec = float(stats.spearmanr(phi, y).correlation)
    pooled_alp = float(stats.spearmanr(alpha_idx, y).correlation)

    return {
        "n_subjects": int(valid.sum()),
        "per_subject_rho_median": float(np.median(rho_dec)),
        "per_subject_rho_iqr": [float(np.percentile(rho_dec, 25)), float(np.percentile(rho_dec, 75))],
        "pct_subjects_positive": float((rho_dec > 0).mean() * 100),
        "pct_subjects_monotonic": float(np.mean(monos) * 100),
        "decoder_rho_pooled": pooled_dec,
        "alpha_rho_pooled": pooled_alp,
        "fold_change": float(pooled_dec / pooled_alp) if pooled_alp else float("nan"),
        "paired_median_rho_decoder": float(np.median(rho_dec[valid])),
        "paired_median_rho_alpha": float(np.median(rho_alp[valid])),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }


def run_montage(fam: str, seed: int, cap: int):
    """Table 3: 14-channel consumer montage vs full research montage, ONE run each."""
    f = FEAT / f"COG-BCI_{fam}_sess.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    y = d["y"].astype(int)
    g = np.array([f"s{s}" for s in d["subjects"]])

    if cap and len(y) > cap:
        ix = np.sort(np.random.default_rng(seed).choice(len(y), cap, replace=False))
    else:
        ix = np.arange(len(y))

    out = {"n_full_channels": int(len(list(d["full_channels"])))}
    for tag, pk, ck in (("consumer_14ch", "X_psd14", "X_cov14"), ("full_montage", "X_psdF", "X_covF")):
        views = {"psd": d[pk][ix], "cov": d[ck][ix]}
        phi, pred, f1, _ = oof_decode(views, y[ix], g[ix], seed)
        out[tag] = {
            "macro_f1": float(np.mean(f1)),
            "spearman": float(stats.spearmanr(phi, y[ix]).correlation),
            "ordinal_mae": float(np.mean(np.abs(np.rint(phi) - y[ix]))),
        }
    c, F = out["consumer_14ch"], out["full_montage"]
    out["retention"] = {
        "macro_f1_pct": 100 * c["macro_f1"] / F["macro_f1"],
        "spearman_pct": 100 * c["spearman"] / F["spearman"],
    }
    return out


def run_seed(seed: int, cap: int, analyses: set[str]) -> dict:
    res = {"seed": seed}
    if "graded" in analyses:
        res["graded"] = {lab: run_graded(n, seed, cap) for n, lab in GRADED.items()}
    if "baseline" in analyses:
        res["baseline"] = {lab: run_baseline(n, seed, cap) for n, lab in GRADED.items()}
    if "montage" in analyses:
        m = {fam: run_montage(fam, seed, cap) for fam in ("nback", "matb")}
        res["montage"] = {k: v for k, v in m.items() if v is not None}
    return res


# ------------------------------------------------------------ aggregate + FDR
def agg(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if not len(a):
        return None
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        "min": float(a.min()),
        "max": float(a.max()),
        "n_seeds": int(len(a)),
    }


def bh_fdr(pvals: dict, alpha: float = 0.05):
    """Benjamini-Hochberg. Returns per-key adjusted p and reject decision."""
    keys = [k for k, v in pvals.items() if v is not None and np.isfinite(v)]
    if not keys:
        return {}
    p = np.array([pvals[k] for k in keys], dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = min(prev, 1.0)
    return {keys[i]: {"p_raw": float(p[i]), "p_adj": float(adj[i]),
                      "significant_at_%.2f" % alpha: bool(adj[i] <= alpha)}
            for i in range(m)}


def collect_pvalues(per_seed: list[dict], eegmat: dict | None):
    """Median (and range) p per family member, across seeds."""
    raw: dict[str, list[float]] = {k: [] for k in FAMILY}
    for r in per_seed:
        for lab in GRADED.values():
            if "graded" in r and lab in r["graded"]:
                raw[f"spearman_subjectmean_{lab}"].append(r["graded"][lab]["spearman_subjectmean_p"])
                raw[f"spearman_epoch_{lab}"].append(r["graded"][lab]["spearman_epoch_p"])
            if "baseline" in r and lab in r["baseline"]:
                raw[f"decoder_vs_alpha_{lab}"].append(r["baseline"][lab]["wilcoxon_p"])
    if eegmat:
        pairs = [("eegmat_load", "assoc_decoded_load"), ("eegmat_theta", "assoc_theta_increase"),
                 ("eegmat_alpha", "assoc_alpha_desync")]
        for pre, key in pairs:
            blk = eegmat.get(key)
            if blk:
                raw[f"{pre}_good_vs_poor"].append(float(blk["GvsB_mwu_p"]))
                raw[f"{pre}_vs_subtractions"].append(float(blk["spearman_p"]))
    summary, medians = {}, {}
    for k, v in raw.items():
        vv = [x for x in v if x is not None and np.isfinite(x)]
        if not vv:
            continue
        med = float(np.median(vv))
        medians[k] = med
        summary[k] = {"median": med, "min": float(min(vv)), "max": float(max(vv)), "n": len(vv)}
    return medians, summary


# --------------------------------------------------------------------- output
def fmt(a, nd=3):
    """LaTeX: mean $\\pm$ sd."""
    if a is None:
        return "--"
    return f"{a['mean']:.{nd}f} $\\pm$ {a['sd']:.{nd}f}"


def fmt_plain(a, nd=3):
    """Markdown/plain: mean +/- sd."""
    if a is None:
        return "--"
    return f"{a['mean']:.{nd}f} +/- {a['sd']:.{nd}f}"


def write_tables(S, out: Path):
    L = []
    L.append("% Auto-generated by multiseed_fdr.py -- mean $\\pm$ SD across seeds.")
    L.append("% Replace the corresponding rows in MANUSCRIPT_ADAPTIVE.tex.\n")
    g = S.get("graded", {})
    if g:
        L.append("% ---- Table 1 (graded) ----")
        for lab in ("n-back", "MATB"):
            if lab not in g:
                continue
            b = g[lab]
            lvl = b.get("index_per_level", {})
            lvls = " / ".join(f"{lvl[k]['mean']:.2f}" for k in sorted(lvl)) if lvl else "--"
            L.append(f"{lab} & {fmt(b.get('macro_f1'))} & {lvls} & "
                     f"{fmt(b.get('spearman_subjectmean_rho'), 2)} & {fmt(b.get('ordinal_mae'))} \\\\")
        L.append("")
    m = S.get("montage", {})
    if m:
        L.append("% ---- Table 3 (montage) ----")
        for fam, lab in (("nback", "$n$-back"), ("matb", "MATB")):
            if fam not in m:
                continue
            b = m[fam]
            for tag, nm in (("consumer_14ch", "14-ch consumer"), ("full_montage", "58-ch research")):
                t = b.get(tag, {})
                L.append(f"{lab} & {nm} & {fmt(t.get('macro_f1'))} & "
                         f"{fmt(t.get('spearman'))} & {fmt(t.get('ordinal_mae'))} \\\\")
            r = b.get("retention", {})
            if r:
                L.append(f"% retention {fam}: macro-F1 {fmt(r.get('macro_f1_pct'), 1)}\\%, "
                         f"rho {fmt(r.get('spearman_pct'), 1)}\\%")
        L.append("")
    out.write_text("\n".join(L))


MANUSCRIPT = {  # current single-seed values, for the drift report
    "graded": {"n-back": {"macro_f1": 0.512, "spearman_subjectmean_rho": 0.485, "ordinal_mae": 0.579},
               "MATB": {"macro_f1": 0.615, "spearman_subjectmean_rho": 0.661, "ordinal_mae": 0.498}},
    "montage": {"nback": {"consumer_14ch": {"macro_f1": 0.508, "spearman": 0.353},
                          "full_montage": {"macro_f1": 0.529, "spearman": 0.423}},
                "matb": {"consumer_14ch": {"macro_f1": 0.613, "spearman": 0.539},
                         "full_montage": {"macro_f1": 0.652, "spearman": 0.608}}},
}


def write_report(S, fdr, psum, out: Path, n_seeds: int):
    L = [f"# Multi-seed + FDR report ({n_seeds} seeds)\n",
         "Values are mean +/- SD across seeds. `manuscript` is the current single-seed",
         "number in MANUSCRIPT_ADAPTIVE.tex. A value is FLAGGED when the manuscript",
         "number falls outside the multi-seed 95% interval -- it must then be updated.\n",
         "## Drift vs manuscript\n",
         "| quantity | manuscript | multi-seed | 95% interval | verdict |",
         "|---|---|---|---|---|"]
    for lab in ("n-back", "MATB"):
        b = S.get("graded", {}).get(lab, {})
        for key, nm in (("macro_f1", "macro-F1"), ("spearman_subjectmean_rho", "rho (subj-mean)"),
                        ("ordinal_mae", "ordinal MAE")):
            a = b.get(key)
            old = MANUSCRIPT["graded"].get(lab, {}).get(key)
            if a is None or old is None:
                continue
            lo, hi = a["ci95"]
            ok = lo <= old <= hi
            L.append(f"| {lab} {nm} | {old:.3f} | {a['mean']:.3f} +/- {a['sd']:.3f} | "
                     f"[{lo:.3f}, {hi:.3f}] | {'ok' if ok else '**FLAGGED**'} |")
    for fam, lab in (("nback", "n-back"), ("matb", "MATB")):
        b = S.get("montage", {}).get(fam, {})
        for tag, nm in (("consumer_14ch", "14ch"), ("full_montage", "58ch")):
            for key in ("macro_f1", "spearman"):
                a = b.get(tag, {}).get(key)
                old = MANUSCRIPT["montage"].get(fam, {}).get(tag, {}).get(key)
                if a is None or old is None:
                    continue
                lo, hi = a["ci95"]
                ok = lo <= old <= hi
                L.append(f"| {lab} {nm} {key} | {old:.3f} | {a['mean']:.3f} +/- {a['sd']:.3f} | "
                         f"[{lo:.3f}, {hi:.3f}] | {'ok' if ok else '**FLAGGED**'} |")

    L += ["\n## Hardware-retention claim (abstract says 94-96% and 83-89%)\n",
          "| paradigm | macro-F1 retained | rho retained |", "|---|---|---|"]
    for fam, lab in (("nback", "n-back"), ("matb", "MATB")):
        r = S.get("montage", {}).get(fam, {}).get("retention", {})
        if r:
            L.append(f"| {lab} | {fmt_plain(r.get('macro_f1_pct'), 1)} % "
                     f"| {fmt_plain(r.get('spearman_pct'), 1)} % |")
    L.append("\nIf these intervals no longer span 94-96 / 83-89, update the abstract, the")
    L.append("Results paragraph, AND the cover letter -- the claim appears in all three.\n")

    L += ["## Alpha-baseline comparison (new: with a significance test)\n",
          "| paradigm | decoder rho | alpha rho | fold-change | Wilcoxon p (median) |",
          "|---|---|---|---|---|"]
    for lab in ("n-back", "MATB"):
        b = S.get("baseline", {}).get(lab, {})
        if not b:
            continue
        pk = f"decoder_vs_alpha_{lab}"
        pm = psum.get(pk, {}).get("median")
        pcol = f"{pm:.2g}" if pm is not None else "--"
        L.append(f"| {lab} | {fmt_plain(b.get('decoder_rho_pooled'))} "
                 f"| {fmt_plain(b.get('alpha_rho_pooled'))} "
                 f"| {fmt_plain(b.get('fold_change'), 2)}x | {pcol} |")
    L.append("\nThe manuscript claims 2.5- to 3.4-fold. Update that range if it has moved.\n")

    L += ["## Benjamini-Hochberg FDR\n",
          f"Family declared before running ({len(FAMILY)} tests), corrected as one family.",
          "Per-seed p-values summarised by their median; max shown so fragile claims are visible.",
          "",
          "`kind` records what the manuscript claims. For **effect** rows, surviving FDR is",
          "required. For **null** rows (the demand-not-skill argument), NON-significance is",
          "the claim -- a surviving p there would contradict the manuscript instead.\n",
          "| test | kind | p median | p max | p adjusted | significant | verdict |",
          "|---|---|---|---|---|---|---|"]
    broken_effect, broken_null, fragile = [], [], []
    for k in FAMILY:
        if k not in fdr:
            continue
        d = fdr[k]
        s = psum.get(k, {})
        kind = FAMILY_KIND[k]
        sig = d["significant_at_0.05"]
        if kind == "effect":
            ok = sig
            if not ok:
                broken_effect.append(k)
            elif s.get("max", 0) > 0.05:
                fragile.append(k)
            verdict = "holds" if ok else "**CLAIM BROKEN**"
        else:
            ok = not sig
            if not ok:
                broken_null.append(k)
            verdict = "null confirmed" if ok else "**NULL BROKEN**"
        L.append(f"| {k} | {kind} | {d['p_raw']:.3g} | {s.get('max', float('nan')):.3g} | "
                 f"{d['p_adj']:.3g} | {'yes' if sig else 'no'} | {verdict} |")

    L.append("")
    if broken_effect:
        L.append("### Effects that did NOT survive FDR\n")
        L.append("Every sentence resting on these must be reworded as a trend, or dropped:\n")
        L += [f"- {k}" for k in broken_effect]
        L.append("")
    if fragile:
        L.append("### Fragile effects\n")
        L.append("Significant at the median p but not in every seed. Report as trends, "
                 "not established effects:\n")
        L += [f"- {k}" for k in fragile]
        L.append("")
    if broken_null:
        L.append("### Nulls that became significant\n")
        L.append("The manuscript argues the index reflects demand, NOT skill. If these are "
                 "significant, that argument no longer holds and Section 4.8 must be rewritten:\n")
        L += [f"- {k}" for k in broken_null]
        L.append("")
    if not (broken_effect or broken_null or fragile):
        L.append("All declared effects survive FDR and all declared nulls remain null. "
                 "No manuscript claim needs rewording on statistical grounds.\n")
    out.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (>=5 recommended)")
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--cap", type=int, default=9000, help="epoch subsample per run (0 = all)")
    ap.add_argument("--analyses", default="graded,baseline,montage")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel seeds (CPU cores, NOT GPU)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "results_v2"))
    ap.add_argument("--eegmat", default=None, help="path to eegmat_full.json for the FDR family")
    ap.add_argument("--resume", action="store_true", help="reuse existing per-seed checkpoints")
    args = ap.parse_args()

    _require_deps()

    analyses = {a.strip() for a in args.analyses.split(",") if a.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed_start + i for i in range(args.seeds)]

    if not FEAT.exists():
        sys.exit(f"[fatal] features not found: {FEAT}\n"
                 f"        restore the pipeline first:\n"
                 f"        git checkout HEAD -- 03_Features/ 04_Agentic/ 08_Learning_Transfer/")

    print(f"[info] seeds={seeds} analyses={sorted(analyses)} cap={args.cap} n_jobs={args.n_jobs}")
    print("[info] CPU-bound scikit-learn; a GPU contributes nothing. Scale --n-jobs with cores.", flush=True)

    todo = []
    per_seed = []
    for s in seeds:
        ck = out_dir / f"multiseed_seed{s}.json"
        if args.resume and ck.exists():
            per_seed.append(json.loads(ck.read_text()))
            print(f"[skip] seed {s} (checkpoint found)", flush=True)
        else:
            todo.append(s)

    def one(s):
        print(f"[run ] seed {s} ...", flush=True)
        r = run_seed(s, args.cap, analyses)
        (out_dir / f"multiseed_seed{s}.json").write_text(json.dumps(r, indent=2))
        print(f"[done] seed {s}", flush=True)
        return r

    if todo:
        done = None
        if args.n_jobs > 1:
            try:
                from joblib import Parallel, delayed
                done = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(one)(s) for s in todo)
            except Exception as exc:  # restricted sandbox, no joblib, low semaphore limits
                print(f"[warn] parallel backend unavailable ({type(exc).__name__}: {exc}).",
                      flush=True)
                print("[warn] falling back to sequential; per-seed checkpoints still written, "
                      "so --resume works if you interrupt.", flush=True)
                done = None
        if done is None:
            done = [one(s) for s in todo if not (out_dir / f"multiseed_seed{s}.json").exists()
                    or not args.resume]
        per_seed += done

    per_seed.sort(key=lambda r: r["seed"])

    # ---- aggregate
    S = {}
    if "graded" in analyses:
        S["graded"] = {}
        for lab in GRADED.values():
            blk = [r["graded"][lab] for r in per_seed if "graded" in r and lab in r["graded"]]
            if not blk:
                continue
            e = {k: agg([b[k] for b in blk]) for k in
                 ("macro_f1_orchestrator", "spearman_epoch_rho", "spearman_subjectmean_rho", "ordinal_mae")}
            e["macro_f1"] = e.pop("macro_f1_orchestrator")
            lv = sorted({int(k) for b in blk for k in b["load_index_per_true_level"]})
            e["index_per_level"] = {l: agg([b["load_index_per_true_level"][str(l)]
                                            if str(l) in b["load_index_per_true_level"]
                                            else b["load_index_per_true_level"][l] for b in blk]) for l in lv}
            e["monotonic_in_all_seeds"] = bool(all(b["monotonic_increasing"] for b in blk))
            S["graded"][lab] = e
    if "baseline" in analyses:
        S["baseline"] = {}
        for lab in GRADED.values():
            blk = [r["baseline"][lab] for r in per_seed if "baseline" in r and lab in r["baseline"]]
            if not blk:
                continue
            S["baseline"][lab] = {k: agg([b[k] for b in blk]) for k in
                                  ("decoder_rho_pooled", "alpha_rho_pooled", "fold_change",
                                   "pct_subjects_positive", "pct_subjects_monotonic",
                                   "per_subject_rho_median")}
    if "montage" in analyses:
        S["montage"] = {}
        for fam in ("nback", "matb"):
            blk = [r["montage"][fam] for r in per_seed if "montage" in r and fam in r["montage"]]
            if not blk:
                continue
            e = {}
            for tag in ("consumer_14ch", "full_montage"):
                e[tag] = {k: agg([b[tag][k] for b in blk]) for k in ("macro_f1", "spearman", "ordinal_mae")}
            e["retention"] = {k: agg([b["retention"][k] for b in blk])
                              for k in ("macro_f1_pct", "spearman_pct")}
            S["montage"][fam] = e

    # ---- FDR
    eeg = None
    cand = Path(args.eegmat) if args.eegmat else out_dir / "eegmat_full.json"
    if cand.exists():
        eeg = json.loads(cand.read_text())
        print(f"[info] EEGMAT p-values from {cand}", flush=True)
    else:
        print(f"[warn] {cand} not found - FDR family limited to the graded analyses.", flush=True)

    medians, psum = collect_pvalues(per_seed, eeg)
    fdr = bh_fdr(medians)

    (out_dir / "multiseed_summary.json").write_text(json.dumps(
        {"seeds": seeds, "n_seeds": len(per_seed), "analyses": sorted(analyses), "cap": args.cap,
         "aggregate": S, "pvalue_summary": psum, "fdr_family": FAMILY, "fdr": fdr}, indent=2))
    write_tables(S, out_dir / "multiseed_tables.tex")
    write_report(S, fdr, psum, out_dir / "multiseed_report.md", len(per_seed))

    print(f"\n-> {out_dir/'multiseed_summary.json'}")
    print(f"-> {out_dir/'multiseed_tables.tex'}")
    print(f"-> {out_dir/'multiseed_report.md'}   <- read this one first")
    broke = [k for k in FAMILY if k in fdr and FAMILY_KIND[k] == "effect"
             and not fdr[k]["significant_at_0.05"]]
    nulls = [k for k in FAMILY if k in fdr and FAMILY_KIND[k] == "null"
             and fdr[k]["significant_at_0.05"]]
    if broke:
        print(f"\n[ACTION] {len(broke)} claimed EFFECT(s) did not survive FDR: {', '.join(broke)}")
    if nulls:
        print(f"\n[ACTION] {len(nulls)} claimed NULL(s) became significant: {', '.join(nulls)}")
    if not (broke or nulls):
        print("\n[ok] All claimed effects survive FDR; all claimed nulls remain null.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

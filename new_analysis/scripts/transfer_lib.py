"""
transfer_lib.py — shared helpers for the cross-device transfer analysis.

All experiments in this folder import from here so that the decoder, the
transfer protocols, and the evaluation are defined in exactly one place.

Data source: pre-computed feature tensors (.npz) with keys
  X_psd (n,70), X_cov (n,14,14), X_tangent (n,105), y (n,), groups (n,)
located in:
  - 03_Features/features/   (COG-BCI, STEW, EEGMAT, combined)   [original pipeline]
  - new_analysis/features/  (Hinss_*)                            [this study]

Reproducibility: every stochastic step uses SEED = 42.
"""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

SEED = 42

# --- paths (resolved relative to the Pelaksanaan Penelitian root) ---
ROOT = Path(__file__).resolve().parents[2]          # .../Pelaksanaan Penelitian
import os as _os
# Dapat ditimpa lewat variabel lingkungan untuk menjalankan pada data v2 (COG-BCI n=21)
# tanpa mengubah perilaku bawaan. Tanpa variabel itu, jalurnya persis seperti semula.
FEAT_ORIG = ROOT / _os.environ.get("ND_FEAT_ORIG", "03_Features/features")
FEAT_NEW  = ROOT / "new_analysis" / "features"
DATA_OUT  = ROOT / _os.environ.get("ND_DATA_OUT", "new_analysis/data")
FIG_OUT   = ROOT / "new_analysis" / "figures"

# device family per dataset (research-grade multi-channel vs consumer Emotiv)
DEVICE = {"COG-BCI": "research", "Hinss": "research", "STEW": "consumer",
          "EEGMAT": "research"}


def feat_path(ds, scheme):
    base = FEAT_NEW if ds == "Hinss" else FEAT_ORIG
    return base / f"{ds}_{scheme}.npz"


def load(ds, scheme):
    """Return (X_psd, X_cov, X_tangent, y, groups) for a dataset/scheme."""
    d = np.load(feat_path(ds, scheme), allow_pickle=True)
    return (d["X_psd"].astype(np.float64), d["X_cov"].astype(np.float64),
            d["X_tangent"].astype(np.float64), d["y"].astype(int),
            d["groups"])


def subsample(n_total, cap, seed=SEED):
    """Deterministic index subsample (no replacement) or all indices."""
    if n_total <= cap:
        return np.arange(n_total)
    return np.random.RandomState(seed).choice(n_total, cap, replace=False)


def clf_pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def within_dataset_f1(X, y, groups, n_splits=5):
    """Cross-subject GroupKFold macro-F1 (mean, std)."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in gkf.split(X, y, groups):
        clf = clf_pipeline().fit(X[tr], y[tr])
        scores.append(f1_score(y[te], clf.predict(X[te]), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def transfer_f1(Xs, ys, Xt, yt):
    """Train on full source, test on full target; return (macro_f1, acc)."""
    clf = clf_pipeline().fit(Xs, ys)
    pred = clf.predict(Xt)
    return (float(f1_score(yt, pred, average="macro")),
            float(accuracy_score(yt, pred)))

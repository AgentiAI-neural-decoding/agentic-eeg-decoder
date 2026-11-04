"""Arsitektur Agentic AI — decoder multi-spesialis + orkestrator (Tahap 04).

Memetakan "orchestrator + spesialis" proposal ke decoder beban kognitif yang
dapat divalidasi offline:

- **SpecialistAgent**: satu decoder per "view" fitur dari Tahap 03
  (spectral=PSD, riemann=tangent-space, csp=spatial filter). Tiap agen adalah
  pipeline sklearn yang hanya melihat view-nya sendiri.
- **AgenticDecoder** (orkestrator): menggabungkan keputusan spesialis. Dua mode:
  * ``stacking``  — meta-classifier dilatih atas probabilitas spesialis, memakai
    **out-of-fold predictions** (anti-leakage) agar meta-learner tak melihat data
    yang sama dengan yang dipakai melatih spesialis.
  * ``soft_vote`` — rerata probabilitas spesialis (baseline tanpa parameter).

API mirip sklearn tetapi menerima ``views`` = dict {"psd": X_psd, "cov": X_cov}
karena tiap view berdimensi berbeda (vektor vs matriks kovarians).
"""
from __future__ import annotations

import numpy as np
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_classifier(name: str, class_weight=None):
    """Factory base-classifier (selaras proposal: SVM, RF/LDA, LogReg)."""
    if name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=42)
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=class_weight)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, class_weight=class_weight, random_state=42, n_jobs=-1)
    raise ValueError(f"classifier tak dikenal: {name}")


def build_specialist(name: str, cfg: dict, class_weight=None):
    """Bangun pipeline spesialis untuk satu view fitur."""
    clf = make_classifier(cfg["classifier"], class_weight)
    if name == "spectral":
        return make_pipeline(StandardScaler(), clf)
    if name == "riemann":
        return make_pipeline(TangentSpace(metric="riemann"), StandardScaler(), clf)
    if name == "csp":
        return make_pipeline(CSP(nfilter=cfg.get("n_filters", 6), log=True), clf)
    raise ValueError(f"spesialis tak dikenal: {name}")


class SpecialistAgent:
    """Pembungkus pipeline spesialis + view yang dikonsumsinya."""

    def __init__(self, name: str, view: str, pipeline):
        self.name = name
        self.view = view
        self.pipeline = pipeline
        self.classes_ = None

    def fit(self, views: dict, y):
        self.pipeline.fit(views[self.view], y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict_proba(self, views: dict) -> np.ndarray:
        return self.pipeline.predict_proba(views[self.view])


def _oof_proba(pipeline, X, y, groups, n_splits, classes) -> np.ndarray:
    """Out-of-fold predict_proba (anti-leakage) — manual agar kolom kelas konsisten."""
    oof = np.zeros((len(y), len(classes)), dtype=float)
    cls_idx = {c: i for i, c in enumerate(classes)}
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y)
    from sklearn.base import clone
    for tr, te in splitter:
        p = clone(pipeline).fit(X[tr], y[tr])
        proba = p.predict_proba(X[te])
        for j, c in enumerate(p.classes_):
            oof[te, cls_idx[c]] = proba[:, j]
    return oof


class AgenticDecoder:
    """Orkestrator multi-agen. ``fit(views, y, groups)`` / ``predict(views)``."""

    def __init__(self, specialists: list[SpecialistAgent], orch_cfg: dict):
        self.specialists = specialists
        self.orch_cfg = orch_cfg
        self.meta = None
        self.classes_ = None

    def fit(self, views: dict, y, groups=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.orch_cfg["type"] == "stacking":
            # 1) meta-fitur = OOF proba tiap spesialis (anti-leakage)
            n_splits = self.orch_cfg.get("internal_cv_folds", 3)
            meta_X = np.hstack([
                _oof_proba(s.pipeline, views[s.view], y, groups, n_splits, self.classes_)
                for s in self.specialists
            ])
            self.meta = make_classifier(self.orch_cfg.get("meta_classifier", "logreg"),
                                        class_weight="balanced")
            self.meta.fit(meta_X, y)
        # 2) refit tiap spesialis pada SELURUH data latih
        for s in self.specialists:
            s.fit(views, y)
        return self

    def _stack_proba(self, views: dict) -> np.ndarray:
        return np.hstack([s.predict_proba(views) for s in self.specialists])

    def predict_proba(self, views: dict) -> np.ndarray:
        if self.orch_cfg["type"] == "stacking":
            return self.meta.predict_proba(self._stack_proba(views))
        # soft_vote: rerata proba spesialis
        probas = [s.predict_proba(views) for s in self.specialists]
        return np.mean(probas, axis=0)

    def predict(self, views: dict) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(views), axis=1)]


def build_decoder(cfg: dict, class_weight=None) -> AgenticDecoder:
    """Rakit AgenticDecoder dari config."""
    specs = []
    for name, sc in cfg["specialists"].items():
        if sc.get("enabled", True):
            specs.append(SpecialistAgent(name, sc["view"], build_specialist(name, sc, class_weight)))
    return AgenticDecoder(specs, cfg["orchestrator"])

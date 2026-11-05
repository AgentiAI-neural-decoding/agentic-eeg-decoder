"""
riemann_lib.py — Riemannian covariance transfer helpers (pyriemann).

Re-centering (Zanini et al. 2018): transport each domain's covariances by
M^{-1/2}, where M is the Riemannian (geometric) mean of that domain, so that
each domain is whitened to identity before tangent-space projection.
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
from pyriemann.tangentspace import TangentSpace
from transfer_lib import clf_pipeline, SEED
from sklearn.metrics import f1_score


def recenter(covs):
    """Whiten a set of covariances to identity by their Riemannian mean."""
    M = mean_riemann(covs)
    W = invsqrtm(M)
    return np.array([W @ c @ W for c in covs])


def tangent_transfer(Xs_cov, ys, Xt_cov, yt, do_recenter):
    """
    Tangent-space transfer, optionally with per-domain re-centering.
    Fits the tangent projection on the (possibly re-centered) source.
    Returns macro-F1 on the target.
    """
    Xs, Xt = Xs_cov, Xt_cov
    if do_recenter:
        Xs = recenter(Xs)
        Xt = recenter(Xt)
    ts = TangentSpace().fit(Xs)
    clf = clf_pipeline().fit(ts.transform(Xs), ys)
    return float(f1_score(yt, clf.predict(ts.transform(Xt)), average="macro"))

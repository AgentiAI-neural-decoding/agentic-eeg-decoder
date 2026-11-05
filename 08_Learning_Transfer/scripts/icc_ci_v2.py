#!/usr/bin/env python3
"""Selang kepercayaan ICC(2,1) untuk indeks beban lintas-sesi COG-BCI (n = 21, v2).

Menjawab butir M4 telaah Modul 7: `icc_montage.json` hanya menyimpan titik taksiran,
sehingga selangnya tidak dapat dihitung dari berkas yang ada. Skrip ini membangun ulang
matriks subjek x sesi memakai jalur yang sama persis dengan `icc_montage_v2.py`, lalu:

  1. menyimpan matriksnya (agar selang tidak perlu dihitung ulang lagi selamanya),
  2. menghitung ICC(2,1) beserta komponen ragamnya (MSR, MSC, MSE),
  3. menghitung selang 95 persen eksak berbasis F (McGraw & Wong 1996, ICC(A,1)),
  4. menghitung selang 95 persen bootstrap atas subjek sebagai pemeriksaan silang,
     yang penting di sini karena dua dari empat taksiran bernilai negatif dan selang
     berbasis F dapat berperilaku ganjil di dekat nol.

Tidak menimpa berkas mana pun. Keluaran: results_v2/icc_montage_ci.json dan
results_v2/icc_matrices.npz
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "04_Agentic" / "scripts"))
from agents import build_decoder  # noqa: E402

FEAT = BASE / "03_Features" / "features_v2"
RES = Path(__file__).resolve().parent.parent / "results_v2"
CFG = {"specialists": {"spectral": {"enabled": True, "view": "psd", "classifier": "svm_rbf"},
                       "riemann": {"enabled": True, "view": "cov", "classifier": "logreg"},
                       "csp": {"enabled": True, "view": "cov", "classifier": "lda", "n_filters": 6}},
       "orchestrator": {"type": "stacking", "meta_classifier": "logreg", "internal_cv_folds": 3}}
ALPHA = 0.05
NBOOT = 5000


def icc21_full(M):
    """ICC(2,1) dua-arah acak, ukuran tunggal, beserta komponen ragamnya."""
    n, k = M.shape
    gm = M.mean()
    MSR = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    MSC = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    resid = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm
    MSE = (resid ** 2).sum() / ((n - 1) * (k - 1))
    icc = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    return float(icc), float(MSR), float(MSC), float(MSE), n, k


def ci_f(M, alpha=ALPHA):
    """Selang eksak berbasis F untuk ICC(A,1) (McGraw & Wong 1996)."""
    icc, MSR, MSC, MSE, n, k = icc21_full(M)
    if not np.isfinite(icc) or abs(1 - icc) < 1e-12:
        return None, None
    a = (k * icc) / (n * (1 - icc))
    b = 1 + (k * icc * (n - 1)) / (n * (1 - icc))
    den = (a * MSC) ** 2 / (k - 1) + (b * MSE) ** 2 / ((n - 1) * (k - 1))
    if den <= 0:
        return None, None
    v = (a * MSC + b * MSE) ** 2 / den
    fl = stats.f.ppf(1 - alpha / 2, n - 1, v)
    fu = stats.f.ppf(1 - alpha / 2, v, n - 1)
    lo = (n * (MSR - fl * MSE)) / (fl * (k * MSC + (k * n - k - n) * MSE) + n * MSR)
    up = (n * (fu * MSR - MSE)) / (k * MSC + (k * n - k - n) * MSE + n * fu * MSR)
    return float(lo), float(up)


def ci_boot(M, nboot=NBOOT, alpha=ALPHA, seed=20260905):
    """Selang persentil bootstrap dengan pengambilan ulang atas SUBJEK (baris)."""
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    vals = []
    for _ in range(nboot):
        ix = rng.integers(0, n, n)
        Mb = M[ix]
        if np.allclose(Mb.std(0), 0) or np.allclose(Mb.std(1), 0):
            continue
        v = icc21_full(Mb)[0]
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < nboot * 0.5:
        return None, None, len(vals)
    vals = np.array(vals)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))), len(vals))


def oof_index(views, y, g, cap=8000, k=8):
    if cap and len(y) > cap:
        ix = np.sort(np.random.default_rng(42).choice(len(y), cap, replace=False))
        views = {kk: v[ix] for kk, v in views.items()}; y = y[ix]; g = g[ix]; sub_idx = ix
    else:
        sub_idx = np.arange(len(y))
    phi = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(k, len(np.unique(g)))).split(views["psd"], y, g):
        vtr = {kk: v[tr] for kk, v in views.items()}; vte = {kk: v[te] for kk, v in views.items()}
        dec = build_decoder(CFG, "balanced"); dec.fit(vtr, y[tr], groups=g[tr])
        phi[te] = dec.predict_proba(vte) @ np.array(dec.classes_)
    return phi, y, g, sub_idx


def run(fam, store):
    d = np.load(FEAT / f"COG-BCI_{fam}_sess.npz", allow_pickle=True)
    y = d["y"].astype(int); subj = d["subjects"]; sess = d["sessions"]
    g = np.array([f"s{s}" for s in subj])
    v14 = {"psd": d["X_psd14"], "cov": d["X_cov14"]}
    phi, y2, g2, ix = oof_index(v14, y, g)
    subj2, sess2 = subj[ix], sess[ix]
    sessions = sorted(set(sess2)); subs = sorted(set(subj2))

    def mat(metric):
        M = np.full((len(subs), len(sessions)), np.nan)
        for i, s in enumerate(subs):
            for j, se in enumerate(sessions):
                m = (subj2 == s) & (sess2 == se)
                if m.sum() == 0:
                    continue
                if metric == "mean":
                    M[i, j] = phi[m].mean()
                else:
                    hi = phi[m & (y2 == y2.max())]; lo = phi[m & (y2 == y2.min())]
                    if len(hi) and len(lo):
                        M[i, j] = hi.mean() - lo.mean()
        return M

    out = {}
    for metric in ("mean", "contrast"):
        M = mat(metric)
        keep = ~np.isnan(M).any(1)
        Mk = M[keep]
        if keep.sum() < 5:
            continue
        store[f"{fam}_{metric}"] = Mk
        store[f"{fam}_{metric}_subjects"] = np.array([subs[i] for i in np.where(keep)[0]], dtype=object)
        icc, MSR, MSC, MSE, n, k = icc21_full(Mk)
        lo_f, up_f = ci_f(Mk)
        lo_b, up_b, nb = ci_boot(Mk)
        out[metric] = {
            "icc21": round(icc, 3),
            "n_subjects": int(n), "n_sessions": int(k),
            "ci95_F": [None if lo_f is None else round(lo_f, 3),
                       None if up_f is None else round(up_f, 3)],
            "ci95_bootstrap": [None if lo_b is None else round(lo_b, 3),
                               None if up_b is None else round(up_b, 3)],
            "n_bootstrap_sah": nb,
            "komponen_ragam": {"MSR": round(MSR, 6), "MSC": round(MSC, 6), "MSE": round(MSE, 6)},
        }
        print(f"  {fam}/{metric}: ICC={icc:.3f}  CI-F=[{lo_f}, {up_f}]  CI-boot=[{lo_b}, {up_b}]  n={n},k={k}", flush=True)
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    store, out = {}, {}
    for fam in ("nback", "matb"):
        f = FEAT / f"COG-BCI_{fam}_sess.npz"
        if not f.exists():
            print(f"[lewat] {fam}: {f.name} tidak ada", flush=True); continue
        print(f"=== {fam} ===", flush=True)
        out[fam] = run(fam, store)
    meta = {"dibuat": "2026-09-05", "sumber_fitur": str(FEAT),
            "catatan": ("Selang kepercayaan untuk nilai ICC yang sudah dilaporkan "
                        "icc_montage.json. Jalur perhitungan indeks identik dengan "
                        "icc_montage_v2.py (cap 8000, seed 42, GroupKFold 8). "
                        "Selang F mengikuti McGraw & Wong 1996 untuk ICC(A,1); "
                        "bootstrap persentil 5000 ulangan atas subjek."),
            "alpha": ALPHA, "n_bootstrap": NBOOT}
    (RES / "icc_montage_ci.json").write_text(json.dumps({"meta": meta, **out}, indent=2))
    np.savez(RES / "icc_matrices.npz", **store)
    print(f"\n-> {RES/'icc_montage_ci.json'}", flush=True)
    print(f"-> {RES/'icc_matrices.npz'}", flush=True)
    print("SELESAI", flush=True)


if __name__ == "__main__":
    sys.exit(main())

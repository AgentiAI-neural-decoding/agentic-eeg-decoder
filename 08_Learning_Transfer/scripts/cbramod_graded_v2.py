"""cbramod_graded_v2.py - baseline foundation model untuk indeks beban bertingkat (COG-BCI n=21).

Menjawab "mengapa tidak memakai foundation model?" dengan protokol yang sama dengan
graded_v2.py: GroupKFold-8 lintas subjek, subsampel 9000 epoch (seed 42), indeks kontinu
E[level] = sum_k k * P(k), Spearman tingkat epoch dan rerata subjek, macro-F1 3 kelas.
Encoder: CBraMod pretrained (braindecode/cbramod-pretrained), BEKU, norm z-score, mean pooling,
14 kanal Emotiv, 200 Hz, 2 s. Probe: StandardScaler + LogisticRegression (balanced).
Kontrol: arsitektur sama dengan bobot acak (tanpa pretraining).
Input: 02_Preprocessing/derivatives_v2/COG-BCI (21 subjek unik terverifikasi).
Output: 08_Learning_Transfer/results_v2/cbramod_graded.json
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import mne
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "06_FoundationModel" / "scripts"))
from fm_encoder import build_encoder, resolve_device, _parse  # noqa: E402
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
CFG = json.loads((BASE / "06_FoundationModel" / "config" / "fm_v2.json").read_text())
DERIV = BASE / "02_Preprocessing" / "derivatives_v2" / "COG-BCI"
RES = BASE / "08_Learning_Transfer" / "results_v2"
FAMILIES = {"COG-BCI_matb_graded": {"MATBeasy": 0, "MATBmed": 1, "MATBdiff": 2},
            "COG-BCI_nback_graded": {"zeroBACK": 0, "oneBACK": 1, "twoBACK": 2}}
CAP, SEED, K = 9000, 42, 8


def load_family(levels, common, sfreq):
    Xs, ys, gs = [], [], []
    for f in sorted(DERIV.rglob("*_proc-clean_epo.fif")):
        meta = _parse(f, "COG-BCI")
        if meta["task"] not in levels:
            continue
        ep = mne.read_epochs(f).pick(common); ep.reorder_channels(common)
        if abs(ep.info["sfreq"] - sfreq) > 1e-3:
            ep.resample(sfreq)
        d = ep.get_data(copy=False).astype("float32")
        Xs.append(d); ys += [levels[meta["task"]]] * len(d); gs += ["COG-BCI-" + meta["subject"]] * len(d)
    X = np.concatenate(Xs); y = np.array(ys); g = np.array(gs)
    if len(y) > CAP:
        idx = np.random.default_rng(SEED).choice(len(y), CAP, replace=False)
        X, y, g = X[idx], y[idx], g[idx]
    return X, y, g


def embed(X, opts, device, random_init):
    enc = build_encoder("cbramod", opts, X.shape[1], device)
    if random_init:
        from braindecode.models import CBraMod
        fresh = CBraMod(n_chans=X.shape[1], n_times=opts.get("n_times", 400), sfreq=200,
                        n_outputs=2).to(device).eval()
        for p in fresh.parameters():
            p.requires_grad_(False)
        enc.model = fresh
    E = np.concatenate([enc.embed(X[i:i + 512]) for i in range(0, len(X), 512)])
    del enc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return E


def graded_probe(E, y, g):
    gkf = GroupKFold(min(K, len(np.unique(g))))
    idx_cont = np.full(len(y), np.nan); pred = np.full(len(y), -1); f1s = []
    for tr, te in gkf.split(E, y, g):
        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(E[tr]), y[tr])
        pr = clf.predict_proba(sc.transform(E[te])); cls = clf.classes_
        idx_cont[te] = pr @ cls; pred[te] = cls[np.argmax(pr, 1)]
        f1s.append(f1_score(y[te], pred[te], average="macro"))
    per_level = {int(l): round(float(idx_cont[y == l].mean()), 3) for l in np.unique(y)}
    sp_ep = stats.spearmanr(idx_cont, y)
    rows = [(idx_cont[(g == s) & (y == l)].mean(), l) for s in np.unique(g) for l in np.unique(y)
            if ((g == s) & (y == l)).any()]
    sp_s = stats.spearmanr([r[0] for r in rows], [r[1] for r in rows])
    per_subj = []
    for s in np.unique(g):
        m = g == s
        if len(np.unique(y[m])) > 1:
            per_subj.append(stats.spearmanr(idx_cont[m], y[m]).correlation)
    per_subj = np.array(per_subj)
    return {"n_epochs": int(len(y)), "n_subjects": int(len(np.unique(g))), "n_folds": len(f1s),
            "macro_f1": round(float(np.mean(f1s)), 3), "chance_macro_f1": round(1 / len(np.unique(y)), 3),
            "load_index_per_true_level": per_level,
            "monotonic_increasing": bool(all(np.diff(list(per_level.values())) > 0)),
            "spearman_index_vs_level_epoch": {"rho": round(float(sp_ep.correlation), 3), "p": float(sp_ep.pvalue)},
            "spearman_index_vs_level_subjectmean": {"rho": round(float(sp_s.correlation), 3), "p": round(float(sp_s.pvalue), 4)},
            "per_subject_rho_median": round(float(np.median(per_subj)), 3),
            "pct_subjects_positive": round(float(100 * np.mean(per_subj > 0)), 1)}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    dev = resolve_device("auto")
    opts = {**CFG["encoder_options"]["cbramod"], "norm": "zscore", "pooling": "mean"}
    common = CFG["common_channels"]; sfreq = opts.get("sfreq", 200)
    out = {"method": __doc__, "device": dev}
    for fam, levels in FAMILIES.items():
        t0 = time.time()
        X, y, g = load_family(levels, common, sfreq)
        print(f"== {fam}: X={X.shape} subjects={len(np.unique(g))} ({time.time()-t0:.0f}s load)", flush=True)
        out[fam] = {}
        for tag, ri in (("cbramod_pretrained", False), ("cbramod_randominit", True)):
            t1 = time.time(); E = embed(X, opts, dev, ri); r = graded_probe(E, y, g)
            out[fam][tag] = r
            print(f"  {tag:20s} F1={r['macro_f1']} rho_subj={r['spearman_index_vs_level_subjectmean']} ({time.time()-t1:.0f}s)", flush=True)
        (RES / "cbramod_graded.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES / 'cbramod_graded.json'}", flush=True)


if __name__ == "__main__":
    main()

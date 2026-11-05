"""13_hinss_leakfree.py - transfer COG-BCI <-> Hinss tanpa kebocoran subjek.

Latar: 12_overlap_hinss_cogbci.py membuktikan Hinss P01-P15 = rekaman COG-BCI sub-01..15
(26/30 rekaman cocok subjek dan sesi). Transfer COG-BCI <-> Hinss lama karena itu menguji
model pada orang yang sama. Skrip ini memisahkan:
  - COG_all  : seluruh 21 subjek COG-BCI bersih (pembanding angka lama)
  - COG_ov   : sub-01..15 (tumpang-tindih dengan Hinss)
  - COG_free : sub-16,17,25,26,28,29 (tidak ada di Hinss) -> transfer bebas kebocoran
Kontrol ukuran: 30 undian acak 6 subjek dari COG_ov, agar beda COG_free vs COG_ov tidak
tercampur efek jumlah subjek. Rujukan pada sumber/target COG_free yang sama:
device-matched independen (ds003838) dan cross-device (STEW).
Fitur PSD 70-dim, LogReg (transfer_lib.transfer_f1). Jalankan dengan
ND_FEAT_ORIG=03_Features/features_v2 ND_DATA_OUT=new_analysis/data_v2.
Output: $ND_DATA_OUT/hinss_leakfree.json
"""
import json
import numpy as np
from transfer_lib import load, transfer_f1, DATA_OUT, SEED

SCHEMES = ["workload_extreme", "workload_binary"]
OV = {f"COG-BCI-{i:02d}" for i in range(1, 16)}
N_DRAW = 30


def sub(ds, scheme, keep=None):
    Xp, _, _, y, g = load(ds, scheme)
    if keep is None:
        return Xp, y, g
    m = np.isin(g, list(keep))
    return Xp[m], y[m], g[m]


def tr(a, b):
    f1, acc = transfer_f1(a[0], a[1], b[0], b[1])
    return {"f1": round(f1, 4), "acc": round(acc, 4),
            "n_src_subj": int(len(np.unique(a[2]))), "n_tgt_subj": int(len(np.unique(b[2])))}


def main():
    rng = np.random.RandomState(SEED)
    out = {"method": __doc__}
    for scheme in SCHEMES:
        cog_all = sub("COG-BCI", scheme)
        subs = np.unique(cog_all[2])
        free = sorted(set(subs) - OV)
        ov = sorted(set(subs) & OV)
        cog_ov, cog_free = sub("COG-BCI", scheme, ov), sub("COG-BCI", scheme, free)
        hin, stew, ds = sub("Hinss", scheme), sub("STEW", scheme), sub("ds003838", scheme)
        r = {"free_subjects": free,
             "COG_all->Hinss": tr(cog_all, hin), "Hinss->COG_all": tr(hin, cog_all),
             "COG_ov->Hinss": tr(cog_ov, hin), "Hinss->COG_ov": tr(hin, cog_ov),
             "COG_free->Hinss": tr(cog_free, hin), "Hinss->COG_free": tr(hin, cog_free),
             "COG_free->ds003838": tr(cog_free, ds), "ds003838->COG_free": tr(ds, cog_free),
             "COG_free->STEW": tr(cog_free, stew), "STEW->COG_free": tr(stew, cog_free)}
        fw, bw = [], []
        for _ in range(N_DRAW):
            pick = list(rng.choice(ov, size=len(free), replace=False))
            s = sub("COG-BCI", scheme, pick)
            fw.append(transfer_f1(s[0], s[1], hin[0], hin[1])[0])
            bw.append(transfer_f1(hin[0], hin[1], s[0], s[1])[0])
        r["size_matched_COG_ov6->Hinss"] = {"mean_f1": round(float(np.mean(fw)), 4), "sd": round(float(np.std(fw)), 4), "n_draws": N_DRAW}
        r["size_matched_Hinss->COG_ov6"] = {"mean_f1": round(float(np.mean(bw)), 4), "sd": round(float(np.std(bw)), 4), "n_draws": N_DRAW}
        out[scheme] = r
        print(f"\n== {scheme} (free={free})", flush=True)
        for k, v in r.items():
            if k != "free_subjects":
                print(f"  {k:30s} {v}", flush=True)
    DATA_OUT.mkdir(exist_ok=True)
    (DATA_OUT / "hinss_leakfree.json").write_text(json.dumps(out, indent=2))
    print("\n-> " + str(DATA_OUT / "hinss_leakfree.json"), flush=True)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()

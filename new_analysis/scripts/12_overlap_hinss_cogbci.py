"""Uji tumpang-tindih rekaman Hinss2021 (pBCI hackathon) vs COG-BCI (Zenodo).

Untuk tiap rekaman Hinss MATBeasy (15 peserta x 3 sesi), 60 s pertama kanal Fz/Pz/Oz
(setelah rata-rata referensi atas 61 kanal yang sama di kedua dataset) dicocokkan ke seluruh rekaman MATBeasy COG-BCI (29 folder x 3 sesi) dengan korelasi
silang ternormalisasi (sinyal disamakan ke 250 Hz, high-pass 1 Hz). Rekaman yang sama
menghasilkan r mendekati 1 pada satu lag; rekaman berbeda menghasilkan r kecil.
Output: new_analysis/data_v2/overlap_hinss_cogbci.json
"""
import json, os, warnings
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt, resample_poly, correlate

warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
ROOT = os.path.expanduser("~/neuraldecoding/01_Akuisisi_Data")
HIN = f"{ROOT}/raw/Hinss2021_PassiveBCI"
COG = f"{ROOT}/raw_v2/COG-BCI"
CH = ["Fz", "Pz", "Oz"]
FS = 250.0
SOS = butter(4, 1.0, "highpass", fs=FS, output="sos")
TPL = int(60 * FS)


def prep(x):
    x = sosfiltfilt(SOS, x, axis=-1)
    return x - x.mean(axis=-1, keepdims=True)


def ncc(sig, tpl):
    m = tpl.shape[-1]
    t = (tpl - tpl.mean()) / (np.linalg.norm(tpl - tpl.mean()) + 1e-12)
    num = correlate(sig, t, mode="valid", method="fft")
    c1 = np.concatenate([[0], np.cumsum(sig)]); c2 = np.concatenate([[0], np.cumsum(sig ** 2)])
    s1 = c1[m:] - c1[:-m]; s2 = c2[m:] - c2[:-m]
    den = np.sqrt(np.maximum(s2 - s1 ** 2 / m, 1e-12))
    r = num / den
    k = int(np.argmax(r)); return float(r[k]), k


cog = {}
for sub in sorted(d for d in os.listdir(COG) if d.startswith("sub-")):
    for ses in ("ses-S1", "ses-S2", "ses-S3"):
        p = f"{COG}/{sub}/{ses}/eeg/MATBeasy.set"
        if not os.path.exists(p):
            continue
        r = mne.io.read_raw_eeglab(p, preload=True).drop_channels([c for c in ("ECG1", "TP10") if c in mne.io.read_raw_eeglab(p).ch_names])
        r.set_eeg_reference("average", projection=False)
        x = r.pick(CH).get_data()
        if r.info["sfreq"] != FS:
            x = resample_poly(x, int(FS), int(r.info["sfreq"]), axis=-1)
        cog[f"{sub}/{ses}"] = prep(x)
print("COG recordings:", len(cog), flush=True)

out = {"method": __doc__, "n_cog_recordings": len(cog), "hinss": {}}
for P in sorted(d for d in os.listdir(HIN) if d.startswith("P")):
    for S in ("S1", "S2", "S3"):
        n = int(P[1:])
        p = f"{HIN}/{P}/{S}/eeg/alldata_sbj{n:02d}_sess{S[1]}_MATBeasy.set"
        if not os.path.exists(p):
            continue
        e = mne.io.read_epochs_eeglab(p)
        e.set_eeg_reference("average", projection=False)
        x = np.concatenate(list(e.pick(CH).get_data()), axis=-1)
        if e.info["sfreq"] != FS:
            x = resample_poly(x, int(FS), int(e.info["sfreq"]), axis=-1)
        tpl = prep(x)[:, :TPL]
        scores = []
        for key, sig in cog.items():
            if sig.shape[-1] < TPL:
                continue
            rs = [ncc(sig[c], tpl[c]) for c in range(len(CH))]
            scores.append((float(np.mean([a for a, _ in rs])), key, [round(a, 4) for a, _ in rs], [b for _, b in rs]))
        scores.sort(reverse=True)
        best, second = scores[0], scores[1]
        same = f"sub-{n:02d}/ses-{S}"
        rank = next((i + 1 for i, t in enumerate(scores) if t[1] == same), None)
        r_same = next((round(t[0], 4) for t in scores if t[1] == same), None)
        out["hinss"][f"{P}/{S}"] = {"best_match": best[1], "r_mean": round(best[0], 4), "r_per_channel": best[2],
                                   "lag_samples": best[3], "second_match": second[1], "r_second": round(second[0], 4),
                                   "same_id_candidate": same, "same_id_rank": rank, "r_same_id": r_same,
                                   "median_r_all": round(float(np.median([t[0] for t in scores])), 4)}
        print(f"{P}/{S} -> {best[1]} r={best[0]:.3f} (2nd {second[1]} r={second[0]:.3f}) | same-id rank {rank} r={r_same}", flush=True)

H = out["hinss"].values()
out["summary"] = {"n_hinss": len(H), "best_is_same_id": sum(h["best_match"] == h["same_id_candidate"] for h in H),
                  "chance_rate": 1 / len(cog)}
print(out["summary"])
with open(os.path.expanduser("~/neuraldecoding/new_analysis/data_v2/overlap_hinss_cogbci.json"), "w") as f:
    json.dump(out, f, indent=2)
print("-> new_analysis/data_v2/overlap_hinss_cogbci.json")

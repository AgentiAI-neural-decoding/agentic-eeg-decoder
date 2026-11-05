"""make_features_v3.py - build features_v3: COG-BCI with every shared-recording ID removed.

Exclusion rule (fixed before re-running any analysis): every participant ID that shares
at least one byte-identical recording with another ID is excluded, except the
lowest-numbered ID of each cluster. In the Zenodo release this keeps sub-17 (cluster
17-24, 27) and sub-25 (sub-25/sub-28 share session S1), giving 20 participants.
features_v2 already excluded sub-18..24 and 27; this step additionally removes sub-28.

COG-BCI arrays are filtered row-wise. X_tangent (global tangent space, exploratory) is
recomputed on the filtered covariances with the same recipe as extract_features.py
(pyriemann TangentSpace, metric="riemann"). Files of other datasets are hard-linked
unchanged. features_v2 is not modified.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
from pyriemann.tangentspace import TangentSpace

BASE = Path(__file__).resolve().parent.parent.parent
SRC = BASE / "03_Features" / "features_v2"
DST = BASE / "03_Features" / "features_v3"
DROP = {"28"}


def main() -> None:
    DST.mkdir(exist_ok=True)
    log = {"rule": __doc__.strip().splitlines()[2:6], "dropped_subjects": sorted(DROP), "files": {}}
    for p in sorted(SRC.glob("*.npz")):
        out = DST / p.name
        if p.name.startswith("combined_"):
            continue
        if not p.name.startswith("COG-BCI_"):
            if not out.exists():
                os.link(p, out)
            log["files"][p.name] = "hard link (unchanged)"
            continue
        d = np.load(p, allow_pickle=True)
        keep = ~np.isin(d["subjects"].astype(str), list(DROP))
        arrays = {k: (d[k][keep] if d[k].shape[:1] == keep.shape else d[k]) for k in d.files}
        if "X_tangent" in arrays and len(arrays["X_cov"]):
            arrays["X_tangent"] = TangentSpace(metric="riemann").fit_transform(arrays["X_cov"]).astype(np.float32)
        np.savez_compressed(out, **arrays)
        subs = np.unique(arrays["subjects"].astype(str))
        log["files"][p.name] = {"rows_before": int(len(keep)), "rows_after": int(keep.sum()),
                                "n_subjects": int(len(subs)), "subjects": subs.tolist()}
        print(f"{p.name}: {len(keep)} -> {int(keep.sum())} rows, {len(subs)} subjects", flush=True)
    (DST / "feature_manifest_v3.json").write_text(json.dumps(log, indent=2))
    print("-> " + str(DST), flush=True)


if __name__ == "__main__":
    main()

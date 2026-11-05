#!/usr/bin/env python3
"""Detect participant IDs in the COG-BCI release that contain the same EEG recording.

COG-BCI (Hinss et al., Sci. Data 10:85, 2023; Zenodo record 6874129) ships one EEGLAB
.set/.fdt pair per participant, session and task. This script hashes every .fdt file and
reports groups of participant IDs whose recordings are byte-identical for the same
session and task. Any cross-subject evaluation that treats such IDs as different people
places the same recording in both training and test data.

Usage
-----
    python verify_cogbci_duplicates.py --root /path/to/COG-BCI
    python verify_cogbci_duplicates.py --root /path/to/COG-BCI --quick   # first 64 MiB only
    python verify_cogbci_duplicates.py --root /path/to/COG-BCI --json report.json

Expected layout: <root>/sub-XX/ses-SY/eeg/<task>.fdt (the layout of the Zenodo release
after extraction). Standard library only.

On the release we analysed (all 29 zips verified against the Zenodo md5 manifest):
  * sub-17, 18, 19, 20, 21, 22, 23, 24 and 27 share identical EEG in every session;
  * sub-25 and sub-28 share identical EEG in session S1 (all 12 tasks); S2 and S3 differ.
Exclusion rule used in our analyses (fixed before re-analysis): drop every participant ID
that shares at least one identical recording with another ID, keeping the lowest-numbered
ID of each cluster. This keeps sub-17 and sub-25 and leaves 20 participants.

Note: the script groups recordings by session and task. A global scan across all
subjects, sessions and tasks on the same release found no other shared recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

CHUNK = 1 << 20
QUICK_BYTES = 64 << 20


def digest(path: Path, quick: bool) -> str:
    h = hashlib.md5()
    read = 0
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            read += len(block)
            if quick and read >= QUICK_BYTES:
                break
    return f"{h.hexdigest()}:{path.stat().st_size}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="COG-BCI root containing sub-XX folders")
    ap.add_argument("--quick", action="store_true", help="hash only the first 64 MiB of each file")
    ap.add_argument("--json", type=Path, help="optional path for a JSON report")
    args = ap.parse_args()

    files = sorted(args.root.glob("sub-*/ses-*/eeg/*.fdt"))
    if not files:
        print(f"No .fdt files found under {args.root}", file=sys.stderr)
        return 2

    by_key: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for i, f in enumerate(files, 1):
        subject = f.parts[-4]
        session = f.parts[-3]
        task = f.stem
        by_key[(session, task, digest(f, args.quick))].append(subject)
        if i % 50 == 0:
            print(f"  hashed {i}/{len(files)}", file=sys.stderr, flush=True)

    shared = defaultdict(set)
    for (session, task, _), subjects in by_key.items():
        if len(subjects) > 1:
            shared[tuple(sorted(subjects))].add(f"{session}/{task}")

    subjects_all = sorted({f.parts[-4] for f in files})
    print(f"Recordings hashed: {len(files)}; participant IDs: {len(subjects_all)}")
    if not shared:
        print("No identical recordings shared between participant IDs.")
    for group, where in sorted(shared.items()):
        print(f"Identical EEG in {', '.join(group)}: {len(where)} recordings "
              f"(e.g. {', '.join(sorted(where)[:3])})")

    duplicated = sorted({s for g in shared for s in g})
    groups = [sorted(g) for g in shared]
    keep = sorted({g[0] for g in groups})
    drop = sorted(set(duplicated) - set(keep))
    unique_n = len(subjects_all) - len(drop)
    print(f"Unique participants: {unique_n} (keep {keep}, drop {drop})")

    if args.json:
        args.json.write_text(json.dumps({
            "root": str(args.root), "quick": args.quick, "n_recordings": len(files),
            "participant_ids": subjects_all,
            "shared_groups": [{"subjects": list(g), "recordings": sorted(w)} for g, w in shared.items()],
            "keep": keep, "drop": drop, "unique_participants": unique_n}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

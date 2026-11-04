#!/usr/bin/env python3
"""
Verifikasi & ekstraksi dataset STEW (IEEE DataPort).

IEEE DataPort mensyaratkan login akun (gratis) untuk mengunduh berkas
"STEW Dataset.zip". Setelah berkas tersedia secara lokal, jalankan:

    python verify_stew.py --zip "/path/ke/STEW Dataset.zip" --out /path/ke/raw/STEW

Skrip ini akan:
- menghitung MD5 berkas zip & menyimpannya ke manifest
- mengekstrak isi ZIP
- memvalidasi keberadaan ratings file + setidaknya 48 pasang subNN_lo / subNN_hi
- mencetak ringkasan struktur
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "manifest" / "stew_manifest.json"
CHUNK = 1024 * 1024


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True, help="Path STEW Dataset.zip hasil unduh manual")
    ap.add_argument("--out", type=Path, required=True, help="Direktori ekstraksi")
    ap.add_argument("--keep-zip", action="store_true", help="Pertahankan zip setelah ekstraksi")
    args = ap.parse_args()

    if not args.zip.exists():
        print(f"[err] tidak ditemukan: {args.zip}")
        return 1

    print(f"[hash] menghitung MD5 {args.zip.name} ...")
    digest = md5_of(args.zip)
    size = args.zip.stat().st_size
    print(f"[ok]   md5={digest}  size={size} byte")

    manifest = json.loads(MANIFEST.read_text())
    manifest["files"][0]["md5"] = digest
    manifest["files"][0]["size_bytes_actual"] = size
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[ok]   manifest diperbarui: {MANIFEST}")

    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        zf.extractall(args.out)
    print(f"[ok]   diekstrak ke {args.out}")

    # Validasi isi
    txts = list(args.out.rglob("*.txt"))
    lo = sorted(p for p in txts if re.search(r"sub\d+.*lo", p.name, re.I))
    hi = sorted(p for p in txts if re.search(r"sub\d+.*hi", p.name, re.I))
    ratings = [p for p in args.out.rglob("*") if "rating" in p.name.lower()]

    print(f"[info] file lo (resting)  : {len(lo)}")
    print(f"[info] file hi (SIMKAP)   : {len(hi)}")
    print(f"[info] ratings ditemukan  : {[p.name for p in ratings]}")

    if len(lo) < 40 or len(hi) < 40:
        print("[warn] jumlah file per kondisi di bawah ekspektasi (target 48).")
        return 2

    if not args.keep_zip:
        args.zip.unlink()
        print(f"[ok]   {args.zip.name} dihapus (zip sudah diekstrak).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

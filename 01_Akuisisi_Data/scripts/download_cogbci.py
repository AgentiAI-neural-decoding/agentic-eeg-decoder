#!/usr/bin/env python3
"""
Unduh dataset COG-BCI dari Zenodo (record 6874129) dengan:
- resume otomatis (HTTP Range)
- verifikasi MD5 per file
- daftar subjek dapat disaring (mis. --subjects 01 02 03)

Penggunaan:
    python download_cogbci.py --out /path/ke/raw/COG-BCI
    python download_cogbci.py --out /path/ke/raw/COG-BCI --subjects 01 02 03
    python download_cogbci.py --out /path/ke/raw/COG-BCI --metadata-only

Catatan: ~31,5 GB total bila mengunduh seluruh 29 subjek. Pastikan ruang disk cukup.
Lisensi: CC-BY-4.0 — wajib sertakan sitasi Hinss et al. (2023).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MANIFEST_DEFAULT = Path(__file__).resolve().parent.parent / "manifest" / "cogbci_manifest.json"
BASE_URL = "https://zenodo.org/api/records/6874129/files/{name}/content"
CHUNK = 1024 * 1024  # 1 MiB


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def download(name: str, expected_size: int, expected_md5: str, out_dir: Path) -> bool:
    url = BASE_URL.format(name=name)
    dest = out_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[skip] {name} sudah lengkap ({human(expected_size)}). Verifikasi MD5...")
        if md5_of(dest) == expected_md5:
            print(f"[ok]   {name} MD5 cocok.")
            return True
        print(f"[warn] {name} MD5 tidak cocok — mengunduh ulang.")
        dest.unlink()

    pos = dest.stat().st_size if dest.exists() else 0
    if pos > expected_size:
        print(f"[warn] {name} berkas lokal lebih besar dari ekspektasi — menghapus.")
        dest.unlink()
        pos = 0

    headers = {"Range": f"bytes={pos}-"} if pos else {}
    req = urllib.request.Request(url, headers=headers)
    print(f"[get]  {name} dari byte {pos} (target {human(expected_size)})")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp, dest.open("ab") as f:
            downloaded = pos
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                pct = downloaded * 100 / expected_size
                print(f"\r       {human(downloaded)} / {human(expected_size)} ({pct:5.1f}%)", end="", flush=True)
        print()
    except Exception as e:
        print(f"\n[err]  {name} gagal: {e}")
        return False

    if dest.stat().st_size != expected_size:
        print(f"[err]  {name} ukuran tidak cocok ({dest.stat().st_size} vs {expected_size}).")
        return False

    actual = md5_of(dest)
    if actual != expected_md5:
        print(f"[err]  {name} MD5 mismatch — file rusak. (got {actual})")
        return False

    print(f"[ok]   {name} terunduh & MD5 cocok.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Pengunduh COG-BCI (Zenodo 6874129)")
    ap.add_argument("--out", type=Path, required=True, help="Direktori tujuan")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT, help="Path manifest JSON")
    ap.add_argument("--subjects", nargs="*", help="Daftar nomor subjek (01..29). Kosong = semua.")
    ap.add_argument("--metadata-only", action="store_true", help="Hanya unduh PDF/notebook/triggerlist")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    files = manifest["files"]
    if args.metadata_only:
        files = [f for f in files if not f["name"].startswith("sub-")]
    elif args.subjects:
        wanted = {f"sub-{s.zfill(2)}.zip" for s in args.subjects}
        files = [f for f in files if f["name"] in wanted or not f["name"].startswith("sub-")]

    total = sum(f["size_bytes"] for f in files)
    print(f"Akan mengunduh {len(files)} file, total ~{human(total)}.\n")

    failed = []
    for f in files:
        if not download(f["name"], f["size_bytes"], f["md5"], args.out):
            failed.append(f["name"])

    if failed:
        print(f"\nSelesai dengan kegagalan: {failed}")
        return 1
    print("\nSemua file berhasil diunduh & terverifikasi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

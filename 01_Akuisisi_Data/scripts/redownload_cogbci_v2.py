#!/usr/bin/env python3
"""Unduh + ekstrak ulang COG-BCI dengan verifikasi DUA LAPIS.

Dijalankan di server, terlepas dari sesi SSH (setsid nohup).

Menutup celah yang meloloskan duplikasi 2026: skrip lama memverifikasi md5
**zip**, tetapi tidak memverifikasi hasil **ekstraksi**. Skrip ini:

  1. unduh tiap zip (resume HTTP Range) → verifikasi md5 terhadap manifes Zenodo
  2. ekstrak tiap anggota zip satu per satu
  3. verifikasi CRC32 tiap berkas hasil ekstrak terhadap CRC yang tercatat di zip
  4. setelah semua subjek selesai: **uji silang antar subjek** — tidak boleh ada
     dua subjek yang berbagi md5 `.fdt` yang sama (inilah uji yang dulu tidak ada)

Tidak menyentuh data lama: menulis ke `raw_v2/`, zip disimpan di `zips_v2/`
supaya ekstraksi dapat diaudit ulang tanpa mengunduh lagi.

Aman diulang: berkas yang sudah lengkap dan cocok dilewati.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, urllib.request, zipfile, zlib
from pathlib import Path

HOME = Path.home()
BASE = HOME / "neuraldecoding" / "01_Akuisisi_Data"
MANIFEST = BASE / "manifest" / "cogbci_manifest.json"
ZIPDIR = BASE / "zips_v2"
OUTDIR = BASE / "raw_v2" / "COG-BCI"
STATE = BASE / "redownload_v2_state.json"   # ditimpa oleh --state
URL = "https://zenodo.org/api/records/6874129/files/{name}/content"
CHUNK = 1 << 20
RETRIES = 6


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def human(n):
    n = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PiB"


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def crc32_of(p: Path) -> int:
    c = 0
    with p.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"downloaded": {}, "extracted": {}}


def save_state(s: dict):
    STATE.write_text(json.dumps(s, indent=1))


# ------------------------------------------------------------------ unduh
def fetch(name: str, size: int, md5: str) -> bool:
    dest = ZIPDIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == size:
        log(f"  {name}: sudah ada, verifikasi md5…")
        if md5_of(dest) == md5:
            log(f"  {name}: md5 COCOK")
            return True
        log(f"  {name}: md5 TIDAK cocok — unduh ulang")
        dest.unlink()

    for attempt in range(1, RETRIES + 1):
        pos = dest.stat().st_size if dest.exists() else 0
        if pos > size:
            dest.unlink()
            pos = 0
        if pos == size:
            break
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            req = urllib.request.Request(URL.format(name=name), headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r, dest.open("ab") as f:
                got = pos
                last = time.time()
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    if time.time() - last > 30:
                        log(f"  {name}: {human(got)} / {human(size)} ({got*100/size:.1f}%)")
                        last = time.time()
            break
        except Exception as e:
            log(f"  {name}: percobaan {attempt}/{RETRIES} gagal — {e}")
            time.sleep(min(60, 5 * attempt))
    else:
        log(f"  {name}: GAGAL setelah {RETRIES} percobaan")
        return False

    if not dest.exists() or dest.stat().st_size != size:
        log(f"  {name}: ukuran tidak cocok")
        return False
    got_md5 = md5_of(dest)
    if got_md5 != md5:
        log(f"  {name}: md5 SALAH (dapat {got_md5}, harap {md5})")
        return False
    log(f"  {name}: terunduh, md5 COCOK")
    return True


# --------------------------------------------------------------- ekstrak
def extract(name: str) -> tuple[bool, int, int]:
    """Ekstrak dan verifikasi CRC32 tiap anggota. -> (ok, n_berkas, n_fdt)"""
    z = ZIPDIR / name
    OUTDIR.mkdir(parents=True, exist_ok=True)
    n_file = n_fdt = 0
    with zipfile.ZipFile(z) as zf:
        bad = zf.testzip()
        if bad is not None:
            log(f"  {name}: CRC rusak DI DALAM zip pada {bad}")
            return False, 0, 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            zf.extract(info, OUTDIR)
            target = OUTDIR / info.filename
            actual = crc32_of(target)
            if actual != info.CRC:
                log(f"  {name}: CRC hasil ekstrak SALAH pada {info.filename} "
                    f"({actual:08x} vs {info.CRC:08x})")
                return False, n_file, n_fdt
            n_file += 1
            if info.filename.endswith(".fdt"):
                n_fdt += 1
    return True, n_file, n_fdt


# ------------------------------------------------------------------ main
def cross_check() -> int:
    """Uji silang antar subjek: tidak boleh ada .fdt kembar."""
    log("=== UJI SILANG: apakah ada subjek yang berbagi .fdt identik? ===")
    seen: dict[str, list[str]] = {}
    n = 0
    for p in sorted(OUTDIR.rglob("*.fdt")):
        rel = p.relative_to(OUTDIR)
        key = f"{md5_of(p)}|{'/'.join(rel.parts[1:])}"
        seen.setdefault(key, []).append(rel.parts[0])
        n += 1
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    log(f"  {n} berkas .fdt diperiksa")
    subs = sorted({p.relative_to(OUTDIR).parts[0] for p in OUTDIR.rglob("*.fdt")})
    log(f"  subjek terekstrak: {len(subs)}")
    if dup:
        log(f"  *** MASIH ADA DUPLIKAT pada {len(dup)} berkas ***")
        for k, v in list(dup.items())[:10]:
            log(f"      {k.split('|')[1]}: {sorted(v)}")
        log("  -> ekstraksi masih bermasalah; JANGAN dipakai")
        return 1
    log("  BERSIH — tidak ada dua subjek yang berbagi .fdt yang sama")
    return 0


def main() -> int:
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="0/1", help="i/n — pekerja ke-i dari n")
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if args.state:
        STATE = args.state
    if args.verify_only:
        return cross_check()
    i, nw = (int(x) for x in args.slice.split("/"))

    man = json.loads(MANIFEST.read_text())
    files = [f for k, f in enumerate(man["files"]) if k % nw == i]
    state = load_state()
    ZIPDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    total = sum(f["size_bytes"] for f in files)
    log(f"MULAI pekerja {i+1}/{nw} — {len(files)} berkas, total {human(total)}")
    log(f"  zip     -> {ZIPDIR}")
    log(f"  ekstrak -> {OUTDIR}")
    log("  data lama di raw/COG-BCI TIDAK disentuh")

    gagal = []
    for k, f in enumerate(files, 1):
        name, size, md5 = f["name"], f["size_bytes"], f["md5"]
        log(f"[{k}/{len(files)}] {name} ({human(size)})")

        if state["downloaded"].get(name) != md5:
            if not fetch(name, size, md5):
                gagal.append(name)
                continue
            state["downloaded"][name] = md5
            save_state(state)
        else:
            log(f"  {name}: sudah terunduh & terverifikasi (state)")

        if not name.endswith(".zip"):
            log(f"  {name}: bukan zip, dibiarkan di {ZIPDIR}")
            continue

        if state["extracted"].get(name):
            log(f"  {name}: sudah diekstrak (state)")
            continue
        ok, nf, nfdt = extract(name)
        if not ok:
            gagal.append(name)
            continue
        state["extracted"][name] = {"n_file": nf, "n_fdt": nfdt}
        save_state(state)
        log(f"  {name}: ekstrak OK — {nf} berkas ({nfdt} .fdt), CRC semua cocok")

    log("")
    if gagal:
        log(f"SELESAI DENGAN KEGAGALAN: {gagal}")
        return 1
    log("SELESAI — semua berkas terunduh, terekstrak, dan terverifikasi")
    log(f"Langkah berikutnya (keputusan penulis): bandingkan raw_v2/COG-BCI "
        f"dengan raw/COG-BCI, lalu jalankan ulang pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

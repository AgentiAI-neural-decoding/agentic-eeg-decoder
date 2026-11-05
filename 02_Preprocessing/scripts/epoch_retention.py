#!/usr/bin/env python3
"""Rekap retensi epoch per rekaman, diturunkan dari log prapemrosesan.

Log ditulis dengan mode append, sehingga satu berkas dapat memuat beberapa
kali proses (uji coba + proses penuh). Untuk tiap rekaman diambil kemunculan
TERAKHIR, yaitu hasil proses final yang menghasilkan derivatives.

Ambang retensi minimum dibaca dari config tiap dataset (`min_epoch_retention`).
Rekaman di bawah ambang dieksklusi dari himpunan analisis; rekapitulasi
melaporkan angka sebelum dan sesudah eksklusi agar keduanya dapat ditelusuri.

Pemakaian:  python3 epoch_retention.py [--csv DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
SOURCES = [("STEW", REPORTS / "preprocess_stew.log", CONFIG / "stew.json"),
           ("COG-BCI", REPORTS / "preprocess_cogbci.log", CONFIG / "cogbci.json")]

RE_REC = re.compile(r"=== (\S+) ===")
RE_EPO = re.compile(r"epoch tersisa:\s*(\d+)\s*/\s*(\d+)")


def parse(path: Path) -> tuple[dict[str, tuple[int, int]], int]:
    """Kembalikan {rekaman: (tersisa, total)} kemunculan terakhir, dan jumlah baris mentah."""
    current = None
    last: dict[str, tuple[int, int]] = {}
    raw = 0
    with path.open(errors="replace") as fh:
        for line in fh:
            m = RE_REC.search(line)
            if m:
                current = m.group(1)
                continue
            m = RE_EPO.search(line)
            if m and current:
                raw += 1
                last[current] = (int(m.group(1)), int(m.group(2)))
    return last, raw


def threshold_of(path: Path) -> float | None:
    """Ambil `min_epoch_retention` dari config; None bila tidak diatur."""
    if not path.exists():
        return None
    def walk(node):
        if isinstance(node, dict):
            if "min_epoch_retention" in node:
                return node["min_epoch_retention"]
            for v in node.values():
                got = walk(v)
                if got is not None:
                    return got
        return None
    return walk(json.load(path.open()))


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=REPORTS)
    args = ap.parse_args()

    print("=" * 74)
    print("REKAP RETENSI EPOCH PER REKAMAN")
    print("=" * 74)
    print(f"Dibuat  : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Sumber  : 02_Preprocessing/reports/preprocess_{{stew,cogbci}}.log")
    print("Metode  : kemunculan TERAKHIR per rekaman (log bermode append)")
    print("Ambang  : min_epoch_retention dibaca dari 02_Preprocessing/config/*.json")
    print()

    for name, log, cfg in SOURCES:
        if not log.exists():
            print(f"[{name}] log tidak ditemukan: {log}")
            continue

        last, raw = parse(log)
        thr = threshold_of(cfg)
        rates = {r: (k / t if t else 0.0) for r, (k, t) in last.items()}
        drop = sorted((r for r, v in rates.items() if thr is not None and v < thr),
                      key=lambda r: rates[r])
        keep = [r for r in last if r not in set(drop)]

        print(f"[{name}]  ambang retensi: "
              + (f"{100 * thr:.0f}%" if thr is not None else "tidak diatur di config"))
        print(f"  baris log mentah    : {fmt(raw)}  ->  rekaman unik: {fmt(len(last))}"
              f"  ({fmt(raw - len(last))} baris dari proses ulang, diabaikan)")
        print(f"  retensi per rekaman : median {100 * statistics.median(rates.values()):.1f}%"
              f" | min {100 * min(rates.values()):.1f}%"
              f" | maks {100 * max(rates.values()):.1f}%")
        print()

        pre_k = sum(k for k, _ in last.values())
        pre_t = sum(t for _, t in last.values())
        print(f"  SEBELUM eksklusi    : {fmt(len(last))} rekaman | "
              f"{fmt(pre_k)} / {fmt(pre_t)} epoch = {100 * pre_k / pre_t:.1f}%")

        if thr is None:
            print("  (tidak ada eksklusi: dataset ini tidak memakai ambang retensi)")
        else:
            print(f"  DIEKSKLUSI (<{100 * thr:.0f}%) : {len(drop)} rekaman"
                  f" | {fmt(sum(last[r][0] for r in drop))} epoch")
            for r in drop:
                k, t = last[r]
                print(f"      - {r:<24} {k:>4} / {t:<5} {100 * rates[r]:>5.1f}%")
            post_k = sum(last[r][0] for r in keep)
            post_t = sum(last[r][1] for r in keep)
            print(f"  DIPAKAI ANALISIS    : {fmt(len(keep))} rekaman | "
                  f"{fmt(post_k)} / {fmt(post_t)} epoch = {100 * post_k / post_t:.1f}%")
        print()

        out = args.csv / f"epoch_retention_{name.lower().replace('-', '')}.csv"
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rekaman", "epoch_tersisa", "epoch_total", "retensi", "status"])
            for rec in sorted(last):
                k, t = last[rec]
                status = "dieksklusi" if rec in set(drop) else "dipakai"
                w.writerow([rec, k, t, f"{k / t:.4f}" if t else "0", status])
        print(f"  tabel lengkap ({fmt(len(last))} baris) -> reports/{out.name}")
        print()

    print("=" * 74)


if __name__ == "__main__":
    main()

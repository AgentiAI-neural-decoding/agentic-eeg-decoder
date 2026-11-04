#!/usr/bin/env python3
"""Tahap 06 — sweep tuning CBraMod (normalisasi × pooling) untuk meningkatkan hasil.

Hasil awal CBraMod frozen (norm=scale100, pooling=mean): STEW single F1=0.671 (< PSD 0.741),
transfer ccDA +0.011. Sweep ini mencoba kombinasi murah yang paling memengaruhi kualitas
embedding foundation model, lalu melaporkan F1 + transfer per config (terurut).

Lever:
- norm    : 'scale100' (µV/100) | 'scale1' (µV) | 'zscore' (per-epoch per-kanal, bebas tebak skala)
- pooling : 'mean' (200-dim) | 'meanstd' (400-dim, tambah dispersi)

Embedding di-cache per (norm) agar tidak forward berulang untuk pooling berbeda.

Penggunaan (idealnya GPU/Colab):
    python tune_cbramod.py --config ../config/fm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm_encoder import build_encoder, load_harmonized_epochs
from probe_transfer import run

CONFIGS = [
    {"norm": "scale", "scale": 100.0, "pooling": "mean"},     # baseline
    {"norm": "scale", "scale": 1.0, "pooling": "mean"},
    {"norm": "zscore", "pooling": "mean"},
    {"norm": "zscore", "pooling": "meanstd"},
    {"norm": "scale", "scale": 100.0, "pooling": "meanstd"},
]


def embed_dataset(enc, base, cfg, ds, sfreq, log):
    deriv = (base / cfg["derivatives"][ds]).resolve()
    X, y, g = load_harmonized_epochs(deriv, ds, cfg["common_channels"], sfreq,
                                     cfg["cv"]["max_epochs_per_dataset"], cfg["cv"]["random_state"],
                                     scheme=cfg.get("label_scheme", "binary"))
    return enc.embed(X), y, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target", default="STEW")
    ap.add_argument("--source", default="COG-BCI")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent.parent
    out_root = (base / cfg["output_root"]).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    common = cfg["common_channels"]
    base_opts = cfg["encoder_options"]["cbramod"]
    sfreq = base_opts.get("sfreq", 200)

    results = []
    for i, sweep in enumerate(CONFIGS):
        tag = f"norm={sweep.get('norm')}" + (f"/scale={sweep.get('scale')}" if sweep["norm"] == "scale" else "") + f"/pool={sweep['pooling']}"
        log(f"\n[{i+1}/{len(CONFIGS)}] {tag}")
        opts = {**base_opts, **sweep}
        enc = build_encoder("cbramod", opts, len(common), cfg.get("device", "auto"))
        emb = {}
        for ds in {args.target, args.source}:
            emb[ds] = embed_dataset(enc, base, cfg, ds, sfreq, log)
            log(f"   {ds}: emb {emb[ds][0].shape}")
        r = run(args.target, args.source, emb, cfg, log)
        row = {"config": tag, "single_f1": r["single"]["f1"], "multi_f1": r["multi"]["f1"],
               "ccda_f1": r["multi_ccda"]["f1"], "gain_ccda": r["gain_f1_ccda"]}
        results.append(row)
        log(f"   -> single F1={row['single_f1']} | ccDA F1={row['ccda_f1']} (gain {row['gain_ccda']:+})")

    results.sort(key=lambda x: x["ccda_f1"], reverse=True)
    log("\n" + "=" * 70)
    log(f"RANKING (target={args.target}, by ccDA F1):")
    for r in results:
        log(f"  {r['ccda_f1']:.3f}  ccDA | {r['single_f1']:.3f} single | gain {r['gain_ccda']:+.3f} | {r['config']}")
    log("=" * 70)
    log(f"TERBAIK: {results[0]['config']} -> ccDA F1={results[0]['ccda_f1']}")

    (out_root / "tune_cbramod_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nHasil -> {out_root / 'tune_cbramod_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

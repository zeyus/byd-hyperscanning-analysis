#!/usr/bin/env python3
"""
Quick approximation to Poulsen et al.'s ALD, computed from the per-frame
luminance CSV you have ALREADY extracted. No video decoding, runs in a second.

This applies only the two operations that matter most for the correlation
against a 5 s-windowed ISC series:

    step 3  resample to 1 Hz by per-second MAXIMUM   (not mean/interpolation)
    step 4  Gaussian smoothing, variance 2.5 s^2

It CANNOT apply the other two, because signalstats YDIF has already collapsed
the per-pixel information:

    step 1  greyscale as equal-weight RGB mean       (YDIF uses Rec.601 luma)
    step 2  SQUARED per-pixel differences            (YDIF is mean ABSOLUTE)

So treat this as a diagnostic, not as the number for the thesis. If the
correlation jumps here, the disagreement with the reference study is largely a
feature-definition artefact and compute_ald.py will confirm it. If it does not
move, the occipital explanation gets correspondingly stronger. Either way, run
compute_ald.py for the value you actually report.

Usage
-----
    python ald_from_existing_ydif.py \
        "data/BangBangYouAreDead_SerialTrigInterval-1sec-luminance.csv" \
        -o out/ald_approx_byd.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Column order as written by scripts/frame_analysis.fish and parsed by
# src/composite-stimuli-features.py
LUM_COLUMNS = ["timestamp", "min_lum", "mean_lum", "max_lum", "diff_lum"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("luminance_csv", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--sigma2", type=float, default=2.5,
                    help="Gaussian variance parameter in s^2 (default 2.5)")
    ap.add_argument("--square", action="store_true",
                    help="Square YDIF before pooling. This is NOT equivalent to "
                         "step 2 (mean of squares != square of mean) but brackets "
                         "the direction of that approximation.")
    args = ap.parse_args()

    df = pd.read_csv(args.luminance_csv, header=None).drop(columns=[0])
    df.columns = LUM_COLUMNS

    t = df["timestamp"].to_numpy(dtype=float)
    y = df["diff_lum"].to_numpy(dtype=float)
    if args.square:
        y = y ** 2

    # step 3: per-second maximum
    n_bins = int(np.floor(t.max())) + 1
    bin_idx = np.floor(t).astype(int)
    pooled_max = np.full(n_bins, np.nan)
    pooled_mean = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = y[bin_idx == b]
        if sel.size:
            pooled_max[b] = sel.max()
            pooled_mean[b] = sel.mean()
    idx = np.arange(n_bins)
    for arr in (pooled_max, pooled_mean):
        bad = np.isnan(arr)
        if bad.any():
            arr[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])

    # step 4: Gaussian smoothing
    sigma = np.sqrt(args.sigma2)
    ald = gaussian_filter1d(pooled_max, sigma=sigma, mode="nearest")

    out = pd.DataFrame({
        "time_s": idx + 0.5,
        "ydif_mean": pooled_mean,   # closest to what the thesis currently uses
        "ald_raw": pooled_max,      # after step 3 only
        "ald": ald,                 # after steps 3 and 4  <- compare against ISC
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    r = np.corrcoef(out["ydif_mean"], out["ald"])[0, 1]
    print(f"wrote {len(out)} rows to {args.out}")
    print(f"corr(mean-pooled YDIF, max-pooled+smoothed) = {r:.3f}")
    print("If that is well below 1, the two feature definitions are genuinely "
          "different signals and the RQ3 comparison was never like-for-like.")


if __name__ == "__main__":
    main()

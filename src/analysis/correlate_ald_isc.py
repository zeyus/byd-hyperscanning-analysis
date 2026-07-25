#!/usr/bin/env python3
"""
Correlate an ALD series against the component-1 ISC time course, using the same
exact circular-shift test as the rest of the thesis.

    python correlate_ald_isc.py \
        --ald out/ald_byd.csv \
        --isc out/03_ISC_results/byd/full/isc_persecond.npy \
        --isc-times out/03_ISC_results/byd/full/window_times.npy \
        --component 1

Adjust --isc / --isc-times to whatever compute_isc.py actually writes; if it
writes a single .npz or .json, load it and pass the arrays in yourself. The
statistical part below is the bit that matters and is self-contained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def exact_circular_shift_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Two-sided exact p for Pearson r between x and y, enumerating all N-1
    circular shifts of x. Uses the (k+1)/N convention, matching the thesis.
    """
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    r_obs = float(np.mean(x * y))
    r_null = np.array([np.mean(np.roll(x, s) * y) for s in range(1, n)])
    k = int(np.sum(np.abs(r_null) >= abs(r_obs)))
    return r_obs, (1 + k) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ald", type=Path, required=True)
    ap.add_argument("--ald-column", default="ald")
    ap.add_argument("--isc", type=Path, required=True,
                    help=".npy of shape (n_components, n_windows)")
    ap.add_argument("--isc-times", type=Path, required=True,
                    help=".npy of window times, length n_windows")
    ap.add_argument("--component", type=int, default=1)
    args = ap.parse_args()

    ald_df = pd.read_csv(args.ald)
    isc = np.load(args.isc)
    isc_t = np.load(args.isc_times)
    comp = isc[args.component - 1]

    # interpolate the 1 Hz ALD onto the ISC window grid, as the thesis does for
    # every other stimulus feature
    ald_on_grid = np.interp(isc_t, ald_df["time_s"].to_numpy(), ald_df[args.ald_column].to_numpy())

    r, p = exact_circular_shift_p(ald_on_grid, comp)
    print(f"component {args.component} ISC  vs  {args.ald_column}")
    print(f"  N windows = {len(comp)}")
    print(f"  r = {r:.4f}")
    print(f"  exact p = {p:.4f}   (floor = {1/len(comp):.4f})")
    print(f"  r^2 = {r**2:.4f}  -> {100*r**2:.1f} % of variance")
    print()
    print("Poulsen et al. (2017) Table 3 report r = 0.71 for their Individual group.")
    print("The thesis currently reports r = 0.067 using mean-pooled YDIF.")


if __name__ == "__main__":
    main()

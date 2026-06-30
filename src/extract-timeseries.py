"""
extract_timeseries.py
─────────────────────
Extracts digitised timeseries from a calibrated Inkscape SVG, resamples
them to a common x-grid, and computes pairwise Pearson correlations.

Usage
-----
python extract_timeseries.py traced_timeseries_plain.svg \
    [--lab-data lab_results.npy] \
    [--output-csv timeseries.csv]

All three series (Dmochowski, Poulsen, optionally lab results) are written to
a CSV and a comparison figure is saved alongside it.
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from svgpathtools import parse_path

# ── SVG namespace helper ──────────────────────────────────────────────────────

NS = {"svg": "http://www.w3.org/2000/svg"}


def find_by_id(root, element_id):
    # Search anywhere in the tree
    for el in root.iter():
        if el.get("id") == element_id:
            return el
    raise ValueError(f"Element id='{element_id}' not found in SVG")


# ── Group transform parser ────────────────────────────────────────────────────


def parse_translate(transform_str):
    """Return (tx, ty) from a 'translate(x,y)' attribute string."""
    m = re.search(
        r"translate\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)", transform_str
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


# ── Path → polyline sampler ───────────────────────────────────────────────────


def sample_path(d_attr, n_samples=1000):
    """
    Parse an SVG path `d` attribute and sample it at `n_samples` evenly-spaced
    parameter values.  Returns arrays of absolute (x, y) coordinates in SVG
    user-space units (before any group transform is applied).
    """
    path = parse_path(d_attr)
    total = path.length()
    ts = np.linspace(0, 1, n_samples)
    pts = np.array([path.point(t) for t in ts])
    return pts.real, pts.imag  # x, y


def sample_horizontal_line(d_attr):
    """Return the y-coordinate of a horizontal line path (M x1 y H x2)."""
    m = re.search(r"[Mm]\s+([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)", d_attr)
    if m:
        return float(m.group(2))
    raise ValueError(f"Could not parse horizontal line y from: {d_attr}")


# ── Calibration → value mapping ───────────────────────────────────────────────


def make_y_to_value(y_01, y_02, val_01=0.1, val_02=0.2):
    """
    Linear map from SVG y → correlation value using the two reference lines.
    SVG y increases downward, so the 0.2 line should be at a *smaller* y than
    the 0.1 line (higher on the page = smaller y = larger value).
    Returns a callable.
    """
    # y = m * val + b  →  solve for m, b
    # m * val_01 + b = y_01
    # m * val_02 + b = y_02
    m = (y_02 - y_01) / (val_02 - val_01)
    b = y_01 - m * val_01
    return lambda y: (y - b) / m


# ── Main extraction ───────────────────────────────────────────────────────────


def extract_all(svg_path, n_samples=1204):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Find the group that holds everything and read its translate
    group = find_by_id(root, "traced")
    tx, ty = parse_translate(group.get("transform", ""))

    # ── calibration lines (already absolute coordinates because H lines) ──
    cal01_el = find_by_id(root, "ref_0point1")
    cal02_el = find_by_id(root, "ref_0point2")

    # These paths are children of the group, so add group translate to get
    # canvas coordinates.
    y_cal01 = sample_horizontal_line(cal01_el.get("d")) + ty
    y_cal02 = sample_horizontal_line(cal02_el.get("d")) + ty

    print(f"Calibration y (SVG canvas): 0.1 → {y_cal01:.2f},  0.2 → {y_cal02:.2f}")
    assert y_cal02 < y_cal01, (
        "Expected 0.2 line to be above 0.1 line (smaller SVG y). "
        f"Got y_0.1={y_cal01:.1f}, y_0.2={y_cal02:.1f}"
    )

    y2val = make_y_to_value(y_cal01, y_cal02)

    # ── timeseries paths ──────────────────────────────────────────────────
    results = {}
    for series_id in ("dmochowski_timeseries", "poulsen_timeseries"):
        el = find_by_id(root, series_id)
        xs, ys = sample_path(el.get("d"), n_samples=n_samples)

        # Apply group translate to get canvas coords
        xs_abs = xs + tx
        ys_abs = ys + ty

        vals = y2val(ys_abs)
        results[series_id] = (xs_abs, vals)
        print(
            f"{series_id}: x ∈ [{xs_abs.min():.0f}, {xs_abs.max():.0f}]  "
            f"val ∈ [{vals.min():.3f}, {vals.max():.3f}]"
        )

    return results, y2val


# ── Resampling to a common grid ───────────────────────────────────────────────


def resample_to_common_grid(series_dict, n_grid=360):
    """
    Interpolate each series onto a shared x-grid spanning the overlapping
    x-range of all series.
    Returns: x_grid (1-D), dict of {name: values_1d}
    """
    # Find the overlapping x-range
    x_min = max(xs.min() for xs, _ in series_dict.values())
    x_max = min(xs.max() for xs, _ in series_dict.values())
    x_grid = np.linspace(x_min, x_max, n_grid)

    resampled = {}
    for name, (xs, vals) in series_dict.items():
        # Sort by x (paths can have slight back-tracking artefacts)
        order = np.argsort(xs)
        resampled[name] = np.interp(x_grid, xs[order], vals[order])

    return x_grid, resampled


# ── Correlation table ─────────────────────────────────────────────────────────


def correlation_table(resampled):
    keys = list(resampled.keys())
    print("\nPairwise Pearson correlations")
    print("─" * 55)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            r, p = pearsonr(resampled[a], resampled[b])
            print(f"  {a:35s} × {b:35s}:  r = {r:+.4f}  (p = {p:.3e})")


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_series(x_grid, resampled, out_png="timeseries_comparison.png"):
    fig, ax = plt.subplots(figsize=(14, 4))
    labels = {
        "dmochowski_timeseries": "Dmochowski et al.",
        "poulsen_timeseries": "Poulsen et al.",
    }
    colours = {
        "dmochowski_timeseries": "#1f77b4",
        "poulsen_timeseries": "#d90000",
        "lab_results": "#2ca02c",
    }
    for name, vals in resampled.items():
        ax.plot(
            x_grid,
            vals,
            label=labels.get(name, name),
            color=colours.get(name, "grey"),
            linewidth=1.2,
        )

    # Calibration reference lines
    ax.axhline(0.1, color="#888", linewidth=0.8, linestyle="--", label="0.1 ref")
    ax.axhline(0.2, color="#555", linewidth=0.8, linestyle="--", label="0.2 ref")

    ax.set_ylabel("ISC")
    ax.set_xlabel("Time")
    ax.set_title("Digitised ISC timeseries comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nFigure saved → {out_png}")
    return fig


# ── CSV export ────────────────────────────────────────────────────────────────


def save_csv(x_grid, resampled, path="timeseries.csv"):
    import csv

    keys = list(resampled.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_svgunits"] + keys)
        for i, x in enumerate(x_grid):
            w.writerow([f"{x:.4f}"] + [f"{resampled[k][i]:.6f}" for k in keys])
    print(f"CSV saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract & compare SVG timeseries")
    parser.add_argument("svg", help="Path to the Inkscape SVG file")
    parser.add_argument(
        "--lab-data", help="Optional .npy or .csv with lab results ISC timeseries"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1200,
        help="Points to sample per SVG path (default 2000)",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=360,
        help="Points in the common resampling grid (default 1000)",
    )
    parser.add_argument("--output-csv", default="timeseries.csv")
    parser.add_argument("--output-png", default="timeseries_comparison.png")
    args = parser.parse_args()

    series, y2val = extract_all(args.svg, n_samples=args.n_samples)

    # Optionally add your own data
    if args.lab_data:
        if args.lab_data.endswith(".npy"):
            your_vals = np.load(args.lab_data)
        else:
            your_vals = np.genfromtxt(args.lab_data, delimiter=",")
        # Assign a synthetic x-range that spans the same domain as the others
        x_min = min(xs.min() for xs, _ in series.values())
        x_max = max(xs.max() for xs, _ in series.values())
        your_xs = np.linspace(x_min, x_max, len(your_vals))
        series["lab_results"] = (your_xs, your_vals)
        print(
            f"lab_results: {len(your_vals)} points, "
            f"val ∈ [{your_vals.min():.3f}, {your_vals.max():.3f}]"
        )

    x_grid, resampled = resample_to_common_grid(series, n_grid=args.n_grid)
    correlation_table(resampled)
    save_csv(x_grid, resampled, path=args.output_csv)
    plot_series(x_grid, resampled, out_png=args.output_png)


if __name__ == "__main__":
    main()

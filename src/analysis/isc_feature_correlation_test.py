"""Test whether ISC timeseries correlate with stimulus features.

For each CCA component and each requested feature, computes the Pearson
correlation between the ISC by-window timeseries and the feature resampled onto
the same 5-second/1-step windows. Significance is assessed with an exact
circular-shift permutation test: the ISC series is circularly shifted (every one
of the N-1 possible shifts) relative to the fixed feature series and the
correlation recomputed each time, building an exact null distribution. This is
valid here (unlike for the sparse/uneven emotion event timestamps in
event_locked_isc_test.py) because both series are continuous and evenly sampled,
so a circular shift preserves each series' own autocorrelation while destroying
only their relative alignment.

Benjamini-Hochberg FDR correction is applied across all (component, feature)
pairs tested for a given stimulus.

Features may now come from SEVERAL csv files, so a purpose-built feature such as
the Poulsen et al. (2017) ALD can be supplied alongside the composite frame-level
csv. Each feature is given as a spec string:

    NAME:PATH[:COLUMN[:TIMECOL[:AGGREGATE]]]

COLUMN defaults to NAME, TIMECOL is auto-detected from
{timestamp, time_s, time}, and AGGREGATE defaults to 'auto'.

AGGREGATE controls how the feature is brought onto the ISC window grid:

    mean    average every feature sample falling inside each 5 s window. Right
            for raw frame-rate features, because it does what the ISC windowing
            itself does: summarise a stretch of signal.
    interp  sample the feature at each window centre. Right for features that
            are ALREADY at ~1 Hz and already smoothed to the ISC timescale,
            such as ALD, where averaging again would smooth twice.
    auto    interp if the feature's median sample interval is >= half the step
            size, else mean. Prints whichever it picked.

Run with:

    uv run python src/analysis/isc_feature_correlation_test.py \
        --stimulus bangbangyouaredead \
        --feature "ebu_r128_M:in/BangBangYouAreDead_composite_frame_level_analysis.csv" \
        --feature "ald:out/01_extracted_stim_features/byd/ald.csv"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import stats_utils

STIM_KEYS = {"bangbangyouaredead": "byd", "storycorps_q&a": "sc"}

TIME_COL_CANDIDATES = ("timestamp", "time_s", "time")


class FeatureSpec:
    """NAME:PATH[:COLUMN[:TIMECOL[:AGGREGATE]]]"""

    def __init__(self, spec: str) -> None:
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(
                f"Bad --feature spec {spec!r}. Expected NAME:PATH[:COLUMN[:TIMECOL[:AGGREGATE]]]"
            )
        self.name = parts[0]
        self.path = Path(parts[1])
        self.column = parts[2] if len(parts) > 2 and parts[2] else self.name
        self.time_col = parts[3] if len(parts) > 3 and parts[3] else None
        self.aggregate = parts[4] if len(parts) > 4 and parts[4] else "auto"
        if self.aggregate not in ("auto", "mean", "interp"):
            raise ValueError(
                f"Bad aggregate {self.aggregate!r} in {spec!r}; expected auto/mean/interp"
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.name} from {self.path}:{self.column} ({self.aggregate})>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--isc-dir",
        type=Path,
        default=None,
        help="Defaults to out/03_ISC_results/{stim_key}/{range_tag}",
    )
    p.add_argument(
        "--stimulus",
        required=True,
        choices=sorted(STIM_KEYS),
        help="Filename stimulus key, e.g. bangbangyouaredead or storycorps_q&a",
    )
    p.add_argument(
        "--range-tag",
        default="full",
        help="'full' or 'segment', matches the ISC directory produced by compute_isc.py",
    )
    p.add_argument("--components", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument(
        "--feature",
        dest="features",
        action="append",
        required=True,
        metavar="NAME:PATH[:COLUMN[:TIMECOL[:AGGREGATE]]]",
        help="Repeatable. See the module docstring for the spec format.",
    )
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--step-sec", type=float, default=1.0)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to out/04_feature_correlation/{stim_key}/isc_feature_correlation_stats.csv",
    )
    p.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Defaults to out/04_feature_correlation/{stim_key}/isc_feature_correlation_stats.png",
    )
    args = p.parse_args()

    args.feature_specs = [FeatureSpec(s) for s in args.features]

    stim_key = STIM_KEYS[args.stimulus]
    if args.isc_dir is None:
        args.isc_dir = Path("out/03_ISC_results") / stim_key / args.range_tag
    if args.output_csv is None:
        args.output_csv = (
            Path("out/04_feature_correlation")
            / stim_key
            / "isc_feature_correlation_stats.csv"
        )
    if args.output_plot is None:
        args.output_plot = (
            Path("out/04_feature_correlation")
            / stim_key
            / "isc_feature_correlation_stats.png"
        )
    return args


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def resolve_time_column(df: pd.DataFrame, spec: FeatureSpec) -> str:
    if spec.time_col is not None:
        if spec.time_col not in df.columns:
            raise KeyError(f"Time column {spec.time_col!r} not in {spec.path}")
        return spec.time_col
    for cand in TIME_COL_CANDIDATES:
        if cand in df.columns:
            return cand
    raise KeyError(
        f"No time column found in {spec.path}. Tried {TIME_COL_CANDIDATES}; "
        f"columns present: {list(df.columns)}"
    )


def load_feature(
    spec: FeatureSpec,
    n_windows: int,
    window_sec: float,
    step_sec: float,
    t0_s: float,
) -> np.ndarray | None:
    if not spec.path.exists():
        print(f"  Feature '{spec.name}': file not found ({spec.path}) - skipping.")
        return None
    df = pd.read_csv(spec.path)
    if spec.column not in df.columns:
        print(
            f"  Feature '{spec.name}': column {spec.column!r} not in {spec.path} "
            f"(has {list(df.columns)}) - skipping."
        )
        return None

    tcol = resolve_time_column(df, spec)
    df = df[[tcol, spec.column]].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    t = df[tcol].to_numpy(dtype=float)
    v = df[spec.column].to_numpy(dtype=float)

    aggregate = spec.aggregate
    if aggregate == "auto":
        dt = float(np.median(np.diff(t))) if len(t) > 1 else np.inf
        aggregate = "interp" if dt >= 0.5 * step_sec else "mean"
        print(
            f"  Feature '{spec.name}': median sample interval {dt:.4f}s "
            f"-> aggregate='{aggregate}'"
        )
    else:
        print(f"  Feature '{spec.name}': aggregate='{aggregate}' (explicit)")

    if aggregate == "mean":
        return stats_utils.rolling_window_mean(
            t, v, n_windows, window_sec, step_sec, t0_s
        )

    centres = stats_utils.reconstruct_window_times(
        n_windows, window_sec, step_sec, t0_s
    )
    if centres.max() > t.max() + window_sec or centres.min() < t.min() - window_sec:
        print(
            f"    WARNING: window centres span {centres.min():.1f}-{centres.max():.1f}s "
            f"but '{spec.name}' covers {t.min():.1f}-{t.max():.1f}s. "
            "np.interp will clamp at the edges; check your time bases line up."
        )
    return np.interp(centres, t, v)


def main() -> None:
    args = parse_args()

    meta = stats_utils.load_isc_meta(args.isc_dir)
    window_sec = meta["window_sec"] if meta else args.window_sec
    step_sec = meta["step_sec"] if meta else args.step_sec
    t0_s = meta["t0_s"] if meta else 0.0
    if meta is None:
        print(f"  No meta.json in {args.isc_dir}; falling back to CLI defaults.")

    names = [s.name for s in args.feature_specs]
    rows = []
    r_matrix = np.full((len(args.components), len(names)), np.nan)

    feat_values: dict[str, np.ndarray] = {}
    for ci, comp in enumerate(args.components):
        isc_path = args.isc_dir / f"isc_component{comp}_bywindow.npy"
        isc_values = stats_utils.load_isc_bywindow(isc_path)

        if not feat_values:
            print(f"\nISC: {len(isc_values)} windows from {isc_path.parent}")
            print(f"  t0={t0_s}s  window={window_sec}s  step={step_sec}s")
            print("Loading features:")
            for spec in args.feature_specs:
                vals = load_feature(spec, len(isc_values), window_sec, step_sec, t0_s)
                if vals is not None:
                    feat_values[spec.name] = vals
            n = len(isc_values)
            m = len(feat_values) * len(args.components)
            print(
                f"\nExact-test resolution: N={n} windows, m={m} tests. "
                f"Smallest attainable p = {1 / n:.4f}, smallest attainable q = {m / n:.4f}"
                f"{'  <-- below alpha, FDR significance is attainable' if m / n < args.fdr_alpha else '  <-- ABOVE alpha, no FDR-significant result is possible'}\n"
            )

        for fi, name in enumerate(names):
            if name not in feat_values:
                continue
            fvals = feat_values[name]
            observed = pearson_r(isc_values, fvals)

            def statistic_fn(shifted_isc: np.ndarray, _f=fvals) -> float:
                return pearson_r(shifted_isc, _f)

            null = stats_utils.circular_shift_null(
                isc_values,
                statistic_fn,  # pyright: ignore[reportArgumentType]
            )
            p = stats_utils.exact_pvalue(np.array(observed), null, tail="two-sided")

            r_matrix[ci, fi] = observed
            rows.append(
                {
                    "stimulus": args.stimulus,
                    "component": comp,
                    "feature": name,
                    "r": observed,
                    "p_exact": float(p),
                }
            )

    if not rows:
        raise SystemExit("No features were loaded successfully; nothing to test.")

    df = pd.DataFrame(rows)
    q, sig = stats_utils.benjamini_hochberg(df["p_exact"].values, alpha=args.fdr_alpha)  # pyright: ignore[reportArgumentType]
    df["p_fdr"] = q
    df["significant"] = sig

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved stats table to {args.output_csv}")

    # -- Heatmap -----------------------------------------------------------
    kept = [n for n in names if n in feat_values]
    keep_idx = [names.index(n) for n in kept]
    r_matrix = r_matrix[:, keep_idx]
    q_matrix = np.full(r_matrix.shape, np.nan)
    for _, row in df.iterrows():
        ci = args.components.index(row["component"])
        fi = kept.index(row["feature"])
        q_matrix[ci, fi] = row["p_fdr"]

    abs_max = np.nanmax(np.abs(r_matrix))
    fig, ax = plt.subplots(
        figsize=(max(3.5, len(kept) * 1.8), max(2, len(args.components) * 0.9))
    )
    im = ax.imshow(r_matrix, aspect="auto", cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(kept)))
    ax.set_xticklabels(kept, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(args.components)))
    ax.set_yticklabels([f"Comp {c}" for c in args.components], fontsize=9)

    for ci in range(len(args.components)):
        for fi in range(len(kept)):
            q_val = q_matrix[ci, fi]
            sig_str = (
                "***" if q_val < 0.001
                else "**" if q_val < 0.01
                else "*" if q_val < 0.05
                else ""
            )
            text_color = "white" if abs(r_matrix[ci, fi]) > abs_max * 0.6 else "black"
            ax.text(
                fi, ci, f"{r_matrix[ci, fi]:.3f}{sig_str}",
                ha="center", va="center", fontsize=8, color=text_color,
            )

    ax.set_title(
        f"ISC x Stimulus Features\n{args.stimulus}",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")
    plt.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=150)
    print(f"Saved plot to {args.output_plot}")

    print("\n=== Results (FDR-corrected) ===")
    print(df.sort_values(["component", "feature"]).to_string(index=False))


if __name__ == "__main__":
    main()

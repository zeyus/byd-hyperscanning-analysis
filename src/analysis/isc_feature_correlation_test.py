"""Test whether ISC timeseries correlate with frame-level video features.

For each CCA component and each requested frame-level feature (luminance,
loudness, ...), computes the Pearson correlation between the ISC by-window
timeseries and the feature averaged over the same 5-second/1-step windows
(via stats_utils.rolling_window_mean). Significance is assessed
with an exact circular-shift permutation test: the feature series is
circularly shifted (every one of the N-1 possible shifts) relative to the
fixed ISC series and the correlation recomputed each time, building an exact
null distribution. This is valid here (unlike for the sparse/uneven emotion
event timestamps in event_locked_isc_test.py) because both series are
continuous and evenly sampled, so a circular shift preserves each series' own
autocorrelation while destroying only their relative alignment - the same
principle already used for the surrogate chance-level band in isc.py.

Benjamini-Hochberg FDR correction is applied across all (component, feature)
pairs tested for a given stimulus.

Run with: uv run python src/analysis/isc_feature_correlation_test.py \
    --stimulus bangbangyouaredead \
    --feature-csv "in/BangBangYouAreDead_composite_frame_level_analysis.csv"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import stats_utils

STIM_KEYS = {"bangbangyouaredead": "byd", "storycorps_q&a": "sc"}

DEFAULT_FEATURES = [
    "min_lum",
    "mean_lum",
    "max_lum",
    "diff_lum",
    "amp_rms",
    "amp_peak",
    "ebu_r128_M",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--feature-csv", type=Path, required=True)
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
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

    stim_key = STIM_KEYS[args.stimulus]
    if args.isc_dir is None:
        args.isc_dir = Path("out/03_ISC_results") / stim_key / args.range_tag
    if args.output_csv is None:
        args.output_csv = Path("out/04_feature_correlation") / stim_key / "isc_feature_correlation_stats.csv"
    if args.output_plot is None:
        args.output_plot = Path("out/04_feature_correlation") / stim_key / "isc_feature_correlation_stats.png"
    return args


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def load_features(
    feature_csv: Path,
    n_windows: int,
    window_sec: float,
    step_sec: float,
    t0_s: float,
    features: list[str],
) -> dict[str, np.ndarray]:
    feat_df = pd.read_csv(feature_csv)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out = {}
    for feat in features:
        if feat not in feat_df.columns:
            print(f"  Feature '{feat}' not found in {feature_csv} — skipping.")
            continue
        out[feat] = stats_utils.rolling_window_mean(
            feat_df["timestamp"].values,  # pyright: ignore[reportArgumentType]
            feat_df[feat].values,  # pyright: ignore[reportArgumentType]
            n_windows,
            window_sec,
            step_sec,
            t0_s,
        )
    return out


def main() -> None:
    args = parse_args()

    meta = stats_utils.load_isc_meta(args.isc_dir)
    window_sec = meta["window_sec"] if meta else args.window_sec
    step_sec = meta["step_sec"] if meta else args.step_sec
    t0_s = meta["t0_s"] if meta else 0.0

    rows = []
    r_matrix = np.full((len(args.components), len(args.features)), np.nan)
    p_matrix = np.full((len(args.components), len(args.features)), np.nan)

    feat_interp = None
    for ci, comp in enumerate(args.components):
        isc_path = args.isc_dir / f"isc_component{comp}_bywindow.npy"
        isc_values = stats_utils.load_isc_bywindow(isc_path)

        if feat_interp is None:
            feat_interp = load_features(
                args.feature_csv,
                len(isc_values),
                window_sec,
                step_sec,
                t0_s,
                args.features,
            )

        for fi, feat in enumerate(args.features):
            if feat not in feat_interp:
                continue
            fvals = feat_interp[feat]

            observed = pearson_r(isc_values, fvals)

            def statistic_fn(shifted_isc: np.ndarray) -> float:
                return pearson_r(shifted_isc, fvals)

            null = stats_utils.circular_shift_null(
                isc_values,
                statistic_fn,  # pyright: ignore[reportArgumentType]
            )  # (n_shifts,)
            p = stats_utils.exact_pvalue(np.array(observed), null, tail="two-sided")

            r_matrix[ci, fi] = observed
            p_matrix[ci, fi] = p
            rows.append(
                {
                    "stimulus": args.stimulus,
                    "component": comp,
                    "feature": feat,
                    "r": observed,
                    "p_exact": float(p),
                }
            )

    df = pd.DataFrame(rows)
    q, sig = stats_utils.benjamini_hochberg(df["p_exact"].values, alpha=args.fdr_alpha)  # pyright: ignore[reportArgumentType]
    df["p_fdr"] = q
    df["significant"] = sig

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved stats table to {args.output_csv}")

    # ── Heatmap ───────────────────────────────────────────────────────────
    q_matrix = df["p_fdr"].values.reshape(r_matrix.shape)  # pyright: ignore[reportAttributeAccessIssue]
    abs_max = np.nanmax(np.abs(r_matrix))
    fig, ax = plt.subplots(
        figsize=(max(3.5, len(args.features) * 1.8), max(2, len(args.components) * 0.9))
    )
    im = ax.imshow(r_matrix, aspect="auto", cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(args.features)))
    ax.set_xticklabels(args.features, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(args.components)))
    ax.set_yticklabels([f"Comp {c}" for c in args.components], fontsize=9)

    for ci in range(len(args.components)):
        for fi in range(len(args.features)):
            q_val = q_matrix[ci, fi]
            sig_str = (
                "***"
                if q_val < 0.001
                else "**"
                if q_val < 0.01
                else "*"
                if q_val < 0.05
                else ""
            )
            text_color = "white" if abs(r_matrix[ci, fi]) > abs_max * 0.6 else "black"
            ax.text(
                fi,
                ci,
                f"{r_matrix[ci, fi]:.3f}{sig_str}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    ax.set_title(
        f"Pearson r (FDR-corrected) — ISC × Stimulus Features\n{args.stimulus}",
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

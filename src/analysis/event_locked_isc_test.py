"""Test whether ISC rises around emotionally salient moments in a stimulus.

For each labeled event onset in data/stimuli_emotion_events.json, compares mean
ISC in a post-onset "response" window against a pre-onset "baseline" window,
per CCA component. Significance is assessed with an exact circular-shift
permutation test on the ISC by-window timeseries itself: the whole ISC series
is circularly shifted (every one of the N-1 possible shifts, since N is only
~200-1250 windows here) relative to the fixed event onsets, which preserves
the series' own autocorrelation/marginal distribution while destroying any
real alignment to the events. This is the same null-generating principle
already used for the per-window chance-level band in isc.py
(compute_surrogate_chance_level), just applied to the derived ISC timeseries
rather than to subject-level EEG - and, because it shifts the continuous
series rather than the sparse/uneven event timestamps, it sidesteps the
ambiguity of what "shifting a handful of discrete onsets" would even mean.

The event-average ("aggregate") statistic per component is the confirmatory,
headline test. Individual per-event statistics are exploratory and are
Benjamini-Hochberg FDR-corrected across all (component, event) pairs.

Run with: uv run python src/analysis/event_locked_isc_test.py --stimulus bangbangyouaredead --event-group byd
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import stats_utils

STIM_KEYS = {"bangbangyouaredead": "byd", "storycorps_q&a": "sc"}


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
    p.add_argument(
        "--events-json", type=Path, default=Path("data/stimuli_emotion_events.json")
    )
    p.add_argument(
        "--event-group", required=True, help="Key into the events JSON, e.g. byd or sc"
    )
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--step-sec", type=float, default=1.0)
    p.add_argument("--baseline-sec", type=float, default=5.0)
    p.add_argument("--response-sec", type=float, default=5.0)
    p.add_argument(
        "--tail", choices=["greater", "less", "two-sided"], default="greater"
    )
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to out/05_event_correlation/event_locked_isc_stats_{event-group}.csv",
    )
    p.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Defaults to out/05_event_correlation/event_locked_isc_stats_{event-group}.png",
    )
    args = p.parse_args()

    stim_key = STIM_KEYS[args.stimulus]
    if args.isc_dir is None:
        args.isc_dir = Path("out/03_ISC_results") / stim_key / args.range_tag
    if args.output_csv is None:
        args.output_csv = Path("out/05_event_correlation") / f"event_locked_isc_stats_{args.event_group}.csv"
    if args.output_plot is None:
        args.output_plot = Path("out/05_event_correlation") / f"event_locked_isc_stats_{args.event_group}.png"
    return args


_COMP_COLORS = ["black", "#cc0000", "#1f77b4", "#2ca02c", "#9467bd"]


def windowed_mean(
    isc_values: np.ndarray,
    window_times: np.ndarray,
    t_start: float,
    t_end: float,
    fine_step: float = 0.1,
) -> float:
    """Mean of `isc_values` (interpolated) over [t_start, t_end), clipped to the series' domain."""
    lo = max(t_start, window_times[0])
    hi = min(t_end, window_times[-1])
    if hi <= lo:
        return np.nan
    grid = np.arange(lo, hi, fine_step)
    return float(np.interp(grid, window_times, isc_values).mean())


def per_event_stats(
    isc_values: np.ndarray,
    window_times: np.ndarray,
    onsets: np.ndarray,
    baseline_sec: float,
    response_sec: float,
) -> np.ndarray:
    """Return (n_events,) array of response_mean - baseline_mean."""
    diffs = np.empty(len(onsets))
    for i, t0 in enumerate(onsets):
        baseline = windowed_mean(isc_values, window_times, t0 - baseline_sec, t0)
        response = windowed_mean(isc_values, window_times, t0, t0 + response_sec)
        diffs[i] = response - baseline
    return diffs


def main() -> None:
    args = parse_args()

    meta = stats_utils.load_isc_meta(args.isc_dir)
    window_sec = meta["window_sec"] if meta else args.window_sec
    step_sec = meta["step_sec"] if meta else args.step_sec
    t0_s = meta["t0_s"] if meta else 0.0

    with open(args.events_json) as f:
        stimulus_groups = json.load(f)
    try:
        group = next(
            g for g in stimulus_groups if g["stimulus"] == args.event_group
        )
    except StopIteration:
        available = [g["stimulus"] for g in stimulus_groups]
        raise ValueError(
            f"Event group {args.event_group!r} not found. Available: {available}"
        )
    event_names = [e["event"] for e in group["events"]]
    onsets = np.array([float(e["event_time_s"]) for e in group["events"]])
    print(
        f"Loaded {len(event_names)} events for group '{args.event_group}': {event_names}"
    )

    rows = []
    fig, axes = plt.subplots(
        len(args.components), 1, figsize=(13, 3.2 * len(args.components)), squeeze=False
    )

    for ci, comp in enumerate(args.components):
        isc_path = args.isc_dir / f"isc_component{comp}_bywindow.npy"
        isc_values = stats_utils.load_isc_bywindow(isc_path)
        window_times = stats_utils.reconstruct_window_times(
            len(isc_values), window_sec, step_sec, t0_s
        )

        observed = per_event_stats(
            isc_values, window_times, onsets, args.baseline_sec, args.response_sec
        )
        observed_aggregate = np.nanmean(observed)

        def statistic_fn(shifted: np.ndarray) -> np.ndarray:
            return per_event_stats(
                shifted, window_times, onsets, args.baseline_sec, args.response_sec
            )

        null = stats_utils.circular_shift_null(
            isc_values, statistic_fn
        )  # (n_shifts, n_events)
        null_aggregate = np.nanmean(null, axis=1)  # (n_shifts,)

        p_events = stats_utils.exact_pvalue(observed, null, tail=args.tail)
        p_aggregate = stats_utils.exact_pvalue(
            np.array(observed_aggregate), null_aggregate, tail=args.tail
        )

        rows.append(
            {
                "stimulus": args.stimulus,
                "component": comp,
                "event_name": "__aggregate__",
                "onset_s": np.nan,
                "baseline_mean": np.nan,
                "response_mean": np.nan,
                "diff": observed_aggregate,
                "p_exact": float(p_aggregate),
            }
        )
        for name, t0, diff, p in zip(event_names, onsets, observed, p_events):
            baseline = windowed_mean(
                isc_values, window_times, t0 - args.baseline_sec, t0
            )
            response = windowed_mean(
                isc_values, window_times, t0, t0 + args.response_sec
            )
            rows.append(
                {
                    "stimulus": args.stimulus,
                    "component": comp,
                    "event_name": name,
                    "onset_s": t0,
                    "baseline_mean": baseline,
                    "response_mean": response,
                    "diff": diff,
                    "p_exact": float(p),
                }
            )

        # ── Plot ──────────────────────────────────────────────────────────
        ax = axes[ci, 0]
        color = _COMP_COLORS[ci % len(_COMP_COLORS)]
        ax.plot(
            window_times, isc_values, color=color, linewidth=1.2, label=f"Comp {comp}"
        )
        for name, t0, p in zip(event_names, onsets, p_events):
            ax.axvline(t0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_ylabel("ISC")
        ax.set_title(
            f"Comp {comp} — {args.stimulus} "
            f"(aggregate diff={observed_aggregate:.4f}, p={p_aggregate:.4f})"
        )
        ax.set_xlim(window_times[0], window_times[-1])

    axes[-1, 0].set_xlabel("Time (s)")

    df = pd.DataFrame(rows)
    # FDR across all (component, event) pairs, excluding the aggregate rows
    per_event_mask = df["event_name"] != "__aggregate__"
    q, sig = stats_utils.benjamini_hochberg(
        df.loc[per_event_mask, "p_exact"].values,  # pyright: ignore[reportArgumentType]
        alpha=args.fdr_alpha,
    )
    df["p_fdr"] = np.nan
    df["significant"] = False
    df.loc[per_event_mask, "p_fdr"] = q
    df.loc[per_event_mask, "significant"] = sig

    # Annotate significance stars on the plot for per-event tests
    for ci, comp in enumerate(args.components):
        ax = axes[ci, 0]
        comp_rows = df[(df["component"] == comp) & per_event_mask]
        ymax = np.nanmax(axes[ci, 0].lines[0].get_ydata())
        for _, r in comp_rows.iterrows():
            if r["significant"]:
                ax.text(
                    r["onset_s"],
                    ymax * 1.05,
                    "*",
                    ha="center",
                    fontsize=14,
                    color="red",
                )
        ax.legend(loc="upper right", fontsize=9)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved stats table to {args.output_csv}")

    plt.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=150)
    print(f"Saved plot to {args.output_plot}")

    print("\n=== Aggregate (confirmatory) results ===")
    print(
        df[df["event_name"] == "__aggregate__"][
            ["component", "diff", "p_exact"]
        ].to_string(index=False)
    )
    print("\n=== Per-event (exploratory, FDR-corrected) results ===")
    print(
        df[per_event_mask][
            [
                "component",
                "event_name",
                "onset_s",
                "diff",
                "p_exact",
                "p_fdr",
                "significant",
            ]
        ]
        .sort_values(["component", "onset_s"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

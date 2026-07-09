"""Compute inter-subject correlation (ISC) for one stimulus and save results.

Scripted, non-interactive equivalent of the "Run ISC" cell in
notebooks/isc_analysis.ipynb: loads preprocessed EEG for the requested
subjects/stimulus, trains CCA (`analysis.isc.train_cca`), applies it to get a
per-window ISC timecourse (`analysis.isc.apply_cca`), optionally estimates a
per-window surrogate chance-level band (`analysis.isc.compute_surrogate_chance_level`),
and saves everything under out/03_ISC_results/{stim}/{full|segment}/, plus a
quick diagnostic plot under out/06_figures/. Unlike the notebook, the surrogate
RNG is always explicitly seeded (--seed is required) so chance-level results
are reproducible run-to-run.

Run with: uv run python src/analysis/compute_isc.py \
    --stimulus BangBangYouAreDead --seed 42 --t0 300 --t1 660
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import isc
from data import eeg

_COMP_COLORS = ["#1f77b4", "#cc0000", "#2ca02c", "#9467bd", "#8c564b"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stimulus", required=True, choices=eeg.STIMULI)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--subjects", type=int, nargs="+", default=eeg.SUBJECT_IDS)
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--t1", type=float, default=None)
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--step-sec", type=float, default=1.0)
    p.add_argument("--n-comp", type=int, default=3)
    p.add_argument(
        "--no-chance-level",
        action="store_false",
        dest="compute_chance",
        default=True,
        help="Skip the surrogate chance-level estimation.",
    )
    p.add_argument("--n-permutations", type=int, default=200)
    p.add_argument("--p-threshold", type=float, default=0.01)
    p.add_argument(
        "--surrogate-method", choices=["circular", "phase"], default="circular"
    )
    p.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed for the surrogate RNG, required for reproducible chance-level results.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path(os.getenv("EEG_WORK_DIR", "./out"))
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_data(
    data_dir: Path, subjects: list[int], stimulus: eeg.StimulusName
) -> tuple[list[np.ndarray], int]:
    arrays = []
    fs = None
    for sid in subjects:
        raw = eeg.load_preprocessed_eeg(data_dir, sid, stimulus)
        fs = int(raw.info["sfreq"])
        arrays.append(raw.get_data())  # pyright: ignore[reportArgumentType]
    assert fs is not None
    return arrays, fs


def main() -> None:
    args = parse_args()
    stim_key = eeg.STIMULUS_SHORT_KEYS[args.stimulus]
    range_tag = "full" if args.t0 is None and args.t1 is None else "segment"

    data_dir = args.data_dir or (args.out_dir / "02_preprocessed_eeg_data" / stim_key)

    isc_dir = args.out_dir / "03_ISC_results" / stim_key / range_tag
    fig_dir = args.out_dir / "06_figures"

    existing = list(isc_dir.glob("*.npy")) if isc_dir.exists() else []
    if existing and not args.force:
        raise FileExistsError(
            f"{isc_dir} already has results ({len(existing)} .npy files). "
            "Pass --force to overwrite."
        )

    print(f"Loading preprocessed EEG for {args.stimulus} from {data_dir} ...")
    arrays, fs = load_data(data_dir, args.subjects, args.stimulus)
    min_t = min(a.shape[1] for a in arrays)
    full_dur_s = min_t / fs

    t0_s = args.t0 if args.t0 is not None else 0.0
    t1_s = args.t1 if args.t1 is not None else full_dur_s
    t0, t1 = int(t0_s * fs), min(int(t1_s * fs), min_t)
    data_array = np.array([a[:, t0:t1] for a in arrays])
    print(
        f"data_array shape: {data_array.shape}, fs={fs}, range={range_tag} ({t0_s}-{t1_s}s)"
    )

    print("Training CCA...")
    W, ISC_train = isc.train_cca({args.stimulus: data_array})
    print(f"Train-set ISC (top {args.n_comp}): {ISC_train[: args.n_comp]}")

    print("Applying CCA (per-window ISC)...")
    ISC, ISC_persecond, ISC_bysubject, A, window_times = isc.apply_cca(
        data_array, W, fs, window_sec=args.window_sec, step_sec=args.step_sec
    )
    window_times = window_times + t0_s
    n_comp = min(args.n_comp, ISC.shape[0])

    chance_levels = None
    if args.compute_chance:
        print(
            f"Computing surrogate chance level ({args.surrogate_method}, seed={args.seed})..."
        )
        rng = np.random.default_rng(args.seed)
        chance_levels = isc.compute_surrogate_chance_level(
            data_array,
            W,
            fs,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            n_permutations=args.n_permutations,
            p_threshold=args.p_threshold,
            n_comp=n_comp,
            rng=rng,
            method=args.surrogate_method,
        )

    isc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for c in range(1, n_comp + 1):
        np.save(isc_dir / f"isc_component{c}_bywindow.npy", ISC_persecond[c - 1])
        if chance_levels is not None:
            np.save(isc_dir / f"chance_comp{c}.npy", chance_levels[c - 1])

    meta = {
        "stimulus": args.stimulus,
        "t0_s": t0_s,
        "t1_s": t1_s,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "fs": fs,
        "n_comp": n_comp,
        "subject_ids": args.subjects,
        "compute_chance": args.compute_chance,
        "surrogate_method": args.surrogate_method if args.compute_chance else None,
        "n_permutations": args.n_permutations if args.compute_chance else None,
        "p_threshold": args.p_threshold if args.compute_chance else None,
        "seed": args.seed,
    }
    with open(isc_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved ISC results + meta.json to {isc_dir}")

    # ── Diagnostic plot (quick sanity check, not the polished R figure) ────
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]}
    )
    for c in range(n_comp):
        color = _COMP_COLORS[c % len(_COMP_COLORS)]
        if chance_levels is not None:
            ax_top.fill_between(
                window_times, 0, chance_levels[c], color=color, alpha=0.15
            )
        ax_top.plot(
            window_times,
            ISC_persecond[c],
            color=color,
            linewidth=1.2,
            label=f"Comp {c + 1}",
        )
    if range_tag == "segment":
        ax_top.axvline(t0_s, color="gray", linestyle="--", linewidth=0.8)
        ax_top.axvline(t1_s, color="gray", linestyle="--", linewidth=0.8)
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("ISC")
    ax_top.set_title(f"ISC per window — {args.stimulus} ({range_tag})")
    ax_top.legend(loc="upper right", fontsize=9)

    im = ax_bottom.imshow(ISC_bysubject[:n_comp], aspect="auto", cmap="RdBu_r")
    ax_bottom.set_yticks(range(n_comp))
    ax_bottom.set_yticklabels([f"Comp {c + 1}" for c in range(n_comp)])
    ax_bottom.set_xlabel("Subject index (leave-one-out)")
    ax_bottom.set_title("ISC by subject")
    plt.colorbar(im, ax=ax_bottom, fraction=0.03, pad=0.02)

    plt.tight_layout()
    fig_path = fig_dir / f"isc_diagnostic_{stim_key}_{range_tag}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic plot to {fig_path}")


if __name__ == "__main__":
    main()

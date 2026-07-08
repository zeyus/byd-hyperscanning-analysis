"""Empirical stability analysis for the surrogate chance-level permutation count.

Trains CCA on the BYD comparison segment (300-660s, all 10 subjects, component 1)
and re-estimates the per-window 99th-percentile chance-level threshold at several
permutation counts and repeated random seeds, to characterise how much the
threshold estimate varies as a function of n_permutations.

Run with: uv run python src/analysis/permutation_stability.py
"""

import json
from pathlib import Path

import numpy as np

from analysis import isc
from data import eeg

DATA_DIR = Path("out/02_preprocessed_eeg_data/byd")
STIMULUS: eeg.StimulusName = "BangBangYouAreDead"
T0_S, T1_S = 300.0, 660.0  # comparison segment used in the thesis
WINDOW_SEC, STEP_SEC = 5.0, 1.0
N_COMP = 1

N_PERMUTATIONS_GRID = [25, 50, 100, 200, 400, 800, 1600]
N_REPEATS = 10
REFERENCE_N_PERMUTATIONS = 3200
REFERENCE_REPEATS = 10

OUT_PATH = Path("out/06_figures/permutation_stability_results.json")


def load_segment() -> tuple[np.ndarray, int]:
    arrays = []
    fs = None
    for sid in eeg.SUBJECT_IDS:
        raw = eeg.load_preprocessed_eeg(DATA_DIR, sid, STIMULUS)
        fs = int(raw.info["sfreq"])
        t0 = int(T0_S * fs)
        t1 = int(T1_S * fs)
        data = raw.get_data()[:, t0:t1]  # pyright: ignore[reportCallIssue, reportArgumentType]
        arrays.append(data)
    assert fs is not None
    min_t = min(a.shape[1] for a in arrays)
    data_array = np.array([a[:, :min_t] for a in arrays])
    return data_array, fs


def main() -> None:
    print("Loading BYD segment for all subjects...")
    data_array, fs = load_segment()
    print(f"data_array shape: {data_array.shape}, fs={fs}")

    print("Training CCA on the segment...")
    W, ISC_train = isc.train_cca({STIMULUS: data_array})
    print(f"Train-set ISC (top {N_COMP}): {ISC_train[:N_COMP]}")

    results: dict[str, list | float | int] = {
        "n_permutations": [],
        "repeat": [],
        "seed": [],
        "threshold": [],
    }

    def run_once(n_perm: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        chance = isc.compute_surrogate_chance_level(
            data_array,
            W,
            fs,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            n_permutations=n_perm,
            p_threshold=0.01,
            n_comp=N_COMP,
            rng=rng,
        )
        return chance[0]  # (n_windows,) for component 1

    print(
        f"\nComputing reference threshold at n_permutations={REFERENCE_N_PERMUTATIONS} "
        f"({REFERENCE_REPEATS} repeats, averaged)..."
    )
    ref_runs = [
        run_once(REFERENCE_N_PERMUTATIONS, seed=100000 + r)
        for r in range(REFERENCE_REPEATS)
    ]
    reference = np.mean(ref_runs, axis=0)

    seed_counter = 0
    for n_perm in N_PERMUTATIONS_GRID:
        print(f"\nn_permutations = {n_perm}")
        for r in range(N_REPEATS):
            seed = 1000 * n_perm + r
            thr = run_once(n_perm, seed=seed)
            results["n_permutations"].append(n_perm)  # pyright: ignore[reportAttributeAccessIssue]
            results["repeat"].append(r)  # pyright: ignore[reportAttributeAccessIssue]
            results["seed"].append(seed)  # pyright: ignore[reportAttributeAccessIssue]
            results["threshold"].append(thr.tolist())  # pyright: ignore[reportAttributeAccessIssue]
            seed_counter += 1
            print(f"  repeat {r}: mean_thr={thr.mean():.5f}")

    results["reference_n_permutations"] = REFERENCE_N_PERMUTATIONS
    results["reference_threshold"] = reference.tolist()
    results["window_sec"] = WINDOW_SEC
    results["step_sec"] = STEP_SEC
    results["p_threshold"] = 0.01

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"\nSaved raw results to {OUT_PATH}")

    # ---- Summary stats ----
    print("\n=== Summary ===")
    print(
        f"{'n_perm':>8} | {'mean(SD across repeats)':>24} | {'mean|bias vs ref|':>18}"
    )
    for n_perm in N_PERMUTATIONS_GRID:
        idx = [i for i, n in enumerate(results["n_permutations"]) if n == n_perm]  # pyright: ignore[reportArgumentType]
        thrs = np.array(
            [results["threshold"][i] for i in idx]  # pyright: ignore[reportIndexIssue]
        )  # (repeats, n_windows)
        per_window_sd = thrs.std(axis=0, ddof=1)
        mean_sd = per_window_sd.mean()
        mean_curve = thrs.mean(axis=0)
        bias = np.abs(mean_curve - reference).mean()
        print(f"{n_perm:>8} | {mean_sd:>24.5f} | {bias:>18.5f}")


if __name__ == "__main__":
    main()

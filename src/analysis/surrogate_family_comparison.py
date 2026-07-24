"""Compare the two surrogate families used to set the ISC chance level.

Companion to `permutation_stability.py`. That script asked how many circular-shift
permutations are needed; this one asks whether the *family* of surrogate matters:

* ``circular`` -- independent circular time-shift per subject (used throughout the
  thesis), and
* ``phase``    -- Prichard & Theiler (1994) multivariate phase randomization, the
  family most likely used by Poulsen et al. (2017).

Both target the same null (destroy cross-subject alignment, keep each subject's own
structure). The question is whether swapping one for the other moves the per-window
chance band enough to change the RQ1 replication conclusion, i.e. criterion (i): the
proportion of segment windows exceeding the band, tested against pi_0 = 0.01 with a
one-sided exact binomial test.

CCA is trained once and both families are evaluated against the *same* W and the same
observed ISC timecourse, so the only thing that differs between arms is the surrogate
construction. Several seeds are run per family so that the between-family difference
can be read against the seed-to-seed Monte Carlo noise at the same n_permutations.

Also reports, per component, the multiplicative "breakdown factor" lambda*: how much
the chance band would have to be inflated before criterion (i) stops rejecting. This
bounds the impact of the family choice without reference to either family.

Run with: uv run python src/analysis/surrogate_family_comparison.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from analysis import isc
from data import eeg

DATA_DIR = Path("out/02_preprocessed_eeg_data/byd")
STIMULUS: eeg.StimulusName = "BangBangYouAreDead"
# Comparison segment as written by compute_isc.py (see
# out/03_ISC_results/byd/segment/meta.json), NOT the 300-660 s window that the
# older permutation_stability.py hardcodes.
T0_S, T1_S = 296.667, 669.667
WINDOW_SEC, STEP_SEC = 5.0, 1.0
N_COMP = 3
P_THRESHOLD = 0.01
PI_0 = 0.01  # null proportion for criterion (i)
ALPHA = 0.05

METHODS = ["circular", "phase"]
# Matched-n arm: same n_permutations as the reported analysis, several seeds, so the
# family difference can be compared against Monte Carlo noise at that n.
N_PERM_MATCHED = 200
N_SEEDS_MATCHED = 10
# Low-noise arm: high n so the family difference is not confounded with MC error.
N_PERM_HIGH = 1600
N_SEEDS_HIGH = 3

OUT_PATH = Path("out/06_figures/surrogate_family_comparison.json")


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
    return np.array([a[:, :min_t] for a in arrays]), fs


def binom_p(k: int, n: int) -> float:
    return float(binomtest(k, n, PI_0, alternative="greater").pvalue)


def critical_count(n: int) -> int:
    """Smallest supra-threshold count that still rejects pi_0 at ALPHA."""
    return next(k for k in range(n + 1) if binom_p(k, n) < ALPHA)


def breakdown_factor(observed: np.ndarray, chance: np.ndarray, k_star: int) -> float:
    """Largest band inflation lambda for which criterion (i) still rejects.

    The supra-threshold count is monotonically non-increasing in lambda, so a
    bisection on lambda is well defined.
    """
    lo, hi = 1.0, 100.0
    if int((observed > chance * hi).sum()) >= k_star:
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if int((observed > chance * mid).sum()) >= k_star:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    print(f"Loading BYD segment ({T0_S}-{T1_S} s) for all subjects...")
    data_array, fs = load_segment()
    print(f"data_array shape: {data_array.shape}, fs={fs}")

    print("Training CCA on the segment...")
    W, ISC_train = isc.train_cca({STIMULUS: data_array})
    print(f"Train-set ISC (top {N_COMP}): {ISC_train[:N_COMP]}")

    print("Applying CCA (observed per-window ISC)...")
    _, ISC_persecond, _, _, _ = isc.apply_cca(
        data_array, W, fs, window_sec=WINDOW_SEC, step_sec=STEP_SEC
    )
    observed = ISC_persecond[:N_COMP]  # (N_COMP, n_windows)
    n_windows = observed.shape[1]
    k_star = critical_count(n_windows)
    print(f"n_windows={n_windows}, critical count at alpha={ALPHA}: k*={k_star}")

    def run_once(method: str, n_perm: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return isc.compute_surrogate_chance_level(
            data_array,
            W,
            fs,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            n_permutations=n_perm,
            p_threshold=P_THRESHOLD,
            n_comp=N_COMP,
            rng=rng,
            method=method,
        )  # (N_COMP, n_windows)

    runs: list[dict] = []
    for arm, n_perm, n_seeds, seed_base in [
        ("matched", N_PERM_MATCHED, N_SEEDS_MATCHED, 7000),
        ("high", N_PERM_HIGH, N_SEEDS_HIGH, 9000),
    ]:
        for method in METHODS:
            for r in range(n_seeds):
                seed = seed_base + 100 * METHODS.index(method) + r
                print(f"\n[{arm}] {method}, n_perm={n_perm}, seed={seed}")
                chance = run_once(method, n_perm, seed)
                rec = {
                    "arm": arm,
                    "method": method,
                    "n_permutations": n_perm,
                    "seed": seed,
                    "threshold": chance.tolist(),
                    "supra_count": [
                        int((observed[c] > chance[c]).sum()) for c in range(N_COMP)
                    ],
                }
                rec["binom_p"] = [
                    binom_p(k, n_windows)
                    for k in rec["supra_count"]  # pyright: ignore[reportArgumentType]
                ]
                runs.append(rec)
                print(
                    f"  mean_thr(comp1)={chance[0].mean():.5f} "
                    f"supra={rec['supra_count']} "
                    f"p(comp1)={rec['binom_p'][0]:.3e}"  # pyright: ignore[reportIndexIssue]
                )

    results = {
        "stimulus": STIMULUS,
        "t0_s": T0_S,
        "t1_s": T1_S,
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "fs": fs,
        "n_comp": N_COMP,
        "p_threshold": P_THRESHOLD,
        "pi_0": PI_0,
        "alpha": ALPHA,
        "n_windows": n_windows,
        "critical_count": k_star,
        "observed_isc": observed.tolist(),
        "runs": runs,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"\nSaved raw results to {OUT_PATH}")

    # ---- Summary ----
    def arm_runs(arm: str, method: str) -> list[dict]:
        return [r for r in runs if r["arm"] == arm and r["method"] == method]

    for arm in ("matched", "high"):
        n_perm = N_PERM_MATCHED if arm == "matched" else N_PERM_HIGH
        print(f"\n=== {arm} arm (n_permutations = {n_perm}) ===")
        thr = {
            m: np.array([r["threshold"] for r in arm_runs(arm, m)]) for m in METHODS
        }  # (seeds, N_COMP, n_windows)
        for c in range(N_COMP):
            print(f"\n-- component {c + 1} --")
            for m in METHODS:
                curves = thr[m][:, c, :]
                counts = [r["supra_count"][c] for r in arm_runs(arm, m)]
                ps = [r["binom_p"][c] for r in arm_runs(arm, m)]
                # SD across seeds of the per-window threshold, averaged over windows:
                # the Monte Carlo noise floor at this n_permutations.
                sd = (
                    curves.std(axis=0, ddof=1).mean() if curves.shape[0] > 1 else np.nan
                )
                print(
                    f"  {m:>8}: mean_thr={curves.mean():.5f}  MC_sd={sd:.5f}  "
                    f"supra={min(counts)}-{max(counts)}/{n_windows}  "
                    f"max_p={max(ps):.3e}  rejects={all(p < ALPHA for p in ps)}"
                )
            # Paired per-window family difference, averaged over seeds within family.
            mean_circ = thr["circular"][:, c, :].mean(axis=0)
            mean_phase = thr["phase"][:, c, :].mean(axis=0)
            delta = mean_phase - mean_circ
            pct = 100.0 * delta.mean() / mean_circ.mean()
            stricter = "phase" if delta.mean() > 0 else "circular"
            mc_sd = thr["circular"][:, c, :].std(axis=0, ddof=1).mean()
            print(
                f"  delta (phase - circular): mean={delta.mean():+.5f} ISC "
                f"({pct:+.1f}% of circular), median={np.median(delta):+.5f}, "
                f"stricter family = {stricter}"
            )
            print(f"  |delta| / circular MC sd at this n = {abs(delta.mean()) / mc_sd:.2f}")
            for m, curve in (("circular", mean_circ), ("phase", mean_phase)):
                lam = breakdown_factor(observed[c], curve, k_star)
                print(f"  breakdown lambda* ({m} band) = {lam:.3f}x")


if __name__ == "__main__":
    main()

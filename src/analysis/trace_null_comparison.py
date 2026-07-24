"""Robustness of the RQ1 criterion (ii) p-value to the choice of trace-level null.

Criterion (ii) correlates the present study's component-1 ISC trace with each
digitised reference trace (Dmochowski et al. 2012; Poulsen et al. 2017) over the
BYD comparison segment, and tests the correlation against an *exhaustive*
circular-shift null (all T-1 shifts). This script re-tests the same observed
correlations against a phase-randomised null, which preserves each trace's
amplitude spectrum (hence autocorrelation) but is continuous rather than discrete,
so the p-value is not floored at 1/T.

Both nulls represent the same alternative: "two traces with this temporal structure,
not aligned in time". They differ only in how misalignment is produced.

Run with: uv run python src/analysis/trace_null_comparison.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

COMPARISON_CSV = Path("../masters_thesis/data/isc_comparison.csv")
REFERENCES = {
    "Dmochowski et al. (2012)": "dmochowski_timeseries",
    "Poulsen et al. (2017)": "poulsen_timeseries",
}
OBSERVED_COL = "lab_results"
N_PHASE_SURROGATES = 10000
SEED = 2026
OUT_PATH = Path("out/06_figures/trace_null_comparison.json")


def phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Univariate phase randomisation (Theiler et al. 1992).

    Preserves the amplitude spectrum, and hence the autocorrelation, of ``x``;
    randomises the phases, destroying any alignment with another series. The
    Nyquist bin (even-length input) keeps a real coefficient so the surrogate is
    real-valued.
    """
    n = x.shape[0]
    xf = np.fft.rfft(x - x.mean())
    phases = np.zeros(xf.shape[0])
    hi = xf.shape[0] - 1 if n % 2 == 0 else xf.shape[0]
    phases[1:hi] = rng.uniform(-np.pi, np.pi, size=hi - 1)
    return np.fft.irfft(xf * np.exp(1j * phases), n=n) + x.mean()


def main() -> None:
    df = pd.read_csv(COMPARISON_CSV)
    observed = df[OBSERVED_COL].to_numpy(dtype=float)
    t = observed.shape[0]
    rng = np.random.default_rng(SEED)
    print(f"Loaded {COMPARISON_CSV} (T = {t})")

    out: dict = {"T": t, "n_phase_surrogates": N_PHASE_SURROGATES, "seed": SEED}

    for label, col in REFERENCES.items():
        ref = df[col].to_numpy(dtype=float)
        r_obs = float(pearsonr(observed, ref).statistic)
        rho_obs = float(spearmanr(observed, ref).statistic)

        # (a) exhaustive circular shift: all T-1 non-identity shifts.
        shift_null = np.array(
            [
                float(pearsonr(np.roll(observed, s), ref).statistic)
                for s in range(1, t)
            ]
        )
        p_shift = (1 + int((shift_null >= r_obs).sum())) / t

        # (b) phase-randomised null on the present study's trace.
        phase_null = np.array(
            [
                float(pearsonr(phase_randomise(observed, rng), ref).statistic)
                for _ in range(N_PHASE_SURROGATES)
            ]
        )
        p_phase = (1 + int((phase_null >= r_obs).sum())) / (N_PHASE_SURROGATES + 1)

        out[label] = {
            "r": r_obs,
            "rho": rho_obs,
            "p_circular_exact": p_shift,
            "p_phase": p_phase,
            "shift_null_mean": float(shift_null.mean()),
            "shift_null_sd": float(shift_null.std(ddof=1)),
            "shift_null_q99": float(np.quantile(shift_null, 0.99)),
            "phase_null_mean": float(phase_null.mean()),
            "phase_null_sd": float(phase_null.std(ddof=1)),
            "phase_null_q99": float(np.quantile(phase_null, 0.99)),
        }

        print(f"\n=== {label} ===")
        print(f"  r = {r_obs:.4f}   rho = {rho_obs:.4f}")
        print(
            f"  circular (exact, {t - 1} shifts): p = {p_shift:.4f} "
            f"(floor 1/{t} = {1 / t:.4f}); null mean={shift_null.mean():+.4f} "
            f"sd={shift_null.std(ddof=1):.4f} q99={np.quantile(shift_null, 0.99):+.4f}"
        )
        print(
            f"  phase ({N_PHASE_SURROGATES} surrogates):    p = {p_phase:.5f}; "
            f"null mean={phase_null.mean():+.4f} "
            f"sd={phase_null.std(ddof=1):.4f} q99={np.quantile(phase_null, 0.99):+.4f}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()

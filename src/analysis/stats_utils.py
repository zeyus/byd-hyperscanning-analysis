"""Shared helpers for circular-shift permutation testing of ISC by-window timeseries.

Both the event-locked peak test and the ISC-feature correlation test need to:
- regenerate the window center times that `isc.apply_cca` used (not saved to disk)
- build an exact null distribution by circularly shifting a timeseries
- control the false discovery rate across a family of tests

Kept here so the two CLI scripts don't duplicate this logic.
"""

from pathlib import Path
from typing import Callable

import numpy as np


def reconstruct_window_times(
    n_windows: int,
    window_sec: float,
    step_sec: float,
    t0_s: float = 0.0,
) -> np.ndarray:
    """Reproduce `isc.apply_cca`'s window center-time formula.

    `apply_cca` computes `window_times[i] = (t + t_end) / 2 / fs` for windows
    starting at sample `t = i * step_samples`, which in seconds is
    `i * step_sec + window_sec / 2`, offset by the analysis start time.
    """
    i = np.arange(n_windows)
    return t0_s + i * step_sec + window_sec / 2.0


def load_isc_bywindow(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"ISC by-window array not found: {path}\n"
            "Run the ISC cell in notebooks/isc_analysis.ipynb (or the relevant "
            "compute script) first to generate it."
        )
    return np.load(path)


def circular_shift_null(
    series: np.ndarray,
    statistic_fn: Callable[[np.ndarray], np.ndarray],
    exhaustive: bool = True,
    n_shifts: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a null distribution of `statistic_fn` under circular shifts of `series`.

    Shift 0 (the observed alignment) is excluded. By default every one of the
    N-1 distinct circular shifts is used (exact/exhaustive test) since N is at
    most ~1250 for these ISC arrays, making this cheap. Pass `exhaustive=False`
    and `n_shifts` to random-subsample shifts instead, for much longer series.

    `statistic_fn` is applied to each shifted copy of `series` and must return
    either a scalar or a fixed-shape array (e.g. one value per event/feature);
    results are stacked along a new leading axis.
    """
    n = len(series)
    if exhaustive:
        shifts = np.arange(1, n)
    else:
        if n_shifts is None:
            raise ValueError("n_shifts is required when exhaustive=False")
        if rng is None:
            rng = np.random.default_rng()
        shifts = rng.integers(1, n, size=n_shifts)

    null_stats = [statistic_fn(np.roll(series, shift)) for shift in shifts]
    return np.stack(null_stats, axis=0)


def exact_pvalue(observed: np.ndarray, null: np.ndarray, tail: str = "two-sided") -> np.ndarray:
    """Exact rank-based p-value: proportion of null values at least as extreme as observed.

    `null` has shape (n_shifts, ...) matching `observed`'s shape. The observed
    value itself is included in the comparison set (add-one correction), so
    p-values are never exactly 0 and remain valid even for small null sizes.
    """
    n = null.shape[0]
    if tail == "greater":
        count = np.sum(null >= observed, axis=0)
    elif tail == "less":
        count = np.sum(null <= observed, axis=0)
    elif tail == "two-sided":
        count = np.sum(np.abs(null) >= np.abs(observed), axis=0)
    else:
        raise ValueError(f"Unknown tail: {tail!r} (expected 'greater', 'less', 'two-sided')")
    return (count + 1) / (n + 1)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns (q_values, significant_mask), both matching `pvals`'s shape.
    """
    pvals = np.asarray(pvals)
    shape = pvals.shape
    flat = pvals.ravel()
    m = len(flat)
    order = np.argsort(flat)
    ranked = flat[order]

    q_sorted = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity (running minimum from the largest p-value down)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q_values = np.empty(m)
    q_values[order] = q_sorted

    significant = q_values <= alpha
    return q_values.reshape(shape), significant.reshape(shape)

# ICA https://github.com/ML-D00M/ISC-Inter-Subject-Correlations/blob/main/Python/ISC.py
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm


def train_cca(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Run Correlated Component Analysis on your training data.

    Parameters:
    ----------
    data : dict
        Dictionary with keys are names of conditions and values are numpy
        arrays structured like (subjects, channels, samples).
        The number of channels must be the same between all conditions!

    Returns:
    -------
    W : np.array
        Columns are spatial filters. They are sorted in descending order, it means that first column-vector maximize
        correlation the most.
    ISC : np.array
        Inter-subject correlation sorted in descending order

    """

    # start = default_timer()

    C = len(data.keys())
    # st.write(f"train_cca - calculations started. There are {C} conditions")

    gamma = 0.1
    Rw: np.ndarray | None = None
    Rb: np.ndarray | None = None
    for c, cond in tqdm(data.items(), desc="Conditions"):
        (
            N,
            D,
            T,
        ) = cond.shape
        # st.write(f"Condition '{c}' has {N} subjects, {D} sensors and {T} samples")
        cond = cond.reshape(D * N, T)

        # Rij
        Rij = np.swapaxes(np.reshape(np.cov(cond), (N, D, N, D)), 1, 2)

        # Rw
        rw_blocks = np.empty((N, D, D))
        for i in tqdm(range(N), desc="Rw blocks", leave=False):
            rw_blocks[i] = Rij[i, i, :, :]
        Rw = (Rw if Rw else 0) + np.mean(rw_blocks, axis=0)

        # Rb
        rb_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        rb_blocks = np.empty((len(rb_pairs), D, D))
        for k, (i, j) in enumerate(tqdm(rb_pairs, desc="Rb blocks", leave=False)):
            rb_blocks[k] = Rij[i, j, :, :]
        Rb = (Rb if Rb else 0) + np.mean(rb_blocks, axis=0)

    if Rw is None or Rb is None:
        raise ValueError(
            "Rw or Rb was not computed. Check if data is provided correctly."
        )

    # Divide by number of condition
    Rw, Rb = Rw / C, Rb / C

    # Regularization
    Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # ISCs and Ws
    [ISC, W] = eigh(Rb, Rw_reg)

    # Make descending order
    ISC, W = ISC[::-1], W[:, ::-1]

    # stop = default_timer()

    # st.write(f"Elapsed time: {round(stop - start)} seconds.")
    return W, ISC


def apply_cca(
    X: np.ndarray,
    W: np.ndarray,
    fs: int,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    Cz_index: int | None = None,
):
    """Applying precomputed spatial filters to your data.

    Parameters:
    ----------
    X : ndarray
        3-D numpy array structured like (subject, channel, sample)
    W : ndarray
        Spatial filters.
    fs : int
        Frequency sampling.
    window_sec : int or float, optional
        Window size in seconds for ISC_persecond calculation. Default is 5.
    step_sec : int or float, optional
        Step size in seconds between windows for ISC_persecond. Default is 1.
    Returns:
    -------
    ISC : ndarray
        Inter-subject correlations values are sorted in descending order.
    ISC_persecond : ndarray
        Inter-subject correlations per window, shape (n_components, n_windows).
    ISC_bysubject : ndarray
        ISC values per component per subject.
    A : ndarray
        Scalp projections of ISC.
    window_times : ndarray
        Center time (in seconds) of each window in ISC_persecond.
    Cz_index: int, optional
        if provided, ensure that this channel polarity is positive.
    """

    # start = default_timer()
    # st.write("apply_cca - calculations started")

    N, D, T = X.shape
    # gamma = 0.1
    X = X.reshape(D * N, T)

    # Rij
    Rij = np.swapaxes(np.reshape(np.cov(X), (N, D, N, D)), 1, 2)

    # Rw
    rw_blocks = np.empty((N, D, D))
    for i in range(N):
        rw_blocks[i] = Rij[i, i, :, :]
    Rw = np.mean(rw_blocks, axis=0)
    # Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # Rb
    rb_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    rb_blocks = np.empty((len(rb_pairs), D, D))
    for k, (i, j) in enumerate(rb_pairs):
        rb_blocks[k] = Rij[i, j, :, :]
    Rb = np.mean(rb_blocks, axis=0)

    # ISCs
    ISC = np.sort(
        np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)
    )[::-1]

    # Scalp projections
    A = np.linalg.solve((np.transpose(W) @ Rw @ W).T, (Rw @ W).T).T

    # ISC by subject
    # st.write("by subject is calculating")
    ISC_bysubject = np.empty((D, N))

    for subj_k in tqdm(range(N), desc="ISC by subject"):
        rw_blocks = np.empty((N - 1, D, D))
        rb_blocks = np.empty((N - 1, D, D))
        k = 0
        for subj_l in range(N):
            if subj_l == subj_k:
                continue
            rw_blocks[k] = (
                1 / (N - 1) * (Rij[subj_k, subj_k, :, :] + Rij[subj_l, subj_l, :, :])
            )
            rb_blocks[k] = (
                1 / (N - 1) * (Rij[subj_k, subj_l, :, :] + Rij[subj_l, subj_k, :, :])
            )
            k += 1
        Rw = np.mean(rw_blocks, axis=0)
        Rb = np.mean(rb_blocks, axis=0)

        ISC_bysubject[:, subj_k] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(
            np.transpose(W) @ Rw @ W
        )

    # ISC per second
    # st.write("by persecond is calculating")
    step_samples = max(1, int(step_sec * fs))
    window_samples = int(window_sec * fs)
    n_windows = max(0, (T - window_samples) // step_samples + 1)
    ISC_persecond = np.empty((D, n_windows))
    window_times = np.empty(n_windows)
    window_i = 0

    # Pre-compute index pairs for Rw/Rb blocks (same structure every window)
    rw_idx = list(range(0, D * N, D))
    rb_pairs_t = [
        (i, j) for i in range(0, D * N, D) for j in range(0, D * N, D) if i != j
    ]
    n_rw = len(rw_idx)
    n_rb = len(rb_pairs_t)

    for t in tqdm(
        range(0, T - window_samples + 1, step_samples), desc="ISC per window"
    ):
        t_end = t + window_samples
        Xt = X[:, t:t_end]
        if Xt.shape[1] < 2:
            break
        Rij = np.cov(Xt)

        rw_blocks_t = np.empty((n_rw, D, D))
        for idx, i in enumerate(rw_idx):
            rw_blocks_t[idx] = Rij[i : i + D, i : i + D]
        Rw = np.mean(rw_blocks_t, axis=0)

        rb_blocks_t = np.empty((n_rb, D, D))
        for k, (i, j) in enumerate(rb_pairs_t):
            rb_blocks_t[k] = Rij[i : i + D, j : j + D]
        Rb = np.mean(rb_blocks_t, axis=0)

        ISC_persecond[:, window_i] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(
            np.transpose(W) @ Rw @ W
        )
        window_times[window_i] = (t + t_end) / 2 / fs  # center time in seconds
        window_i += 1

    # stop = default_timer()
    # st.write(f"Elapsed time: {round(stop - start)} seconds.")

    # Trim to actual number of windows computed
    ISC_persecond = ISC_persecond[:, :window_i]
    window_times = window_times[:window_i]

    return ISC, ISC_persecond, ISC_bysubject, A, window_times


def _phase_randomize_multivariate(
    y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Prichard & Theiler (1994) multivariate phase-randomized surrogate.

    Generates a surrogate of a multivariate signal that preserves each row's
    amplitude spectrum (hence its autocorrelation / power spectrum) and the
    *cross-row* covariance structure, while destroying absolute timing.  A
    single random phase sequence is drawn and added to every row at each
    frequency (Theiler's "same phase across variables" trick), so the phase
    *differences* between rows -- and therefore their covariance -- are left
    intact.  Applied independently to each subject, this destroys only the
    between-subject alignment that drives ISC, giving the ISC null.

    This is the surrogate family used by Poulsen et al. (2017); operating here
    on the projected components (rather than the raw channels of the original
    MATLAB implementation) is equivalent, because applying a common per-subject
    phase across the components of ``W'X`` is, by linearity of the projection,
    the same as applying it across the channels of ``X`` and then projecting.

    Reference: Prichard D, Theiler J. Generating surrogate data for time series
    with several simultaneously measured variables. Phys Rev Lett. 1994;73(7):951.

    Parameters
    ----------
    y : ndarray, shape (M, T)
        Real-valued multivariate signal (e.g. components x samples for one
        subject).  Randomization is over the T axis; the M rows share phases.
    rng : np.random.Generator
        Random generator supplying the phases.

    Returns
    -------
    ndarray, shape (M, T)
        Real-valued surrogate with the same per-row amplitude spectrum and the
        same cross-row covariance as ``y``.
    """
    _M, T = y.shape
    Yf = np.fft.rfft(y, axis=1)  # (M, n_freq) complex
    n_freq = Yf.shape[1]

    # Random phases in (-pi, pi], shared across the M rows, one per frequency.
    # DC (bin 0) is always left untouched; for even T the Nyquist bin (last)
    # must stay real, so it is left untouched too.
    phases = np.zeros(n_freq)
    hi = n_freq - 1 if (T % 2 == 0) else n_freq
    phases[1:hi] = rng.uniform(-np.pi, np.pi, size=hi - 1)

    Yf = Yf * np.exp(1j * phases)[np.newaxis, :]
    return np.fft.irfft(Yf, n=T, axis=1)


def compute_surrogate_chance_level(
    X: np.ndarray,
    W: np.ndarray,
    fs: int,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    n_permutations: int = 200,
    p_threshold: float = 0.01,
    n_comp: int | None = None,
    rng: np.random.Generator | None = None,
    method: str = "circular",
) -> np.ndarray:
    """Estimate per-window chance-level ISC via surrogate data.

    Two surrogate families are available via ``method``; both target the *same*
    null hypothesis -- each subject keeps its own temporal and spatial structure
    while cross-subject alignment (the source of genuine ISC) is destroyed -- and
    differ only in how they do it:

    * ``"circular"`` (default): each subject's projected timeseries is
      independently circularly shifted by a random amount.  Preserves the exact
      waveform, amplitude distribution, and non-stationarity of each subject;
      destroys alignment by rigid time translation.
    * ``"phase"``: each subject is independently phase-randomized following
      Prichard & Theiler (1994), the family used by Poulsen et al. (2017).
      Preserves each subject's amplitude spectrum (hence autocorrelation) and
      cross-component covariance, but replaces the signal with the equivalent
      linear-Gaussian process, so transients and non-linear/non-stationary
      structure are washed out.

    ISC is computed on the surrogate data for every window position across many
    permutations to build a per-window null distribution.  Which family gives a
    stricter (higher) threshold is data-dependent and should be treated as an
    empirical result, not assumed in advance.

    The returned threshold varies over time (one value per window), matching
    the grey area shown in Poulsen et al. (2017): "chance levels for ISC
    (p > 0.01 estimated with time-shuffled surrogate data, uncorrected for
    multiple comparisons)".

    Parameters
    ----------
    X : ndarray, shape (N, D, T)
        EEG data: N subjects, D channels, T samples.
    W : ndarray, shape (D, D)
        Spatial filters from :func:`train_cca`.
    fs : int
        Sampling frequency in Hz.
    window_sec : float
        Analysis window length in seconds.  Must match the :func:`apply_cca`
        call whose output you want to threshold.
    step_sec : float
        Step size in seconds.  Must match the :func:`apply_cca` call.
    n_permutations : int
        Number of surrogate permutations.  200 is usually sufficient for
        p = 0.01; use ≥ 500 for p = 0.001.
    p_threshold : float
        Significance level (default ``0.01``).  At each window, values above
        the returned threshold occur less than ``p_threshold * 100 %`` of the
        time by chance.
    n_comp : int or None
        Number of CCA components to evaluate.  Defaults to all columns of W.
    rng : np.random.Generator or None
        Optional seeded RNG for reproducible results, e.g.
        ``np.random.default_rng(42)``.
    method : {"circular", "phase"}
        Surrogate family (see above).  ``"circular"`` is the time-shift method;
        ``"phase"`` is Prichard & Theiler (1994) multivariate phase
        randomization, matching Poulsen et al. (2017).

    Returns
    -------
    chance_level : ndarray, shape (n_comp, n_windows)
        Per-component, per-window ISC threshold.  Plot
        ``fill_between(times, 0, chance_level[c])`` to draw the grey band
        used in Poulsen et al. (2017).
    """
    if rng is None:
        rng = np.random.default_rng()
    if method not in ("circular", "phase"):
        raise ValueError(f"Unknown method {method!r}; expected 'circular' or 'phase'.")

    N, _D, T = X.shape
    window_samples = int(window_sec * fs)
    step_samples = max(1, int(step_sec * fs))

    if n_comp is None:
        n_comp = W.shape[1]
    Wc = W[:, :n_comp]  # (D, n_comp)

    # Project all subjects into component space once: (N, n_comp, T)
    Y = np.einsum("dc,ndt->nct", Wc, X)

    # For circular shifts, require shifts large enough to avoid temporal
    # self-overlap.  Phase randomization has no such constraint.
    min_shift = window_samples
    max_shift = T - min_shift
    if method == "circular" and max_shift <= min_shift:
        raise ValueError(
            f"Recording too short ({T} samples) for circular-shift surrogates "
            f"with window_samples={window_samples}. Use a shorter window."
        )

    # Pre-build index pairs once
    rw_slices = [(i * n_comp, (i + 1) * n_comp) for i in range(N)]  # pyright: ignore[reportOperatorIssue]
    rb_pairs_idx = [
        (i * n_comp, (i + 1) * n_comp, j * n_comp, (j + 1) * n_comp)  # pyright: ignore[reportOperatorIssue]
        for i in range(N)
        for j in range(N)
        if i != j
    ]

    # null_isc accumulates shape (n_permutations, n_windows, n_comp)
    null_isc_list: list[np.ndarray] = []

    for _ in tqdm(range(n_permutations), desc=f"Surrogate permutations ({method})"):
        if method == "circular":
            # Independent circular shift per subject (preserves autocorrelation
            # and the exact waveform; destroys cross-subject alignment).
            shifts = rng.integers(min_shift, max_shift, size=N)
            Y_surrogate = np.stack(
                [np.roll(Y[i], int(shifts[i]), axis=-1) for i in range(N)]
            )  # (N, n_comp, T)
        else:  # method == "phase"
            # Independent multivariate phase randomization per subject
            # (Prichard & Theiler 1994; preserves each subject's amplitude
            # spectrum and cross-component covariance, destroys alignment).
            Y_surrogate = np.stack(
                [_phase_randomize_multivariate(Y[i], rng) for i in range(N)]
            )  # (N, n_comp, T)

        Y_flat = Y_surrogate.reshape(N * n_comp, T)

        perm_isc: list[np.ndarray] = []
        for t in range(0, T - window_samples + 1, step_samples):
            Yw = Y_flat[:, t : t + window_samples]
            Rij = np.cov(Yw)  # (N*n_comp, N*n_comp)

            Rw = np.mean([Rij[s:e, s:e] for s, e in rw_slices], axis=0)
            Rb = np.mean([Rij[si:ei, sj:ej] for si, ei, sj, ej in rb_pairs_idx], axis=0)

            perm_isc.append(np.diag(Rb) / np.diag(Rw))

        if perm_isc:
            null_isc_list.append(np.array(perm_isc))  # (n_windows, n_comp)

    # null_isc: (n_permutations, n_windows, n_comp)
    null_isc = np.stack(null_isc_list, axis=0)

    # One-tailed per-window threshold: (1 - p_threshold) quantile across permutations
    # Result shape: (n_windows, n_comp) → transpose to (n_comp, n_windows)
    return np.quantile(null_isc, 1.0 - p_threshold, axis=0).T

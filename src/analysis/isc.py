# ICA https://github.com/ML-D00M/ISC-Inter-Subject-Correlations/blob/main/Python/ISC.py
from scipy.linalg import eigh
import numpy as np
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
    Rw: np.ndarray|None = None
    Rb: np.ndarray|None = None
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
        raise ValueError("Rw or Rb was not computed. Check if data is provided correctly.")

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


def apply_cca(X: np.ndarray, W: np.ndarray, fs: int, window_sec: float = 5.0, step_sec: float = 1.0, Cz_index: int | None = None):
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
            rw_blocks[k] = 1 / (N - 1) * (Rij[subj_k, subj_k, :, :] + Rij[subj_l, subj_l, :, :])
            rb_blocks[k] = 1 / (N - 1) * (Rij[subj_k, subj_l, :, :] + Rij[subj_l, subj_k, :, :])
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
    rb_pairs_t = [(i, j) for i in range(0, D * N, D) for j in range(0, D * N, D) if i != j]
    n_rw = len(rw_idx)
    n_rb = len(rb_pairs_t)

    for t in tqdm(range(0, T - window_samples + 1, step_samples), desc="ISC per window"):
        t_end = t + window_samples
        Xt = X[:, t : t_end]
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

#!/usr/bin/env python3
"""
Compute the average luminance difference (ALD) exactly as defined by
Poulsen et al. (2017), 'EEG in the classroom', Methods -> 'Average luminance
difference (ALD)'.

Their definition, verbatim in structure:

  1. "Video clips were converted to grey scale (0-255) by averaging over the
     three colour channels."                  -> equal-weight RGB mean, NOT Rec.601 luma
  2. "We then calculated the squared difference in pixel intensity from one
     frame to the next and took the average across pixels."
                                              -> mean over pixels of (I_t - I_{t-1})^2
  3. "These signals were non-linearly re-sampled at 1 Hz by selecting the
     maximum ALD for each 1 s interval to emphasise the large differences
     during changes in camera position."      -> per-second MAX, not mean
  4. "These values were then smoothed in time by convolving with a Gaussian
     kernel with a 'variance' parameter of 2.5 s^2."
                                              -> sigma = sqrt(2.5) s ~= 1.581 s

Every one of these four steps differs from ffmpeg's signalstats YDIF, which is
the mean *absolute* difference of the Rec.601 luma plane, at frame rate, with no
resampling and no smoothing.

Usage
-----
    python compute_ald.py VIDEO -o out/ald_byd.csv
    python compute_ald.py VIDEO -o out/ald_byd.csv --sigma2 2.5 --width 320

Output CSV columns
------------------
    time_s     centre time of each 1 s bin
    ald_raw    per-second maximum of the mean squared frame difference
    ald        ald_raw after Gaussian smoothing  <- use this one

Requires ffmpeg/ffprobe on PATH, plus numpy and scipy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def probe_video(path: Path) -> tuple[int, int, float]:
    """Return (width, height, fps) for the first video stream."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    num, _, den = st["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    return int(st["width"]), int(st["height"]), fps


def frame_squared_differences(
    path: Path, width: int, height: int, scale_width: int | None
) -> np.ndarray:
    """
    Stream the video as raw RGB and return, for each consecutive frame pair,
    the mean over pixels of the squared intensity difference.

    Greyscale is the equal-weight mean of R, G and B, matching Poulsen et al.
    rather than the Rec.601 weighting that ffmpeg's Y plane would give.
    """
    vf = []
    if scale_width is not None and scale_width < width:
        new_h = int(round(height * scale_width / width)) // 2 * 2
        vf = ["-vf", f"scale={scale_width}:{new_h}"]
        width, height = scale_width, new_h

    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), *vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None

    frame_bytes = width * height * 3
    prev: np.ndarray | None = None
    diffs: list[float] = []
    n = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            # step 1: equal-weight RGB mean, kept in float to avoid rounding
            grey = rgb.mean(axis=2, dtype=np.float64)
            if prev is not None:
                # step 2: mean over pixels of the squared difference
                d = grey - prev
                diffs.append(float(np.mean(d * d)))
            prev = grey
            n += 1
            if n % 2000 == 0:
                print(f"    ...{n} frames", file=sys.stderr)
    finally:
        proc.stdout.close()
        proc.wait()

    if not diffs:
        raise RuntimeError("No frames decoded. Is the path correct and ffmpeg working?")
    print(f"    {n} frames decoded, {len(diffs)} frame pairs", file=sys.stderr)
    return np.asarray(diffs, dtype=np.float64)


def per_second_max(diffs: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Step 3: non-linear resampling to 1 Hz by taking the max within each 1 s bin."""
    # diff i sits between frames i and i+1, so its timestamp is (i + 1) / fps
    times = (np.arange(len(diffs)) + 1) / fps
    n_bins = int(np.floor(times[-1])) + 1
    bin_idx = np.floor(times).astype(int)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = diffs[bin_idx == b]
        if sel.size:
            out[b] = sel.max()
    # a bin can only be empty for pathological frame rates; carry the neighbour
    if np.isnan(out).any():
        idx = np.arange(n_bins)
        good = ~np.isnan(out)
        out = np.interp(idx, idx[good], out[good])
    centres = np.arange(n_bins) + 0.5
    return centres, out


def smooth(ald: np.ndarray, sigma2_s: float, fs_hz: float = 1.0) -> np.ndarray:
    """Step 4: Gaussian smoothing with the given variance parameter, in seconds^2."""
    sigma_samples = np.sqrt(sigma2_s) * fs_hz
    return gaussian_filter1d(ald, sigma=sigma_samples, mode="nearest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--sigma2", type=float, default=2.5,
                    help="Gaussian variance parameter in s^2 (default 2.5, per the paper)")
    ap.add_argument("--width", type=int, default=None,
                    help="Downscale to this width first. ALD is a pixel-mean, so it is "
                         "near-invariant to scale and this is much faster. Try 320.")
    args = ap.parse_args()

    w, h, fps = probe_video(args.video)
    print(f"  {args.video.name}: {w}x{h} @ {fps:.4f} fps", file=sys.stderr)

    diffs = frame_squared_differences(args.video, w, h, args.width)
    t, ald_raw = per_second_max(diffs, fps)
    ald = smooth(ald_raw, args.sigma2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("time_s,ald_raw,ald\n")
        for ti, ri, si in zip(t, ald_raw, ald):
            fh.write(f"{ti:.3f},{ri:.6f},{si:.6f}\n")
    print(f"  wrote {len(t)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

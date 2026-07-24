import os
from argparse import ArgumentParser
from pathlib import Path

import mne  # type: ignore
import numpy as np
from tqdm import tqdm

from data import eeg


def get_trigger_sample(raw: mne.io.Raw, trigger_id: int | None = None) -> int:
    """Return the sample index of the first trigger event in a recording.

    Tries annotations first, then falls back to stim channels.
    """
    events, _ = mne.events_from_annotations(raw, verbose="error")

    if len(events) == 0:
        stim_chs = mne.pick_types(raw.info, stim=True)
        if len(stim_chs) > 0:
            events = mne.find_events(raw, verbose="error")

    if len(events) == 0:
        raise ValueError("No trigger events found in recording")

    if trigger_id is not None:
        mask = events[:, 2] == trigger_id
        if not mask.any():
            available = sorted(set(events[:, 2]))
            raise ValueError(
                f"Trigger ID {trigger_id} not found. Available IDs: {available}"
            )
        return int(events[mask][0, 0])

    return int(events[0, 0])


def align_recordings(
    subject_data: dict[int, mne.io.Raw],
    trigger_id: int | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Align recordings by trigger onset, cropping start and end as needed.

    For each recording, finds the first trigger event and crops the start so
    all recordings share a common trigger-aligned origin. Recordings that had
    a later trigger onset (more pre-trigger samples) lose samples from the
    start; all recordings are then trimmed to the shortest aligned length.

    Modifies raw objects in subject_data in-place.

    Returns
    -------
    start_crops : dict[int, int]
        Number of samples removed from the start for each subject.
    end_crops : dict[int, int]
        Number of samples removed from the end for each subject.
    """
    trigger_samples = {
        sid: get_trigger_sample(raw, trigger_id) for sid, raw in subject_data.items()
    }

    min_offset = min(trigger_samples.values())
    sfreq = next(iter(subject_data.values())).info["sfreq"]

    start_crops: dict[int, int] = {}
    for sid, raw in subject_data.items():
        crop_n = trigger_samples[sid] - min_offset
        start_crops[sid] = crop_n
        if crop_n > 0:
            raw.crop(tmin=crop_n / sfreq, verbose="error")

    min_length = min(raw.n_times for raw in subject_data.values())
    end_crops: dict[int, int] = {}
    for sid, raw in subject_data.items():
        end_n = raw.n_times - min_length
        end_crops[sid] = end_n
        if end_n > 0:
            raw.crop(tmax=raw.times[min_length - 1], verbose="error")

    return start_crops, end_crops


def show_trigger_info(data_dir: str) -> None:
    """Load all raw recordings and print a summary of trigger events."""
    all_eeg_data = eeg.load_all_eeg(Path(data_dir))
    mne.utils.set_log_level("WARNING")

    for stimulus, subject_data in all_eeg_data.items():
        print(f"\n{'=' * 60}")
        print(f"Stimulus: {stimulus}")
        print(f"{'=' * 60}")
        for subject_id, raw in subject_data.items():
            sfreq = raw.info["sfreq"]
            events, event_id = mne.events_from_annotations(raw, verbose="error")

            if len(events) == 0:
                stim_chs = mne.pick_types(raw.info, stim=True)
                if len(stim_chs) > 0:
                    events = mne.find_events(raw, verbose="error")
                    event_id = {str(v): v for v in np.unique(events[:, 2])}

            print(
                f"\n  Subject {subject_id}  ({raw.n_times} samples, {raw.n_times / sfreq:.1f} s)"
            )
            if len(events) == 0:
                print("    No trigger events found.")
                continue

            id_to_desc = {v: k for k, v in event_id.items()}
            for uid in np.unique(events[:, 2]):
                mask = events[:, 2] == uid
                count = mask.sum()
                first_sample = events[mask][0, 0]
                first_time = first_sample / sfreq
                desc = id_to_desc.get(uid, str(uid))
                print(
                    f"    ID {uid:>4}  '{desc}'  count={count:>4}  "
                    f"first at sample {first_sample} ({first_time:.3f} s)"
                )


def preprocess_eeg_data(
    data_dir: str,
    out_dir: str,
    lfreq: float = 0.5,
    hfreq: float = 45.0,
    regress_eog: bool = True,
    remove_outliers: bool = True,
    force: bool = False,
    align_trigger_id: int | None = None,
) -> tuple[
    dict[str, list[tuple[int, int, int]]],
    dict[str, list[tuple[int, int]]],
    dict[str, list[tuple[int, list[str]]]],
]:
    """Preprocess EEG data

    Parameters:
    ----------
    data_dir : str
        Path to the directory containing raw EEG data.
    out_dir : str
        Path to the directory where preprocessed EEG data will be saved.
    lfreq : float, optional
        Low cutoff frequency for bandpass filter (default: 0.5 Hz).
    hfreq : float, optional
        High cutoff frequency for bandpass filter (default: 40.0 Hz).
    regress_eog : bool, optional
        If True, perform EOG regression to remove artifacts (default: True).
    remove_outliers : bool, optional
        If True, remove outlier epochs based on amplitude thresholds > 4 IQD (default: True).
    force : bool, optional
        If True, overwrite existing files in the output directory (default: False).
    align_trigger_id : int or None, optional
        Trigger event ID to use as the alignment reference. If None, the first
        event found in each recording is used (default: None).

    Returns:
    -------
    tuple[
        dict[str, list[tuple[int, int, int]]],
        dict[str, list[tuple[int, int]]],
        dict[str, list[tuple[int, list[str]]]],
    ]
        A tuple containing three dictionaries:
        1. alignment_crops: Start and end samples removed per subject and stimulus.
        2. zeroed_outlier_sample_count: Number of samples zeroed out due to outliers.
        3. bad_channels: List of bad channels identified for each subject and stimulus.
    """

    eeg.validate_paths(data_dir, out_dir)
    alignment_crops: dict[str, list[tuple[int, int, int]]] = {
        "StoryCorps_Q&A": [],
        "BangBangYouAreDead": [],
    }
    zeroed_outlier_sample_count: dict[str, list[tuple[int, int]]] = {
        "StoryCorps_Q&A": [],
        "BangBangYouAreDead": [],
    }
    bad_channels: dict[str, list[tuple[int, list[str]]]] = {
        "StoryCorps_Q&A": [],
        "BangBangYouAreDead": [],
    }

    all_eeg_data = eeg.load_all_eeg(Path(data_dir))
    mne.utils.set_log_level("WARNING")

    for stimulus, subject_data in all_eeg_data.items():
        print(
            f"\nStimulus '{stimulus}' - aligning {len(subject_data)} recordings by trigger..."
        )
        start_crops, end_crops = align_recordings(
            subject_data, trigger_id=align_trigger_id
        )
        for sid in subject_data:
            alignment_crops[stimulus].append((sid, start_crops[sid], end_crops[sid]))
            if start_crops[sid] or end_crops[sid]:
                print(
                    f"  Subject {sid}: -{start_crops[sid]} samples from start, "
                    f"-{end_crops[sid]} samples from end"
                )

        for subject_id, raw in tqdm(
            subject_data.items(), desc=f"Processing {stimulus}"
        ):
            raw.load_data(verbose="error")

            raw.filter(
                l_freq=lfreq, h_freq=hfreq, method="fir", picks="all", verbose="error"
            )

            if regress_eog:
                raw.pick(picks=["eeg", "eog"], verbose="error")
                raw.set_eeg_reference("average", projection=False, verbose="error")
                weights: mne.preprocessing.EOGRegression = (
                    mne.preprocessing.EOGRegression().fit(raw)
                )
                raw = weights.apply(raw)

            raw.pick(picks="eeg", verbose="error")

            if remove_outliers:
                eeg_channel_indices = mne.pick_types(
                    raw.info, eeg=True, eog=False, meg=False, exclude="bads"
                )
                percentiles = np.percentile(
                    abs(raw.get_data(verbose="error")),  # pyright: ignore[reportArgumentType]
                    [25, 75],
                    axis=1,
                )
                bad_samples_count = 0
                for ch_idx in eeg_channel_indices:
                    iqd = percentiles[1, ch_idx] - percentiles[0, ch_idx]
                    threshold = percentiles[1, ch_idx] + 4 * iqd
                    data = raw.get_data(picks=[ch_idx], verbose="error")[0]
                    bad_samples = np.where(abs(data) > threshold)[0]

                    for sample in bad_samples:
                        start = max(0, sample - int(0.04 * raw.info["sfreq"]))
                        end = min(len(data), sample + int(0.04 * raw.info["sfreq"]))
                        data[start:end] = 0
                        bad_samples_count += end - start
                    raw._data[ch_idx, :] = data  # pyright: ignore[reportOptionalSubscript]
                zeroed_outlier_sample_count[stimulus].append(
                    (subject_id, bad_samples_count)  # pyright: ignore[reportArgumentType]
                )

            log_power = np.log(np.std(raw.get_data(verbose="error"), axis=1))
            power_threshold = np.percentile(
                np.log(np.std(raw.get_data(verbose="error"), axis=1)), [25, 50, 75]
            )
            bad_channel_indices = np.where(
                log_power
                > power_threshold[2] + 4 * (power_threshold[2] - power_threshold[0])
            )[0]
            bad_channel_names: list[str] = [
                raw.ch_names[idx] for idx in bad_channel_indices
            ]
            raw.info["bads"].extend(bad_channel_names)
            bad_channels[stimulus].append((subject_id, bad_channel_names))

            file_code = eeg.STIMULI_FILE_CODES[stimulus]
            stim_out_dir = (
                Path(out_dir) / "02_preprocessed_eeg_data" / eeg.STIMULUS_SHORT_KEYS[stimulus]
            )
            stim_out_dir.mkdir(parents=True, exist_ok=True)
            out_file_path = stim_out_dir / eeg.PREPROCESSED_FILE_FORMAT.format(
                file_code=file_code, subject_id=subject_id, stimulus=stimulus
            )
            if out_file_path.exists() and not force:
                raise FileExistsError(
                    f"Output file {out_file_path} already exists. Use --force to overwrite."
                )
            raw.save(
                out_file_path, picks=["eeg", "eog"], overwrite=True, verbose="error"
            )

    return alignment_crops, zeroed_outlier_sample_count, bad_channels


def print_results(
    alignment_crops: dict[str, list[tuple[int, int, int]]],
    zeroed_outlier_sample_count: dict[str, list[tuple[int, int]]],
    bad_channels: dict[str, list[tuple[int, list[str]]]],
):
    for stimulus in alignment_crops.keys():
        print(f"\nStimulus: {stimulus}")
        print("Alignment crops (start samples, end samples):")
        for subject_id, start_crop, end_crop in alignment_crops[stimulus]:
            print(f"  Subject {subject_id}: -{start_crop} start, -{end_crop} end")
        print("Zeroed outlier samples:")
        for subject_id, count in zeroed_outlier_sample_count[stimulus]:
            print(f"  Subject {subject_id}: {count} samples zeroed out")
        print("Bad channels:")
        for subject_id, channels in bad_channels[stimulus]:
            print(
                f"  Subject {subject_id}: {', '.join(channels) if channels else 'None'}"
            )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Preprocess EEG data by removing ear-EEG channels."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        help="Path to the directory containing raw EEG data.",
        default=os.getenv("EEG_DATA_PATH", None),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=False,
        help="Path to the directory where preprocessed EEG data will be saved.",
        default=os.getenv("EEG_WORK_DIR", "./out"),
    )
    parser.add_argument(
        "--lfreq",
        type=float,
        default=0.5,
        help="Low cutoff frequency for bandpass filter (default: 0.5 Hz)",
    )
    parser.add_argument(
        "--hfreq",
        type=float,
        default=40.0,
        help="High cutoff frequency for bandpass filter (default: 40.0 Hz)",
    )
    parser.add_argument(
        "--no-regress-eog",
        action="store_false",
        help="If set, do not perform EOG regression to remove artifacts.",
        default=True,
    )
    parser.add_argument(
        "--no-remove-outliers",
        action="store_false",
        help="If set, do not remove outlier epochs based on amplitude thresholds > 4 IQD.",
        default=True,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, overwrite existing files.",
        default=False,
    )
    parser.add_argument(
        "--align-trigger-id",
        type=int,
        default=None,
        help=(
            "Trigger event ID to use as the alignment reference across subjects. "
            "If not set, the first event in each recording is used."
        ),
    )

    parser.add_argument(
        "--info",
        action="store_true",
        default=False,
        help="Print trigger event info for all recordings and exit (no preprocessing).",
    )

    args = parser.parse_args()

    if args.info:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        show_trigger_info(args.data_dir)
        raise SystemExit(0)

    eeg.validate_paths(args.data_dir, args.out_dir)
    alignment_crops, zeroed_outlier_sample_count, bad_channels = preprocess_eeg_data(
        args.data_dir,
        args.out_dir,
        lfreq=args.lfreq,
        hfreq=args.hfreq,
        regress_eog=args.no_regress_eog,
        remove_outliers=args.no_remove_outliers,
        force=args.force,
        align_trigger_id=args.align_trigger_id,
    )
    print_results(alignment_crops, zeroed_outlier_sample_count, bad_channels)

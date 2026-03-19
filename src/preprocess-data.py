from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
from data import eeg
import mne  # type: ignore
from tqdm import tqdm


def preprocess_eeg_data(
    data_dir: str,
    out_dir: str,
    lfreq: float = 0.5,
    hfreq: float = 40.0,
    regress_eog: bool = True,
    remove_outliers: bool = True,
    force: bool = False,
) -> tuple[
    dict[str, list[tuple[int, int]]],
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

    Returns:
    -------
    tuple[
        dict[str, list[tuple[int, int]]],
        dict[str, list[tuple[int, int]]],
        dict[str, list[tuple[int, list[str]]]],
    ]
        A tuple containing three dictionaries:
        1. cropped_samples: Number of samples cropped for each subject and stimulus.
        2. zeroed_outlier_sample_count: Number of samples zeroed out due to outliers for each subject and stimulus.
        3. bad_channels: List of bad channels identified for each subject and stimulus.
    """

    # Validate input and output paths
    eeg.validate_paths(data_dir, out_dir)
    cropped_samples: dict[str, list[tuple[int, int]]] = {
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
    # Load all EEG data
    all_eeg_data = eeg.load_all_eeg(Path(data_dir))
    mne.utils.set_log_level("WARNING")
    for stimulus, subject_data in all_eeg_data.items():
        min_length = min(
            raw.get_data(verbose="error").shape[1] for raw in subject_data.values()
        )
        print(
            f"Stimulus '{stimulus}' - minimum number of samples across subjects: {min_length}, truncating all to this length."
        )
        for subject_id, raw in tqdm(
            subject_data.items(), desc=f"Processing {stimulus}"
        ):
            # Preprocess the raw data (e.g., filter, remove EOG artifacts)
            raw.load_data(verbose="error")
            # warn if cropping
            if raw.get_data(verbose="error").shape[1] > min_length:
                dropped_samples = raw.get_data(verbose="error").shape[1] - min_length
                cropped_samples[stimulus].append((subject_id, dropped_samples))
                raw.crop(
                    tmax=min_length / raw.info["sfreq"], verbose="error"
                )  # Ensure all have the same number of samples

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

            raw.pick(picks="eeg", verbose="error")  # Keep only EEG channels for output

            if remove_outliers:
                eeg_channel_indices = mne.pick_types(
                    raw.info, eeg=True, eog=False, meg=False, exclude="bads"
                )
                percentiles = np.percentile(
                    abs(raw.get_data(verbose="error")), [25, 75], axis=1
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
                    raw._data[ch_idx, :] = data
                zeroed_outlier_sample_count[stimulus].append(
                    (subject_id, bad_samples_count)
                )

            # mark bads
            # find bad channels base on power outliers
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

            # Save the preprocessed data
            file_code = eeg.STIMULI_FILE_CODES[stimulus]
            out_file_path = Path(out_dir) / eeg.PREPROCESSED_FILE_FORMAT.format(
                file_code=file_code, subject_id=subject_id, stimulus=stimulus
            )
            if out_file_path.exists() and not force:
                raise FileExistsError(
                    f"Output file {out_file_path} already exists. Use --force to overwrite."
                )
            raw.save(
                out_file_path, picks=["eeg", "eog"], overwrite=True, verbose="error"
            )
    return cropped_samples, zeroed_outlier_sample_count, bad_channels


def print_results(
    cropped_samples: dict[str, list[tuple[int, int]]],
    zeroed_outlier_sample_count: dict[str, list[tuple[int, int]]],
    bad_channels: dict[str, list[tuple[int, list[str]]]],
):
    for stimulus in cropped_samples.keys():
        print(f"\nStimulus: {stimulus}")
        print("Cropped samples:")
        for subject_id, count in cropped_samples[stimulus]:
            print(f"  Subject {subject_id}: {count} samples cropped")
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
    # filter options
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

    args = parser.parse_args()

    eeg.validate_paths(args.data_dir, args.out_dir)
    cropped_samples, zeroed_outlier_sample_count, bad_channels = preprocess_eeg_data(
        args.data_dir,
        args.out_dir,
        lfreq=args.lfreq,
        hfreq=args.hfreq,
        regress_eog=args.no_regress_eog,
        remove_outliers=args.no_remove_outliers,
        force=args.force,
    )
    print_results(cropped_samples, zeroed_outlier_sample_count, bad_channels)

import os
from typing import Literal
from tqdm import tqdm

import mne  # type: ignore
from pathlib import Path

# type alias for stimulus names
StimulusName = Literal["StoryCorps_Q&A", "BangBangYouAreDead"]

# Ear-EEG - exclude from 10-20 montage
EAR_EEG_CHANNELS = [
    "ELA1",
    "ELA2",
    "ELB1",
    "ELB2",
    "ELC1",
    "ELC2",
    "ELK",
    "ELT",
    "ERA1",
    "ERA2",
    "ERB1",
    "ERB2",
    "ERC1",
    "ERC2",
    "ERK",
    "ERT",
]

# EOG - set channel type
EOG_CHANNELS = ["HL_EOG", "HR_EOG", "VA_EOG", "VB_EOG"]

# scalp_channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8', 'AFz']

SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
STIMULI: list[StimulusName] = [
    "StoryCorps_Q&A",
    "BangBangYouAreDead",
]

STIMULI_FILE_CODES: dict[StimulusName, int] = {
    "StoryCorps_Q&A": 71,
    "BangBangYouAreDead": 72,
}

EEG_FILE_FORMAT = "{file_code}_HS1{subject_id:02d}_{stimulus}.set"
PREPROCESSED_FILE_FORMAT = (
    "{file_code}_HS1{subject_id:02d}_{stimulus}_preprocessed_eeg.fif"
)


def validate_paths(data_dir: str, out_dir: str):
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"EEG data directory {data_dir} does not exist or is not a directory."
        )
    # Try and create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)


def load_eeg(
    data_path: Path,
    subject_id: int,
    stimulus: StimulusName,
) -> mne.io.Raw:
    """Load EEG data for a given subject and stimulus.

    Args:
        data_path (Path): The path to the directory containing EEG data.
        subject_id (int): The ID of the subject (e.g., 1, 2, ...).
        stimulus (StimulusName): The stimulus name.

    Returns:
        mne.io.Raw: The loaded EEG data.
    """

    file_code = STIMULI_FILE_CODES[stimulus]

    file_path = data_path / EEG_FILE_FORMAT.format(
        file_code=file_code, subject_id=subject_id, stimulus=stimulus
    )

    raw = mne.io.read_raw_eeglab(
        file_path, preload=True, eog=EOG_CHANNELS, verbose="error"
    )

    if all(ch in raw.ch_names for ch in EAR_EEG_CHANNELS):
        raw.set_channel_types(
            {ch: "eeg" for ch in EAR_EEG_CHANNELS}
        )  # or 'eeg' if you want to keep them

    # Apply standard montage to just the scalp channels
    montage: mne.channels.DigMontage = mne.channels.make_standard_montage(
        "standard_1020"
    )
    raw.set_montage(
        montage, on_missing="ignore", verbose="error"
    )  # ignores ear-EEG and EOG

    return raw


def load_preprocessed_eeg(
    data_path: Path,
    subject_id: int,
    stimulus: StimulusName,
) -> mne.io.Raw:
    """Load preprocessed EEG data for a given subject and stimulus."""
    file_code = STIMULI_FILE_CODES[stimulus]
    file_path = data_path / PREPROCESSED_FILE_FORMAT.format(
        file_code=file_code, subject_id=subject_id, stimulus=stimulus
    )
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose="error")
    return raw


def load_all_eeg(
    data_path: Path,
    preprocessed: bool = False,
) -> dict[StimulusName, dict[int, mne.io.Raw]]:
    """Load all EEG data for all subjects for both stimuli."""
    all_data: dict[StimulusName, dict[int, mne.io.Raw]] = {}

    for stimulus in STIMULI:
        all_data[stimulus] = {}
        for subject_id in tqdm(SUBJECT_IDS, desc=f"Loading {stimulus}"):
            if preprocessed:
                all_data[stimulus][subject_id] = load_preprocessed_eeg(
                    data_path, subject_id, stimulus
                )
            else:
                all_data[stimulus][subject_id] = load_eeg(
                    data_path, subject_id, stimulus
                )

    return all_data

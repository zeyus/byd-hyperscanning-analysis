from typing import Literal

import mne  # type: ignore
from pathlib import Path

# Ear-EEG - exclude from 10-20 montage
EAR_EEG_CHANNELS = ['ELA1', 'ELA2', 'ELB1', 'ELB2', 'ELC1', 'ELC2', 'ELK', 'ELT',
                    'ERA1', 'ERA2', 'ERB1', 'ERB2', 'ERC1', 'ERC2', 'ERK', 'ERT']

# EOG - set channel type
EOG_CHANNELS = ['HL_EOG', 'HR_EOG', 'VA_EOG', 'VB_EOG']

# scalp_channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8', 'AFz']

SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
STIMULI: list[Literal["StoryCorps_Q&A", "BangBangYouAreDead"]] = ["StoryCorps_Q&A", "BangBangYouAreDead"]

STIMULI_FILE_CODES: dict[Literal["StoryCorps_Q&A", "BangBangYouAreDead"], int] = {
    "StoryCorps_Q&A": 71,
    "BangBangYouAreDead": 72
}

EEG_FILE_FORMAT = "{file_code}_HS1{subject_id:02d}_{stimulus}.set"


def load_eeg(data_path: Path, subject_id: int, stimulus: Literal["StoryCorps_Q&A", "BangBangYouAreDead"]) -> mne.io.Raw:
    """Load EEG data for a given subject and stimulus.

    Args:
        data_path (Path): The path to the directory containing EEG data.
        subject_id (int): The ID of the subject (e.g., 1, 2, ...).
        stimulus (str): The stimulus name, either 'StoryCorps_Q&A' or 'BangBangYouAreDead'.

    Returns:
        mne.io.Raw: The loaded EEG data.
    """
    
    file_code = STIMULI_FILE_CODES[stimulus]

    file_path = data_path / EEG_FILE_FORMAT.format(file_code=file_code, subject_id=subject_id, stimulus=stimulus)
    
    raw = mne.io.read_raw_eeglab(file_path, preload=True, eog=EOG_CHANNELS, verbose='error')

    if all(ch in raw.ch_names for ch in EAR_EEG_CHANNELS):
        raw.set_channel_types({ch: 'eeg' for ch in EAR_EEG_CHANNELS})  # or 'eeg' if you want to keep them

    # Apply standard montage to just the scalp channels
    montage: mne.channels.DigMontage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore', verbose='error')  # ignores ear-EEG and EOG

    
    return raw
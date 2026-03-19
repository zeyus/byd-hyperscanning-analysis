from typing import Literal
import os
import argparse
from data import eeg
from pathlib import Path
from tqdm import tqdm


def main(data_dir: str, out_dir: str):
    for stimulus in tqdm(eeg.STIMULI, desc="Processing stimuli"):
        for subject_id in tqdm(eeg.SUBJECT_IDS, desc=f"Processing subjects for {stimulus}", leave=False):
            raw = eeg.load_eeg(Path(data_dir), subject_id, stimulus)
            # Drop ear-EEG channels
            raw.drop_channels(eeg.EAR_EEG_CHANNELS)
            # Save the modified raw object to a new file
            file_code = eeg.STIMULI_FILE_CODES[stimulus]
            out_path = Path(out_dir) / eeg.EEG_FILE_FORMAT.format(file_code=file_code, subject_id=subject_id, stimulus=stimulus)
            if out_path.exists():
                raise FileExistsError(f"Output file {out_path} already exists. Please remove it or change the output directory before running the script.")
            raw.export(out_path, fmt='eeglab', verbose=False)

    print(f"Finished processing. Modified EEG files saved to {out_dir} without EarEEG channels.")


def print_summary(file_info: dict[Literal["StoryCorps_Q&A", "BangBangYouAreDead"], list[dict]]):
    for stimulus, files in file_info.items():
        print(f"\nStimulus: {stimulus}")
        # print channel names and data shape for the first file as a sample
        if files:
            sample_info = files[0]
            print(f"Sample file: {sample_info['file_path']}")
            print(f"Number of channels: {sample_info['n_channels']}")
            print(f"Channel names: {sample_info['channel_names']}")
            print(f"Data shape (channels x samples): {sample_info['data_shape']}")
        else:
            print("No files found for this stimulus.")



def validate_data_dir(out_dir: str) -> bool:
    valid = True
    file_info: dict[Literal["StoryCorps_Q&A", "BangBangYouAreDead"], list[dict]] = {
        "StoryCorps_Q&A": [],
        "BangBangYouAreDead": []
    }
    for stimulus in eeg.STIMULI:
        for subject_id in eeg.SUBJECT_IDS:
            file_code = eeg.STIMULI_FILE_CODES[stimulus]
            file_path = Path(out_dir) / eeg.EEG_FILE_FORMAT.format(file_code=file_code, subject_id=subject_id, stimulus=stimulus)
            if not file_path.exists():
                print(f"Error: Expected file {file_path} does not exist.")
                valid = False
                continue
            
            raw = eeg.load_eeg(Path(out_dir), subject_id, stimulus)
            for ch in eeg.EAR_EEG_CHANNELS:
                if ch in raw.ch_names:
                    print(f"Error: EarEEG channel {ch} found in file {file_path}.")
                    valid = False
            file_info[stimulus].append({
                "subject_id": subject_id,
                "file_path": file_path,
                "n_channels": len(raw.ch_names),
                "channel_names": raw.ch_names,
                "data_shape": raw.get_data().shape,
            })
    
    # ensure every file has the same number of channels and samples
    n_channels_set = set(info["n_channels"] for files in file_info.values() for info in files)
    if len(n_channels_set) > 1:
        print(f"Warning: Not all files have the same number of channels: {n_channels_set}")
        valid = False
    for stimulus, files in file_info.items():
        n_samples_set = set(info["data_shape"][1] for info in files)
        if len(n_samples_set) > 1:
            print(f"Warning: Not all files for stimulus {stimulus} have the same number of samples: {n_samples_set}")
            valid = False
    
    print_summary(file_info)
    return valid



def validate_paths(data_dir: str, out_dir: str):
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        raise FileNotFoundError(f"EEG data directory {data_dir} does not exist or is not a directory.")
    # Try and create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove EA EEG channels")
    parser.add_argument("--out", type=str, required=False, help="Directory to save EEG data without EarEEG channels", default=os.getenv("EEG_DATA_PATH", "./data/eeg"))
    parser.add_argument("--data-dir", type=str, required=False, help="Directory containing original EEG data with EarEEG channels", default=os.getenv("EEG_DATA_PATH_ORIGINAL", None))
    parser.add_argument("--validate-only", action="store_true", help="Only validate output directory and data files without processing")
    args = parser.parse_args()
    try:
        validate_paths(args.data_dir, args.out)
    except Exception as e:
        print(f"Error validating paths: {e}")
        exit(1)
    if not args.validate_only:
        main(args.data_dir, args.out)
    
    # After processing, validate the output files
    valid = validate_data_dir(args.out)
    if not valid:
        print("Validation failed, check output.")
        exit(1)
    else:
        print("All files validated successfully. EarEEG channels removed and data is consistent.")
        



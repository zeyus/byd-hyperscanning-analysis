# Installation

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/zeyus/byd-hyperscanning-analysis.git
cd byd-hyperscanning-analysis
```

## UV

- [Install UV](https://docs.astral.sh/uv/#installation)
  - Macos/Linux:  
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
  - Windows:
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
  - Alternatively, UV can be installed via pip, or possibly from your package manager (e.g. `brew`, `pacman`, `apt`, `WinGet`, `choco`).

## Optional: Direnv

If you want to use [direnv](https://direnv.net/) it can automatically load environment variables for you, which can be nice for setting the path to the data etc. It's optional, but it means you don't have to change anything in the script.

- [Install direnv](https://direnv.net/docs/installation.html), the easiest way is probable via your package manager. 
- Copy the default configuration file, and make changes as appropriate:
  ```bash
  cp .envrc.dist .envrc
  # Open .envrc in your editor and update the values as needed
  ```
- Once you are happy with the contents of the .envrc file, allow it to be loaded for this project:
  ```bash
  direnv allow
  ```

## Create a virtual environment and install dependencies

```bash
uv sync
```

# Usage

## Scripts

There is one script in the `scripts` directory, `frame_analysis.fish`, which is a fish shell script that extracts audiovisual features from the stimuli videos, it's not necessary to run this script because the features are already extracted, but if you want to run it it requires `ffmpeg`/`ffprobe`.

## Source data validation

This step just makes sure that all the data files exist in the specified `EEG_DATA_DIR` and they have the expected format.


```bash
uv run python src/remove-eareeg-channels.py --validate-only
```

## Combine / Composite stimli features

This combines the audio and visual features into a single CSV file for each stimulus, by default this saves to the `./out` directory, but you can change this by setting the `EEG_WORK_DIR` environment variable in your `.envrc` file.

```bash
uv run python src/combine-stimuli-features.py
```

## Preprocess EEG data

This preprocesses the EEG data by applying a bandpass filter, eog regression and removing outlier samples, and saves the preprocessed data to disk. By default this saves to the `./out` directory, but you can change this by setting the `EEG_WORK_DIR` environment variable in your `.envrc` file.

```bash
uv run python src/preprocess_eeg.py
```

Usage: Note, `data-dir` and `out-dir` will default to the values of `EEG_DATA_PATH` and `EEG_WORK_DIR` environment variables respectively, so you only need to specify them if you want to override those defaults.

```text
usage: preprocess-data.py [-h] [--data-dir DATA_DIR] [--out-dir OUT_DIR] [--lfreq LFREQ] [--hfreq HFREQ] [--no-regress-eog] [--no-remove-outliers] [--force]

Preprocess EEG data by removing ear-EEG channels.

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Path to the directory containing raw EEG data.
  --out-dir OUT_DIR     Path to the directory where preprocessed EEG data will be saved.
  --lfreq LFREQ         Low cutoff frequency for bandpass filter (default: 0.5 Hz)
  --hfreq HFREQ         High cutoff frequency for bandpass filter (default: 40.0 Hz)
  --no-regress-eog      If set, do not perform EOG regression to remove artifacts.
  --no-remove-outliers  If set, do not remove outlier epochs based on amplitude thresholds > 4 IQD.
  --force               If set, overwrite existing files.

```


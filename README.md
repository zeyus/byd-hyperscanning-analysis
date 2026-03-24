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

## Without direnv

You can create a `.envrc` file for the project, and add it to UV run commands like so:

- Copy the default configuration file, and make changes as appropriate:
  ```bash
  cp .envrc.dist .envrc
  # Open .envrc in your editor and update the values as needed
  ```

For `uv run` commands, you will need to specify the `--env-file` flag to load the environment variables from the `.envrc` file, for example:
```bash
uv run --env-file .envrc python src/preprocess-data.py -h
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
uv run python src/composite-stimuli-features.py
```

## Preprocess EEG data

This preprocesses the EEG data by applying a bandpass filter, eog regression and removing outlier samples, and saves the preprocessed data to disk. By default this saves to the `./out` directory, but you can change this by setting the `EEG_WORK_DIR` environment variable in your `.envrc` file.

```bash
uv run python src/preprocess-data.py
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

Once the preprocessing is done, you will have the data in e.g. `./out/{file_name}_preprocessed_eeg.fif`

## Interactive Inter-Subject Correlation Analysis notebook

There is an interactive Jupyter notebook at [`notebooks/isc_analysis.ipynb`](notebooks/isc_analysis.ipynb) that has sections for loading and exploring the preprocessed data for each stimulus. You can see a preview of the notebook in [`notebooks/isc_analysis.html`](notebooks/isc_analysis.html).

The notebook should use the ipykernel from the UV virtual environment, you can run it using the following command from the project directory (or open it in your editor/IDE of choice and select the uv venv kernel):

```bash
uv run jupyter lab notebooks/isc_analysis.ipynb
```

The notebook also can run the inter-subject correlation for a selected stimulus and (optionally) a subset of participants. The notebook displays the ISC results and the topographic projections of the components. It can also create a timeseries of the data projected through the weights of a selected component.

**Note:** The resulting time series units are not in microvolts, rather in relative units.

### Interactive Notebook Items

After you have run through the cells, there are several interactive elements in the notebook that can be used to specify parameters for plotting and ISC analysis.

#### Data Exploration

<img width="1265" height="1774" alt="image" src="https://github.com/user-attachments/assets/126831a2-8551-4f5d-8336-bbdaa2f50ef0" />

#### ISC Analysis

<img width="2216" height="1739" alt="image" src="https://github.com/user-attachments/assets/1dce5637-b24f-4316-b1e3-a7b5acb6cddd" />

#### Component Inspection

Topographical projection of components

<img width="1626" height="1469" alt="image" src="https://github.com/user-attachments/assets/311840e6-d4d1-45fe-8a88-a9031f94fdea" />

#### Apply component as a spatial filter

The resulting data is available in the `component_data` variable.

<img width="2108" height="876" alt="image" src="https://github.com/user-attachments/assets/7d1a90e1-7a80-436a-900f-e09422c50f27" />






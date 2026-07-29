# Analysis pipeline

This describes the full, scripted analysis pipeline, from raw EEG/stimuli
through ISC computation to final stats and figures — no Jupyter required.
`notebooks/isc_analysis.ipynb` remains available for interactive exploration
(component topomaps, spatial-filter playback, ad-hoc parameter sweeps), but
every step that produces a saved result now has a CLI equivalent.

All commands below assume `EEG_WORK_DIR` defaults to `./out` (see
`.envrc.dist`); pass `--out-dir` to any script to override it.

## Directory structure

```
out/
├── 01_extracted_stim_features/{byd,sc}/   composite frame-level feature CSV + time-series PNG
├── 02_preprocessed_eeg_data/{byd,sc}/     preprocessed .fif files (one per subject)
├── 03_ISC_results/{byd,sc}/{full,segment}/
│                                          isc_component{N}_bywindow.npy, chance_comp{N}.npy, meta.json
├── 04_feature_correlation/{byd,sc}/       isc_feature_correlation_test.py output (stats CSV + heatmap PNG)
├── 05_event_correlation/                  event_locked_isc_test.py output (stim in filename: _byd/_sc)
└── 06_figures/                            final/summary figures (R-generated comparison plots, misc/)

data/
├── stimuli_emotion_events.json            hand-authored event onset labels (git-tracked, not regenerable)
└── isc_comparison.csv                     digitized reference-study timeseries (Dmochowski/Poulsen)
```

`byd` = BangBangYouAreDead, `sc` = StoryCorps_Q&A (see `data.eeg.STIMULUS_SHORT_KEYS`).

## Ordered pipeline steps

1. **Validate source data** *(only if the raw EEG is available, otherwise used the preprocessed EEG)*
   ```bash
   uv run python src/remove-eareeg-channels.py --validate-only
   ```

2. **Extract composite stimulus features** → `01_extracted_stim_features/{byd,sc}/`
   ```bash
   uv run python src/composite-stimuli-features.py
   ```

3. **Preprocess EEG** (filter, EOG regression, outlier removal) → `02_preprocessed_eeg_data/{byd,sc}/`  *(only if the raw EEG is available, otherwise used the preprocessed EEG)*
   ```bash
   uv run python src/preprocess-data.py
   ```

4. **Compute ISC** for a stimulus (trains CCA, applies it for a per-window ISC
   timecourse, estimates a surrogate chance-level band, saves a diagnostic
   plot) → `03_ISC_results/{byd,sc}/{full,segment}/` + a quick-look PNG in
   `06_figures/`
   ```bash
   # Full-recording analysis
   uv run python src/analysis/compute_isc.py --stimulus BangBangYouAreDead --seed 2026

   # Segment analysis (explicit bounds, no implicit default)
   uv run python src/analysis/compute_isc.py --stimulus BangBangYouAreDead --seed 2026 --t0 296.667 --t1 669.667
   ```
   `--seed` is required so surrogate chance-level results are reproducible.
   Run once per stimulus × range you need (StoryCorps currently only has a
   "full" analysis).

5. **ISC × stimulus-feature correlation** → `04_feature_correlation/{byd,sc}/`
   ```bash
   uv run python src/analysis/isc_feature_correlation_test.py \
             --stimulus bangbangyouaredead --range-tag full \
             --feature "ebu_r128_M:out/01_extracted_stim_features/byd/BangBangYouAreDead_composite_frame_level_analysis.csv" \
             --feature "ald:out/01_extracted_stim_features/byd/ald.csv"
   ```

6. **ISC × emotion-event correlation** → `05_event_correlation/`
   ```bash
   uv run python src/analysis/event_locked_isc_test.py --stimulus bangbangyouaredead --event-group byd
   ```

Steps 5 and 6 auto-detect `t0_s`/`window_sec`/`step_sec` from the `meta.json`
sidecar `compute_isc.py` writes alongside each stim/range's results, so segment
analyses are aligned to absolute stimulus time without needing to repeat those
values on the command line.

## Comparison figures (R)

`src/plot-stim-isc.R` and `src/calc-isc-correlation.R` read directly from
`out/03_ISC_results/` and `data/isc_comparison.csv`, and write the polished
comparison figures to `out/06_figures/`. Run with `Rscript src/plot-stim-isc.R`
after step 4 has produced the `full` results for both stimuli.

### Regenerating `data/isc_comparison.csv`

`data/isc_comparison.csv` holds the Dmochowski et al. and Poulsen et al.
reference-study ISC/chance-level timeseries, digitized from a calibrated
Inkscape SVG trace of their published figures (`sources/traced_timeseries*.svg`),
plus this study's own ISC/chance timeseries on the matching BYD
segment (300–660s, the "Poulsen segment") for side-by-side plotting. It is
**not** produced by any step above — it only needs to be regenerated if the
digitized reference curves change, or if the BYD segment ISC is recomputed
(step 4, `--t0 296.667 --t1 669.667`) and you want the comparison CSV to reflect the
new run:

```bash
uv run python src/extract-timeseries.py sources/traced_timeseries_plain.svg \
    --n-samples 370 --n-grid 370 \
    --lab-data out/03_ISC_results/byd/segment/isc_component1_bywindow.npy \
    --lab-chance out/03_ISC_results/byd/segment/chance_comp1.npy \
    --output-csv data/isc_comparison.csv \
    --output-png out/06_figures/isc_comparison.png
```

This is not part of the regular pipeline run — it's a one-off/occasional step,
only needed when the reference-study digitization or the BYD segment ISC
changes. `calc-isc-correlation.R` itself loads the BYD segment `.npy` files
directly (not through this CSV), so `data/isc_comparison.csv` only needs to
stay in sync for the `lab_results`/`lab_chance_estimate` columns used by
`extract-timeseries.py`'s own comparison plot and correlation printout.

## Interactive exploration

`notebooks/isc_analysis.ipynb` still works for exploring the preprocessed
data, inspecting component topographies, and applying a spatial filter as a
live signal — none of which have scripted equivalents. Point it at the same
`out/02_preprocessed_eeg_data/` directory; see the README for how to launch it.

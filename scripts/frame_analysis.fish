#!/usr/bin/env fish

# Extracts per-frame luminance, loudness and amplitude features from a video,
# and additionally computes the average luminance difference (ALD) exactly as
# defined by Poulsen et al. (2017) so that the ISC-luminance analysis can be
# compared like-for-like against theirs.
#
# NOTE: signalstats YDIF is the mean ABSOLUTE difference of the Rec.601 luma
# plane at frame rate. Poulsen's ALD is the mean SQUARED difference of an
# equal-weight RGB greyscale, resampled to 1 Hz by per-second maximum, then
# Gaussian smoothed (variance 2.5 s^2). ffmpeg cannot produce that directly,
# so the ALD step shells out to src/compute_ald.py.

# Check for ffprobe
if not type -q ffprobe
    echo "Error: ffprobe is not installed or not in PATH."
    echo "Please install FFmpeg to use this script."
    exit 1
end

# video file as first argument
set video_file $argv[1]
set out_dir $argv[2]

if test -z $video_file
    echo "Usage: frame_analysis.fish <video_file> [out_dir]"
    echo "Extracts per-frame luminance, loudness, amplitude and Poulsen-style ALD features."
    echo "Outputs CSV files to the specified out_dir or the video's directory if not provided."
    exit 1
end

# resolve the video to an absolute path before any cd, so the ALD step still
# finds it once we have moved into out_dir
set video_file (realpath $video_file)

if test -z $out_dir
    set out_dir (dirname $video_file)
else if test -n $out_dir
    mkdir -p $out_dir
    cd $out_dir
end
set out_dir (realpath $out_dir)

set out_base (basename $video_file | sed 's/\.[^.]*$//')

echo "Processing video file: $video_file"
echo "Output directory: $out_dir"
echo "Analysing luminance..."
ffprobe -f lavfi -i "movie=$video_file,signalstats" \
        -show_entries frame=pkt_pts_time:frame_tags=lavfi.signalstats.YAVG,lavfi.signalstats.YMIN,lavfi.signalstats.YMAX,lavfi.signalstats.YDIF \
        -of csv=p=1 > "$out_dir/$out_base-luminance.csv" 2>/dev/null
echo "Analysing loudness..."
ffprobe -f lavfi -i "amovie=$video_file,pan=mono|c0=FL,ebur128=metadata=1" \
        -show_entries frame=pkt_pts_time:frame_tags=lavfi.r128.M \
        -of csv=p=1 > "$out_dir/$out_base-loudness(EBUR128,LUFS).csv" 2>/dev/null
echo "Analysing amplitude..."
ffprobe -f lavfi -i "amovie=$video_file,pan=mono|c0=FL,astats=metadata=1:reset=1" \
        -show_entries frame=pkt_pts_time:frame_tags=lavfi.astats.Overall.RMS_level,lavfi.astats.Overall.Peak_level \
        -of csv=p=1 > "$out_dir/$out_base-amplitude.csv" 2>/dev/null

# --- Poulsen et al. (2017) average luminance difference -------------------
# --width 320 downscales before differencing. ALD is a mean over pixels, so it
# is close to scale-invariant, and this makes a 20-minute film take seconds
# rather than minutes. Drop the flag for the full-resolution value.
echo "Analysing ALD (Poulsen et al. definition)..."
set script_dir (dirname (status --current-filename))
set ald_script "$script_dir/../src/compute_ald.py"
if not test -f $ald_script
    set ald_script "$script_dir/compute_ald.py"
end

if test -f $ald_script
    if type -q uv
        uv run python $ald_script $video_file -o "$out_dir/$out_base-ald.csv" --width 320
    else
        python3 $ald_script $video_file -o "$out_dir/$out_base-ald.csv" --width 320
    end
else
    echo "  WARNING: compute_ald.py not found next to this script or in ../src/; skipping ALD."
end

echo "Analysis complete. Output files:"
echo " - $out_dir/$out_base-luminance.csv"
echo " - $out_dir/$out_base-loudness(EBUR128,LUFS).csv"
echo " - $out_dir/$out_base-amplitude.csv"
echo " - $out_dir/$out_base-ald.csv"
echo "You can now run 'uv run python src/composite-stimuli-features.py' to combine and visualize these features."

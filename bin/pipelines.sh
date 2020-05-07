#!/bin/bash
# MagellanMapper pipelines script
# Author: David Young 2017, 2020

HELP="
Run MagellanMapper pipelines. Choose various pathways from simple viewing to 
stitching and full volumetric image detection.

Note that currently not all options are settable through at 
command-line and will need to be set manually in the script 
instead.

Arguments:
  -h: Show help documentation.
  -i [path]: Set image path.
  -j [path]: Path to custom JAVA_HOME for ImageJ/Fiji.
  -a [path]: Set AWS S3 path (excluding s://)
  -p [pipeline]: Set pipeline, which takes precedence over individual 
    pathways.
  -r [profile]: Set microscope profile. Give multiple times for each 
    desired channel.
  -l [channel]: Set channel.
  -m [ext]: Compression format extension; defaults to zst.
  -o [path]: Path to output file to send in notification and to S3.
  -n [URL]: Slack notification URL.
  -c: Server clean-up, including upload, notification, and 
    poweroff if appropriate.
  -z [x,y,z]: Size in x,y,z order.
  -e [x,y,z]: Resolutions in x,y,z order for metadata.
  -g [mag]: Lens objective magnification for metadata.
  -s [xy|xz|yz]: Planar surface orientation for transformation, taking
    the original orientation as xy.
  -u [zoom]: Lens objective zoom for metadata.
"


####################################
# REPLACE WITH SETTINGS FOR YOUR IMAGE

# full path to original image, assumed to be within a directory 
# that names the experiment and also used for S3; eg 
# "/data/exp_yyyy-mm-dd/WT-cortex.czi"
IMG="/path/to/your/image"

# lens objective settings
# TODO: option to gather automatically by importing metadata from
# original file, though can take awhile for large files
resolutions="0.913,0.913,4.935"
magnification="5.0"
zoom="1.0"

# parent path of image directory in cloud such as AWS S3, excluding the 
# protocol ("s3://"); eg "MyName/ClearingExps"
# NOTE: the image's directory on S3 is assumed to match that of IMG, 
# so that for the above example IMG, the image on S3 would be in 
# s3://$S3_DIR/exp_yyyy-mm-dd
S3_DIR=""

# Microscope profiles such as "lightsheet", "2p20x", or "lightsheetv02", 
# including modifiers such as "lightsheet_contrast" or 
# "lightsheet_contrast_cytoplasm" affect detections and visualization. Multiple 
# profiles can also be given for multiple channels, such as 
# "lightsheet lightsheet_cytoplasm" for a nuclear marker in channel 0 
# and cytoplasmic marker in channel 1.
microscope=()

# Grouped pathways to follow typical pipelines
PIPELINES=("gui" "full" "detection" "transformation" "download" "stitching")
pipeline="gui"


# OPTIONAL: curate specific pathway(s)

# Choose whether to show GUI, in which case rest of pathways will be 
# ignored; replace ROI offset/size with desired coordinates/dimensions 
# in x,y,z format
gui=0
offset=30,30,8
size=70,70,10

# Series/tile to load; use 0 for fully stitched images
series=0

# Choose stitch pathway type, or "" for none
STITCH_PATHWAYS=("stitching" "bigstitcher")
stitch_pathway=""

# Choose rescale pathway type, or "" for none
TRANSFORM_PATHWAYS=("rescale" "resize")
transform_pathway=""
scale="0.05" # rescaling factor
plane="" # xy, yz, zy, or leave empty to default to xy
animation="" # gif or mp4

# Choose whole image processing type, or "" for none
WHOLE_IMG_PROCS=("process")
whole_img_proc=""

# Choose whether to upload resulting files to AWS S3
UPLOAD_TYPES=("none" "all" "pathways_specific")
upload="${UPLOAD_TYPES[0]}"

# Path to file with nohup output, to upload if specified
output_path=""

# Paths to stats output files
output_stats_paths=()

# Slack notification URL, to post when done
url_notify=""
summary_msg=()

# Server clean-up, to perform post-processing tasks including 
# upload and shutdown
clean_up=0

# Java home for ImageJ/Fiji
java_home=""

# Supported compression formats
COMPRESSION_EXTS=("tar.zst" "zip")
compression="${COMPRESSION_EXTS[0]}"

# Image channels to process, defaulting to 0 but including all channels 
# if multiple microscope profiles are given
channel=0

# MagellanMapper filenames
image5d_npz=""
info_npz=""
proc_npz=""

############################################
# Check for existing files in MagellanMapper Numpy format
# Globals:
#   $OUT_DIR
# Arguments:
#   1: Filename style, passed to setup_clrbrain_filenames
# Returns:
#   None
############################################
get_image_files() {
  setup_clrbrain_filenames "$clr_img_base" "$1"
  echo -n "Looking for ${image5d_npz}..."
  if [[ ! -e "$image5d_npz" ]]; then
    # Get stitched image files from S3
    start=$SECONDS
    name="${image5d_npz%.*}"
    name="$(basename "$name").${compression}"
    echo "could not find locally, attempting to download $name from S3..."
    mkdir "$OUT_DIR"
    get_compressed_file "${s3_exp_path}/${name}" "$OUT_DIR"
    if [[ "$?" -eq 0 ]]; then
      # try getting individual .npz files if archive not present
      echo -n "Could not find compressed files, attempting to download "
      echo "uncompressed files..."
      for npz in "$image5d_npz" "$info_npz"; do
        echo "...attempting to download ${npz}..."
        aws s3 cp "${s3_exp_path}/$(basename "$npz")" "$OUT_DIR"
      done
    fi
    summary_msg+=(
      "MagellanMapper image download and decompression time: $((SECONDS - start)) s")
  else
    echo "found"
  fi
}

############################################
# Download compressed file if available.
# Globals:
#   COMPRESSION_EXTS
# Arguments:
#   1: Path to check on S3, prioritizing the given extension 
#    if it is in COMPRESSION_EXTS. If extension is not 
#    in this array, the original path will be checked last.
#   2: output local directory.
# Returns:
#   1 if the file or corresponding compressed file was 
#   downloaded; 0 if otherwise.
############################################
get_compressed_file() {
  # accommodate paths with multiple extensions (eg .tar.zst) by assuming
  # that anything after the first period in the basename is the extension
  local basename ext path_base paths is_compression name out_path
  basename="$(basename "$1")"
  ext="${basename#*.}"
  path_base="$(dirname "$1")/${basename%%.*}"

  # make array of possible paths
  is_compression=0
  paths=()
  for e in "${COMPRESSION_EXTS[@]}"; do
    if [[ "$ext" = "$e" ]]; then
      # prioritize compression type if given as ext
      is_compression=1
      paths+=("$1")
    fi
  done
  for e in "${COMPRESSION_EXTS[@]}"; do
    if [[ $is_compression -eq 0 || "$e" != "$ext" ]]; then
      # add name with extension if not yet added
      paths+=("${path_base}.${e}")
    fi
  done
  # append original file if not already added
  if [[ $is_compression -eq 0 ]]; then paths+=("$1"); fi
  
  for path in "${paths[@]}"; do
    # attempt to download and, if necessary, extract file if not present
    name="$(basename "$path")"
    out_path="${2}/${name}"
    if [[ ! -f "$out_path" ]]; then
      echo "Checking for $path on S3..."
      aws s3 cp "$path" "$2"
    fi
    if [[ -f "$out_path" ]]; then
      # decompress based on compression type
      cd "$2"
      if [[ "$name" =~ .*\."${COMPRESSION_EXTS[0]}" ]]; then # .tar.zstd
        pzstd -dc "$name" | tar xvf - 
      elif [[ "$name" =~ .*\."${COMPRESSION_EXTS[1]}" ]]; then # .zip
        unzip -u "$name"
      fi
      cd -
      return 1
    fi
  done
  return 0
}

############################################
# Compress and upload files to S3.
# Globals:
#   COMPRESSION_EXTS
# Arguments:
#   1: output base path without extension, where the compressed 
#    file will be output to the directory path, and the 
#    basename will be used as the basis for the compressed 
#    filename.
#   2: extension of compression format.
#   3...: remaining arguments are paths of files to be 
#    compressed.
# Returns:
#   None
############################################
compress_upload() {
  local args=("$@")
  local dir_path
  dir_path="$(dirname "${args[0]}")"
  local base_path
  base_path="$(basename "${args[0]}")"
  echo "$dir_path $base_path ${args[*]:2}"
  local compression="${args[1]}"
  local paths=()
  for path in "${args[@]:2}"; do
    paths+=("$(basename "$path")")
  done
  local out_path=""
  cd "$dir_path"
  echo "Compressing ${paths[*]} to $compression format..."
  case "$compression" in
     "${COMPRESSION_EXTS[0]}") # zstd
       out_path="${base_path}.${compression}"
       tar cf - "${paths[@]}" | pzstd > "$out_path"
       ;;
     "${COMPRESSION_EXTS[1]}") # zip
       out_path="${base_path}.${compression}"
       zip -R "$out_path" "${paths[@]}"
       ;;
  esac
  echo "Uploading ${out_path} to ${s3_exp_path}/..."
  aws s3 cp "$out_path" "${s3_exp_path}/"
  cd -
  return 1
}

############################################
# Set up paths for MagellanMapper formatted files.
# Globals:
#   image5d_npz
#   info_npz
#   proc_npz
# Arguments:
#   1: Base path from which to construct the filenames, typically
#      a full path without extension or series information.
#   2: "series" to use the pre-v1.2 style naming including series string;
#      defaults to use v1.2+ style naming.
# Returns:
#   None
############################################
setup_clrbrain_filenames() {
  if [[ "$2" = "series" ]]; then
    # series-based naming (pre-v1.2)
    local series_filled
    series_filled="$(printf %05d $series)"
    local npz_img_base="${1}_${series_filled}"
    image5d_npz="${npz_img_base}_image5d.npz"
    info_npz="${npz_img_base}_info.npz"
    proc_npz="${npz_img_base}_info_proc.npz"
  else
    # v1.2+ style naming, which does not have the series string and
    # and changes the file suffixes
    image5d_npz="${1}_image5d.npy"
    info_npz="${1}_meta.npz"
    proc_npz="${1}_blobs.npz"
  fi
}

############################################
# Wrapper method for uploading images
# Globals:
#   None
# Arguments:
#   Base path to pass to setup_clrbrain_filenames.
# Returns:
#   None
############################################
upload_images() {
  setup_clrbrain_filenames "$1"
  local args=("${image5d_npz%.*}" "$compression" "$image5d_npz" "$info_npz")
  compress_upload "${args[@]}"
}



####################################
# Script setup
START_TIME=$SECONDS

# override pathway settings with user arguments
OPTIND=1
while getopts hi:a:p:o:n:cz:m:j:r:l:e:g:u:s: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    i)
      IMG="$OPTARG"
      echo "Set image path to $IMG"
      ;;
    a)
      S3_DIR="$OPTARG"
      echo "Set AWS S3 directory to $S3_DIR"
      ;;
    p)
      pipeline="$OPTARG"
      echo "Set pipeline to $pipeline"
      ;;
    m)
      compression="$OPTARG"
      echo "Set compression format to $compression"
      ;;
    o)
      output_path="$OPTARG"
      echo "Set output path to $output_path"
      ;;
    n)
      url_notify="$OPTARG"
      echo "Set Slack notification URL to $url_notify"
      ;;
    c)
      clean_up=1
      echo "Set to perform server clean-up tasks once done"
      ;;
    z)
      size="$OPTARG"
      echo "Set size to $size"
      ;;
    j)
      java_home="$OPTARG"
      echo "Set JAVA_HOME for ImageJ/Fiji to $java_home"
      ;;
    r)
      microscope+=("$OPTARG")
      echo "Added $OPTARG to microscope profile"
      ;;
    l)
      channel="$OPTARG"
      echo "Set channel to $channel"
      ;;
    e)
      resolutions="$OPTARG"
      echo "Set resolutions to $resolutions"
      ;;
    g)
      magnification="$OPTARG"
      echo "Set magnification to $magnification"
      ;;
    s)
      plane="$OPTARG"
      echo "Set planar suface to plane"
      ;;
    u)
      zoom="$OPTARG"
      echo "Set zoom to $zoom"
      ;;
    :)
      echo "Option -$OPTARG requires an argument"
      exit 1
      ;;
    --)
      ;;
    *)
      echo "$HELP" >&2
      exit 1
      ;;
  esac
done
readonly IMG
readonly S3_DIR

# pass arguments after "--" to another script if necessary
shift "$((OPTIND-1))"
EXTRA_ARGS=("$@")


# Parsing names from your image path
OUT_DIR="$(dirname "$IMG")"
EXP="$(basename "$OUT_DIR")"
NAME="$(basename "$IMG")"
EXT="${IMG##*.}"
s3_exp_path=s3://"${S3_DIR}/${EXP}"

num_mic_profiles="${#microscope[@]}"
if [[ $num_mic_profiles -eq 0 ]]; then
  # default to single lightsheet profile
  microscope=("lightsheet")
elif [[ $num_mic_profiles -gt 1 ]]; then
  # if multiple profiles are given, include all channels for detections, 
  # overwriting any command-line arg
  echo "Changing channels to all since multiple microscope profiles are set"
  channel=-1
fi

# run from script's parent directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
BASE_DIR="$PWD"

# set pat combinations for common grouped pathways
if [[ "$pipeline" = "${PIPELINES[0]}" ]]; then
  # gui pathway
  gui=1
elif [[ "$pipeline" = "${PIPELINES[1]}" ]]; then
  # full, including stitching, transformation, and processing
  gui=0
  stitch_pathway="${STITCH_PATHWAYS[1]}"
  transform_pathway="${TRANSFORM_PATHWAYS[1]}"
  whole_img_proc="${WHOLE_IMG_PROCS[0]}"
  upload="${UPLOAD_TYPES[1]}"
elif [[ "$pipeline" = "${PIPELINES[2]}" ]]; then
  # cell detection only
  gui=0
  stitch_pathway=""
  transform_pathway=""
  whole_img_proc="${WHOLE_IMG_PROCS[0]}"
  upload="${UPLOAD_TYPES[2]}"
elif [[ "$pipeline" = "${PIPELINES[3]}" ]]; then
  # transformation only
  gui=0
  stitch_pathway=""
  transform_pathway="${TRANSFORM_PATHWAYS[1]}"
  whole_img_proc=""
  upload="${UPLOAD_TYPES[2]}"
elif [[ "$pipeline" = "${PIPELINES[4]}" ]]; then
  # download only
  gui=0
  # will download npz files correponding to IMG by default
elif [[ "$pipeline" = "${PIPELINES[5]}" ]]; then
  # stitching only
  gui=0
  stitch_pathway="${STITCH_PATHWAYS[1]}"
  transform_pathway=""
  whole_img_proc=""
  upload="${UPLOAD_TYPES[2]}"
fi

if [[ "$S3_DIR" = "" ]]; then
  echo "Unable to upload to S3 as S3 directory is not set"
  upload="${UPLOAD_TYPES[0]}"
fi

####################################
# Display region of interest in main GUI and exit when GUI is closed

if [[ $gui -eq 1 ]]; then
  # Run MagellanMapper GUI, importing the image into Numpy-based format that 
  # MagellanMapper can read if not available. A few additional scenarios are
  # also shown, currently commented out. The script will exit after 
  # displaying the GUI.

  # Import raw image stack into Numpy array if it doesn't exist already
  #python -u -m magmap.io.cli --img "$IMG" --channel 0 --proc import_only

  # Load image and set ROI
  ./run.py --img "$IMG" --offset "$offset" --size "$size" --savefig pdf \
    --microscope "${microscope[@]}"
  
  exit 0
fi


####################################
# Stitching Pipeline

if [[ "$stitch_pathway" != "" && ! -e "$IMG" ]]; then
  # Get large, unstitched image file from cloud, where the fused (all 
  # illuminators merged) image is used for the Stitching pathway, and 
  # the unfused, original image is used for the BigStitcher pathway
  start=$SECONDS
  mkdir "$OUT_DIR"
  echo "Downloading original image from S3..."
  get_compressed_file "${s3_exp_path}/${NAME}" "$OUT_DIR"
  summary_msg+=(
    "Original image download and decompression time: $((SECONDS - start)) s")
fi

clr_img="$IMG"
start_stitching=$SECONDS
if [[ "$stitch_pathway" = "${STITCH_PATHWAYS[0]}" ]]; then
  # ALTERNATIVE 1: Stitching plugin (old)
  
  OUT_NAME_BASE="${NAME%.*}_stitched"
  TIFF_DIR="${OUT_DIR}/${OUT_NAME_BASE}"
  
  # Replace the tile parameters with your image's setup; set up tile 
  # configuration manually and compute alignment refinement
  python -m stitch.tile_config --img "$NAME" --target_dir "$OUT_DIR" \
    --cols 6 --rows 7 --size 1920,1920,1000 --overlap 0.1 \
    --directionality bi --start_direction right
  bin/stitch.sh -f "$IMG" -o "$TIFF_DIR" -s "stitching" -w 0 -j "$java_home"
  
  # Before the next steps, please manually check alignments to ensure that they 
  # fit properly, especially since unregistered tiles may be shifted to 
  # (0, 0, 0)
  bin/stitch.sh -f "$IMG" -o "$TIFF_DIR" -s "stitching" -w 1 -j "$java_home"
  python -u -m magmap.io.cli --img "$TIFF_DIR" --res "$resolutions" \
    --mag "$magnification" --zoom "$zoom" -v --channel 0 --proc import_only
  clr_img="${OUT_DIR}/${OUT_NAME_BASE}.${EXT}"
  
elif [[ "$stitch_pathway" = "${STITCH_PATHWAYS[1]}" ]]; then
  # ALTERNATIVE 2: BigStitcher plugin
  
  OUT_NAME_BASE="${NAME%.*}_bigstitched"
  start=$SECONDS
  stitch_args=()
  if [[ -n "$java_home" ]]; then
    stitch_args+=(-j "$java_home")
  fi
  
  # Import file into BigStitcher HDF5 format (warning: large file, just 
  # under size of original file) and find alignments
  bin/stitch.sh -f "$IMG" -s "bigstitcher" -w 0 "${stitch_args[@]}"
  
  # notify user via Slack and open ImageJ/Fiji for review, which will 
  # also keep script from continuing until user closes ImageJ/Fiji 
  # after review
  msg="Stitching completed for $NAME, now awaiting your alignment review"
  if [[ "$url_notify" != "" ]]; then
    python -u -m magmap.io.notify --notify "$url_notify" "$msg"
  fi
  echo "=================================="
  echo "$msg"
  summary_msg+=("Stitching import and alignment time: $((SECONDS - start)) s")
  bin/stitch.sh -s "none" "${stitch_args[@]}"
  
  # Fuse image for each channel
  start=$SECONDS
  bin/stitch.sh -f "$IMG" -s "bigstitcher" -w 1 "${stitch_args[@]}"
  summary_msg+=("Stitching fusion time: $((SECONDS - start)) s")
  
  # Rename output file(s)
  FUSED="fused_tp_0"
  for f in "${OUT_DIR}/${FUSED}"*.tif; do
    mv "$f" "${f/$FUSED/$OUT_NAME_BASE}";
  done
  
  # Import stacked TIFF file(s) into Numpy arrays for MagellanMapper
  start=$SECONDS
  python -u -m magmap.io.cli --img "${OUT_DIR}/${OUT_NAME_BASE}.tiff" \
    --res "$resolutions" --mag "$magnification" --zoom "$zoom" -v \
    --proc import_only
  summary_msg+=("Stitched file import time: $((SECONDS - start)) s")
  clr_img="${OUT_DIR}/${OUT_NAME_BASE}.${EXT}"
fi
summary_msg+=(
  "Total stitching time (including waiting): $((SECONDS - start_stitching)) s")
clr_img_base="${clr_img%.*}"

if [[ "$pipeline" = "${PIPELINES[1]}" \
  || "$pipeline" = "${PIPELINES[5]}" ]]; then
  if [[ "$upload" != "${UPLOAD_TYPES[0]}" ]]; then
    # upload stitched image for full and stitching pipelines
    start=$SECONDS
    upload_images "$clr_img_base"
    summary_msg+=(
      "Stitched image compression and upload time: $((SECONDS - start)) s")
  fi
fi

# At this point, you can delete the TIFF dir/image since it has been 
# exported into a Numpy-based format for loading into MagellanMapper

# get local image file or download from cloud using v1.2+ style naming
get_image_files ""
if [[ ! -e "$image5d_npz" ]]; then
  # fall back to pre-v1.2 naming
  get_image_files series
fi

# output size in KB in cross-platform way
summary_msg+=(
  "Main image file size (approx): $(du -k "$image5d_npz" | cut -f1) KB")


####################################
# Transform/Resize Image Pipeline

if [[ "$transform_pathway" != "" ]]; then
  start=$SECONDS
  img_transformed=""
  if [[ "$transform_pathway" = "${TRANSFORM_PATHWAYS[0]}" ]]; then
    if [[ "$plane" != "" ]]; then
      # Both rescale and transform an image from z-axis (xy plane)
      # to x-axis (yz plane) orientation
      python -u -m magmap.io.cli --img "$clr_img" --proc transform \
        --transform rescale=${scale} --plane "$plane"
      img_transformed="${clr_img_base}_plane${plane}_scale${scale}.${EXT}"
    else
      # Rescale an image to downsample by the scale factor only
      python -u -m magmap.io.cli --img "$clr_img" --proc transform \
        --transform rescale=${scale}
      img_transformed="${clr_img_base}_scale${scale}.${EXT}"
    fi
  elif [[ "$transform_pathway" = "${TRANSFORM_PATHWAYS[1]}" ]]; then
    # Resize to a set size given by a registration profile, with size 
    # specified by register profile, which needs to be passed as 
    # --reg_file [name] in EXTRA_ARGS, and -z flag to find output name
    python -u -m magmap.io.cli --img "$clr_img" --proc transform \
      "${EXTRA_ARGS[@]}"
    img_transformed="${clr_img_base}_resized(${size}).${EXT}"
  fi
  
  if [[ "$animation" != "" ]]; then
    # Export transformed image to an animated GIF or MP4 video 
    # (requires ImageMagick)
    python -u -m magmap.io.cli --img "$img_transformed" --proc animated \
      --interval 5 --transform rescale=1.0 --savefig "$animation"
  fi
  
  if [[ "$upload" != "${UPLOAD_TYPES[0]}" ]]; then
    # zip and upload transformed files to S3
    base_path="${img_transformed%.*}"
    upload_images "$base_path"
  fi
  
  summary_msg+=("transformation time: $((SECONDS - start)) s")
fi


####################################
# Whole Image Blob Detections Pipeline

if [[ "$whole_img_proc" != "" ]]; then
  start=$SECONDS
  # Process an entire image locally the given channel(s), chunking the 
  # image into multiple smaller stacks to minimize RAM usage and 
  # further chunking to run by multiprocessing for efficiency
  python -u -m magmap.io.cli --img "$clr_img" --proc detect \
    --channel $channel --microscope "${microscope[@]}" "${EXTRA_ARGS[@]}"
  
  if [[ "$upload" != "${UPLOAD_TYPES[0]}" ]]; then
    # upload processed fils to S3
    setup_clrbrain_filenames "$clr_img_base"
    args=("${proc_npz%.*}" "$compression" "$proc_npz")
    compress_upload "${args[@]}"
  fi
  summary_msg+=("Detections and upload time: $((SECONDS - start)) s")
  output_stats_paths+=(
    "blob_ratios_means.csv" "blob_ratios.csv" "stack_detection_times.csv")
fi


####################################
# Post-Processing

if [[ -n "$output_path" ]]; then
  # include nohup output script for upload
  #attach="`sed -e "s/&/&amp;/" -e "s/</&lt;/" -e "s/>/&g;/" $output_path`"
  output_stats_paths+=("$output_path")
fi

if [[ -n "$S3_DIR" && "${#output_stats_paths[@]}" -gt 0 ]]; then
  # compress and upload output stat files
  compress_upload "$(basename "${clr_img_base}"_stats)" \
    "zip" "${output_stats_paths[@]}"
fi

msg="Total time elapsed for MagellanMapper pipeline: $((SECONDS - START_TIME)) s"
summary_msg+=("$msg")
echo "$msg"

if [[ "$url_notify" != "" ]]; then
  # post-processing notification to Slack
  summary_msg=(
    "MagellanMapper \"$pipeline\" pipeline for $NAME completed" "${summary_msg[@]}")
  msg=$(printf "%s\n" "${summary_msg[@]}")
  attach=""
  if [[ "$output_path" != "" ]]; then
    attach="$output_path"
  fi
  python -u -m magmap.io.notify --notify "$url_notify" "$msg" "$attach"
fi

if [[ $clean_up -eq 1 ]]; then
  # Server Clean-Up
  echo "Completed MagellanMapper pipelines, shutting down server..."
  sudo poweroff
fi

exit 0

#!/bin/bash
# Clrbrain pipelines and sample run commands
# Author: David Young 2017, 2018

HELP="
Run Clrbrain pipelines. Choose various pathways from command-line, 
or copy this script to select the commands you desire. You can 
also find examples of various commands to call directly.

Note that currently not all options are settable through at 
command-line and will need to be set manually in the script 
instead.

Arguments:
    -h: Show help documentation.
    -i [path]: Set image path.
    -a [path]: Set AWS S3 path (excluding s://)
    -p [pipeline]: Set pipeline, which takes precedence over individual 
        pathways.
    -s [pathway]: Set stitching pathway.
    -t [pathway]: Set transposition pathway.
    -w [pathway]: Set whole image processing pathway.
    -o [path]: Path to output file to send in notification and to S3.
    -n [URL]: Slack notification URL.
    -c: Server clean-up, including upload, notification, and 
        poweroff if appropriate.
    -z [size]: Size in x,y,z format.
"


####################################
# REPLACE WITH SETTINGS FOR YOUR IMAGE

# full path to original image, assumed to be within a directory 
# that names the experiment and also used for S3; eg 
# "/data/exp_yyyy-mm-dd/WT-cortex.czi"
IMG="/path/to/your/image"

# parent path of image directory in cloud such as AWS S3, excluding the 
# protocol ("s3://"); eg "MyName/ClearingExps"
# NOTE: the image's directory on S3 is assumed to match that of IMG, 
# so that for the above example IMG, the image on S3 would be in 
# s3://$S3_DIR/exp_yyyy-mm-dd
S3_DIR=""

# Replace microscope type with available profiles, such as "lightsheet", 
# "2p_20x", or "lightsheet_v02", or with modifiers, such as 
# "lightsheet_contrast" or "lightsheet_contrast_cytoplasm". Multiple 
# profiles can also be given for multiple channels, such as 
# "lightsheet lightsheet_cytoplasm" for a nuclear marker in channel 0 
# and cytoplasmic marker in channel 1
MICROSCOPE="lightsheet"

# Grouped pathways to follow typical pipelines
PIPELINES=("gui" "full" "process_only" "transpose_only" "download_only")
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
TRANSPOSE_PATHWAYS=("rescale", "resize")
transpose_pathway=""
scale="0.05" # rescaling factor
plane="" # xy, yz, zy, or leave empty
animation="" # gif or mp4

# Choose whole image processing type, or "" for none
WHOLE_IMG_PROCS=("process")
whole_img_proc=""

# Choose whether to upload resulting files to AWS S3
UPLOAD_TYPES=("none", "all", "pathways_specific")
upload="${UPLOAD_TYPES[0]}"

# Path to output file, to upload if specified
output_path=""

# Slack notification URL, to post when done
url_notify=""

# Server clean-up, to perform post-processing tasks including 
# upload and shutdown
clean_up=0

# Supported compression formats
COMPRESSION_EXTS=("zst" "zip")

############################################
# Download compressed file if available.
# Globals:
#   COMPRESSION_EXTS
# Arguments:
#   1: Path to check on S3, with corresponding formats in 
#      COMPRESSION_EXTS checked as well, prioritizing any 
#      given format already in that array.
#   2: output local directory.
# Returns:
#   1 if the file or corresponding compressed file was 
#   downloaded; 0 if otherwise.
############################################
get_compressed_file() {
    ext="${1##*.}"
    path_base="${1%.*}"
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
        echo "Checking for $path on S3..."
        aws s3 cp "$path" "$2"
        out_path="${2}/$(basename $path)"
        if [[ -e "$out_path" ]]; then
            # decompress based on compression type
            case $ext in
               "${COMPRESSION_EXTS[0]}") # zstd
                   pzstd -dc "$out_path" | tar xvf - 
                   ;;
               "${COMPRESSION_EXTS[1]}") # zip
                   cd "$OUT_DIR"
                   unzip -u "$(basename $path)"
                   cd -
                   ;;
            esac
            return 1
        fi
    done
    return 0
}

####################################
# Script setup

# override pathway settings with user arguments
OPTIND=1
while getopts hi:a:p:s:t:w:o:n:cz: opt; do
    case $opt in
        h)  echo "$HELP"
            exit 0
            ;;
        i)  IMG="$OPTARG"
            echo "Set image path to $IMG"
            ;;
        a)  S3_DIR="$OPTARG"
            echo "Set AWS S3 directory to $S3_DIR"
            ;;
        p)  pipeline="$OPTARG"
            echo "Set pipeline to $pipeline"
            ;;
        s)  stitch_pathway="$OPTARG"
            echo "Set stitch pathway to $stitch_pathway"
            ;;
        t)  transpose_pathway="$OPTARG"
            echo "Set transpose pathway to $transpose_pathway"
            ;;
        w)  whole_img_proc="$OPTARG"
            echo "Set whole img proc to $whole_img_proc"
            ;;
        o)  output_path="$OPTARG"
            echo "Set output path to $output_path"
            ;;
        n)  url_notify="$OPTARG"
            echo "Set Slack notification URL to $url_notify"
            ;;
        c)  clean_up=1
            echo "Set to perform server clean-up tasks once done"
            ;;
        z)  size="$OPTARG"
            echo "Set size to $size"
            ;;
        :)  echo "Option -$OPTARG requires an argument"
            exit 1
            ;;
        --) ;;
    esac
done
readonly IMG
readonly S3_DIR

# pass arguments after "--" to another script if necessary
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"


# Parsing names from your image path
OUT_DIR="`dirname $IMG`"
EXP="`basename $OUT_DIR`"
NAME="`basename $IMG`"
IMG_PATH_BASE="${OUT_DIR}/${NAME%.*}"
EXT="${IMG##*.}"
s3_exp_path=s3://"${S3_DIR}/${EXP}"

# run from script's directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD

# set pat combinations for common grouped pathways
if [[ "$pipeline" = "${PIPELINES[0]}" ]]; then
    # gui pathway
    gui=1
elif [[ "$pipeline" = "${PIPELINES[1]}" ]]; then
    # full, including stitching, transposition, and processing
    gui=0
    stitch_pathway="${STITCH_PATHWAYS[1]}"
    transpose_pathway="${TRANSPOSE_PATHWAYS[1]}"
    whole_img_proc="${WHOLE_IMG_PROCS[0]}"
    upload="${UPLOAD_TYPES[1]}"
elif [[ "$pipeline" = "${PIPELINES[2]}" ]]; then
    # processing only
    gui=0
    stitch_pathway=""
    transpose_pathway=""
    whole_img_proc="${WHOLE_IMG_PROCS[0]}"
    upload="${UPLOAD_TYPES[2]}"
elif [[ "$pipeline" = "${PIPELINES[3]}" ]]; then
    # transposition only
    gui=0
    stitch_pathway=""
    transpose_pathway="${TRANSPOSE_PATHWAYS[1]}"
    whole_img_proc=""
    upload="${UPLOAD_TYPES[2]}"
elif [[ "$pipeline" = "${PIPELINES[4]}" ]]; then
    # download only
    gui=0
    # will download npz files correponding to IMG by default
fi

if [[ "$S3_DIR" = "" ]]; then
    echo "Unable to upload to S3 as S3 directory is not set"
    upload="${UPLOAD_TYPES[0]}"
fi

####################################
# Graphical display

if [[ $gui -eq 1 ]]; then
    # Run Clrbrain GUI, importing the image into Numpy-based format that 
    # Clrbrain can read if not available. A few additional scenarios are
    # also shown, currently commented out. The script will exit after 
    # displaying the GUI.

    # Import raw image stack into Numpy array if it doesn't exist already
    #python -u -m clrbrain.cli --img "$IMG" --channel 0 --proc importonly
    
    # Load ROI, starting at the given offset and ROI size
    ./run --img "$IMG" --channel 0 --offset $offset --size $size --savefig pdf --microscope "$MICROSCOPE"
    
    # Extract a single z-plane
    #python -u -m clrbrain.cli --img "$IMG" --proc extract --channel 0 --offset 0,0,0 -v --savefig jpeg --microscope "$MICROSCOPE"
    
    # Process a sub-stack and load it
    substack_offset=100,800,410
    substack_size=800,100,48
    #python -m clrbrain.cli --img "$IMG" --proc processing_mp --channel 0 -v --offset $substack_offset --size $substack_size --microscope "$MICROSCOPE"
    IMG_ROI="${IMG_PATH_BASE}_(${substack_offset})x(${substack_size}).${EXT}"
    #./run --img "$IMG_ROI" -v --channel 0 -v --proc load --offset $substack_offset --size $substack_size --savefig pdf --microscope "$MICROSCOPE"
    
    exit 0
fi


####################################
# Stitching Workflow

# Replace with your lens objective settings
# TODO: option to gather automatically by importing metadata from 
# original file, though can take awhile for large files
RESOLUTIONS="0.913,0.913,4.935"
MAGNIFICATION="5.0"
ZOOM="1.0"

if [[ "$stitch_pathway" != "" && ! -e "$IMG" ]]; then
    # Get large, unstitched image file from cloud, where the fused (all 
    # illuminators merged) image is used for the Stitching pathway, and 
    # the unfused, original image is used for the BigStitcher pathway
    mkdir "$OUT_DIR"
    echo "Downloading original image from S3..."
    get_compressed_file "${s3_exp_path}/${NAME}" "$OUT_DIR"
fi

out_name_base=""
clr_img="$IMG"
if [[ "$stitch_pathway" = "${STITCH_PATHWAYS[0]}" ]]; then
    # ALTERNATIVE 1: Stitching plugin (old)
    
    OUT_NAME_BASE="${NAME%.*}_stitched"
    TIFF_DIR="${OUT_DIR}/${OUT_NAME_BASE}"
    
    # Replace the tile parameters with your image's setup; set up tile 
    # configuration manually and compute alignment refinement
    python -m stitch.tile_config --img "$NAME" --target_dir "$OUT_DIR" --cols 6 --rows 7 --size 1920,1920,1000 --overlap 0.1 --directionality bi --start_direction right
    ./stitch.sh -f "$IMG" -o "$TIFF_DIR" -s "stitching" -w 0
    
    # Before the next steps, please manually check alignments to ensure that they 
    # fit properly, especially since unregistered tiles may be shifted to (0, 0, 0)
    ./stitch.sh -f "$IMG" -o "$TIFF_DIR" -s "stitching" -w 1
    python -u -m clrbrain.cli --img "$TIFF_DIR" --res "$RESOLUTIONS" --mag "$MAGNIFICATION" --zoom "$ZOOM" -v --channel 0 --proc importonly
    clr_img="${OUT_DIR}/${OUT_NAME_BASE}.${EXT}"
    
elif [[ "$stitch_pathway" = "${STITCH_PATHWAYS[1]}" ]]; then
    # ALTERNATIVE 2: BigStitcher plugin
    
    OUT_NAME_BASE="${NAME%.*}_bigstitched"
    
    # Import file into BigStitcher HDF5 format (warning: large file, just 
    # under size of original file) and find alignments
    ./stitch.sh -f "$IMG" -s "bigstitcher" -w 0
    
    # notify user via Slack and open ImageJ/Fiji for review, which will 
    # also keep script from continuing until user closes ImageJ/Fiji 
    # after review
    msg="Stitching completed for $IMG, now awaiting your alignment review"
    if [[ "$url_notify" != "" ]]; then
        python -u -m clrbrain.notify --notify "$url_notify" "$msg"
    fi
    echo "=================================="
    echo "$msg"
    ./stitch.sh -s "none"
    
    # Before writing stitched file, advise checking alignments; when 
    # satisfied, then run this fusion step
    ./stitch.sh -f "$IMG" -s "bigstitcher" -w 1
    
    # Rename output file(s):
    FUSED="fused_tp_0"
    for f in ${OUT_DIR}/${FUSED}*.tif; do mv $f ${f/$FUSED/$OUT_NAME_BASE}; done
    
    # Import stacked TIFF file(s) into Numpy arrays for Clrbrain
    python -u -m clrbrain.cli --img ""${OUT_DIR}/${OUT_NAME_BASE}.tiff"" --res "$RESOLUTIONS" --mag "$MAGNIFICATION" --zoom "$ZOOM" -v --proc importonly
    clr_img="${OUT_DIR}/${OUT_NAME_BASE}.${EXT}"
fi
clr_img_base="${clr_img%.*}"

# At this point, you can delete the TIFF dir/image since it has been 
# exported into a Numpy-based format for loading into Clrbrain


# Check for existing files in Clrbrain Numpy format
series_filled="$(printf %05d $series)"
npz_img_base="${clr_img_base}_${series_filled}"
image5d_npz="${npz_img_base}_image5d.npz"
info_npz="${npz_img_base}_info.npz"
echo -n "Looking for ${image5d_npz}..."
if [[ ! -e "$image5d_npz" ]]; then
    # Get stitched image files from S3
    name="$(basename $clr_img_base).${EXT}"
    echo "downloading $name from S3..."
    mkdir "$OUT_DIR"
    get_compressed_file "${s3_exp_path}/${name}" "$OUT_DIR"
    if [[ "$?" -eq 0 ]]; then
        # get individual .npz files if .zip not present
        echo -n "Could not find compressed files, attempting to download "
        echo "uncompressed files..."
        for npz in "$image5d_npz" "$info_npz"; do
            echo "...attempting to download ${npz}..."
            aws s3 cp "${s3_exp_path}/$(basename $npz)" "$OUT_DIR"
        done
    fi
else
    echo "found"
fi


####################################
# Transpose/Resize Image Workflow

if [[ "$transpose_pathway" != "" ]]; then
    img_transposed=""
    if [[ "$transpose_pathway" = "${TRANSPOSE_PATHWAYS[0]}" ]]; then
        if [[ "$plane" != "" ]]; then
            # Both rescale and transpose an image from z-axis (xy plane) 
            # to x-axis (yz plane) orientation
            python -u -m clrbrain.cli --img "$clr_img" --proc transpose --rescale ${scale} --plane "$plane"
            img_transposed="${clr_img_base}_plane${plane}_scale${scale}.${EXT}"
        else
            # Rescale an image to downsample by the scale factor only
            python -u -m clrbrain.cli --img "$clr_img" --proc transpose --rescale ${scale}
            img_transposed="${clr_img_base}_scale${scale}.${EXT}"
        fi
    elif [[ "$transpose_pathway" = "${TRANSPOSE_PATHWAYS[1]}" ]]; then
        # Resize to a set size given by a registration profile, with size 
        # specified by register profile, which needs to be passed as 
        # --reg_file [name] in EXTRA_ARGS, and -z flag to find output name
        python -u -m clrbrain.cli --img "$clr_img" --proc transpose $EXTRA_ARGS
        img_transposed="${clr_img_base}_resized(${size}).${EXT}"
    fi
    
    if [[ "$animation" != "" ]]; then
        # Export transposed image to an animated GIF or MP4 video (requires ImageMagick)
        python -u -m clrbrain.cli --img "$img_transposed" --proc animated --interval 5 --rescale 1.0 --savefig "$animation"
    fi
    
    if [[ "$upload" == "${UPLOAD_TYPES[2]}" ]]; then
        # zip and upload transposed files to S3
        base_path="${img_transposed%.*}"
        cd "$(dirname $base_path)"
        base_path="$(basename $base_path)"
        zip_name=${base_path}.zip
        zip -R "$zip_name" "${base_path}*"
        echo "uploading $zip_name to S3 at ${s3_exp_path}"
        aws s3 cp "$zip_name" "${s3_exp_path}/${zip_name}"
        cd -
    fi
    
fi


####################################
# Whole Image Processing Workflow

if [[ "$whole_img_proc" != "" ]]; then
    # Process an entire image locally on 1st channel, chunked into multiple 
    # smaller stacks to minimize RAM usage and multiprocessed for efficiency
    python -u -m clrbrain.cli --img "$clr_img" --proc processing_mp --channel 0 --microscope "$MICROSCOPE"
    
    if [[ "$upload" == "${UPLOAD_TYPES[2]}" ]]; then
        # upload processed fils to S3
        aws s3 cp $OUT_DIR "${s3_exp_path}" --recursive --exclude "*" --include "*proc.npz"
    fi
    
fi


####################################
# Post-Processing

if [[ "$url_notify" != "" ]]; then
    # post-processing notification to Slack
    msg="Clrbrain \"$pipeline\" pipeline for $IMG completed"
    attach=""
    if [[ "$output_path" != "" ]]; then
        attach="$output_path"
    fi
    python -u -m clrbrain.notify --notify "$url_notify" "$msg" "$attach"
fi

if [[ $clean_up -eq 1 ]]; then
    # Server Clean-Up
    
    if [[ "$output_path" != "" ]]; then
        # prepare tail of output file for notification and upload full file to S3
        #attach="`sed -e "s/&/&amp;/" -e "s/</&lt;/" -e "s/>/&g;/" $output_path`"
        name="`basename $output_path`"
        aws s3 cp "$output_path" "${s3_exp_path}/${name}"
    fi
    
    if [[ "$upload" == "${UPLOAD_TYPES[1]}" ]]; then
        # upload all resulting files to S3
        aws s3 cp $OUT_DIR "${s3_exp_path}" --recursive --exclude "*" --include "*.npz"
    fi
    
    echo "Finishing clean-up tasks, shutting down..."
    sudo poweroff
fi

exit 0
